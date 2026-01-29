"""Image preprocessing utilities for vision/multimodal requests.

Handles:
- Base64 data URL decoding
- HTTP/HTTPS image downloading
- Local file path loading
- Image validation with PIL
- Size limit enforcement
- SSRF protection (blocking private IPs)
- Automatic resizing based on available RAM (v3.1.0)
- Resize caching with TTL (v3.1.0)
"""

import base64
import re
import logging
import socket
import ipaddress
import os
import subprocess
from typing import Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
import asyncio

# S2: Image bomb protection - set PIL limit at module level
# This MUST be set before any PIL Image operations to prevent decompression bombs
# 50 megapixels = ~7071x7071 image = reasonable limit for vision models
# Duplicated from inference.py for safety (module load order not guaranteed)
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 50_000_000
except ImportError:
    pass  # PIL not available - will fail later if images are processed

logger = logging.getLogger(__name__)

# Configuration (from plan)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB per image
MAX_TOTAL_IMAGES = 5  # Max images per request
MAX_IMAGE_DIMENSIONS = (4096, 4096)  # Max width/height (Opus C3 fix - prevents image bombs)
URL_TIMEOUT_SECONDS = 10.0  # HTTP download timeout
INLINE_THRESHOLD = 10 * 1024 * 1024  # 10MB threshold - inline all images (no shmem needed for typical use)

# Auto-Resize Configuration (v3.1.0)
# RAM-based limits determined by testing (see docs/sessions/VISION-MEMORY-INVESTIGATION.md)
RAM_TIER_LIMITS = {
    128: {"default": 1024, "description": "128+ GB systems"},
    64:  {"default": 1024, "description": "64 GB systems"},
    32:  {"default": 768,  "description": "32 GB systems (MacBook Air)"},
    16:  {"default": 512,  "description": "16 GB systems (M4 Mini)"},
    8:   {"default": 512,  "description": "8 GB systems (not recommended)"},
}

# Resize cache configuration
RESIZE_CACHE_ENABLED = os.getenv("VISION_RESIZE_CACHE_ENABLED", "true").lower() == "true"
RESIZE_CACHE_SIZE = int(os.getenv("VISION_RESIZE_CACHE_SIZE", "20"))
RESIZE_CACHE_TTL = int(os.getenv("VISION_RESIZE_CACHE_TTL", "3600"))  # 1 hour

# Auto-resize configuration
AUTO_RESIZE_ENABLED = os.getenv("VISION_AUTO_RESIZE", "true").lower() == "true"
VISION_MAX_DIMENSION = None  # Will be set by detect_system_limits()

# SSRF Protection: Blocked IP ranges (Opus 4.5 recommendation)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),          # Private: Class A
    ipaddress.ip_network('172.16.0.0/12'),       # Private: Class B
    ipaddress.ip_network('192.168.0.0/16'),      # Private: Class C
    ipaddress.ip_network('127.0.0.0/8'),         # Localhost
    ipaddress.ip_network('169.254.0.0/16'),      # Link-local / Cloud metadata (AWS/GCP/Azure)
    ipaddress.ip_network('::1/128'),             # IPv6 localhost
    ipaddress.ip_network('fc00::/7'),            # IPv6 private
    ipaddress.ip_network('fe80::/10'),           # IPv6 link-local
]


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class ImageTooLargeError(ImageProcessingError):
    """Image exceeds size limit."""
    pass


class ImageDownloadError(ImageProcessingError):
    """Failed to download image from URL."""
    pass


class InvalidImageError(ImageProcessingError):
    """Image data is invalid or corrupted."""
    pass


# ============================================================================
# RAM Detection and Auto-Resize (v3.1.0)
# ============================================================================

def get_system_ram_gb() -> float:
    """Detect total system RAM in GB.

    Returns:
        Total RAM in GB

    Raises:
        RuntimeError: If RAM detection fails
    """
    try:
        # macOS
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True,
            text=True,
            check=True
        )
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024 ** 3)
        return ram_gb
    except Exception as e:
        logger.warning(f"Failed to detect RAM: {e}, assuming 16 GB")
        return 16.0  # Conservative default


def detect_system_limits() -> int:
    """Detect safe vision dimension limit based on system RAM.

    Returns:
        Maximum safe dimension in pixels
    """
    global VISION_MAX_DIMENSION

    # Check for manual override
    env_override = os.getenv("VISION_MAX_DIMENSION")
    if env_override:
        VISION_MAX_DIMENSION = int(env_override)
        logger.info(
            f"Vision max dimension: {VISION_MAX_DIMENSION}px "
            f"(manual override from VISION_MAX_DIMENSION)"
        )
        return VISION_MAX_DIMENSION

    # Auto-detect based on RAM
    system_ram_gb = get_system_ram_gb()

    # Find appropriate tier
    for ram_threshold in sorted(RAM_TIER_LIMITS.keys(), reverse=True):
        if system_ram_gb >= ram_threshold:
            limits = RAM_TIER_LIMITS[ram_threshold]
            VISION_MAX_DIMENSION = limits["default"]
            logger.info(
                f"System RAM: {system_ram_gb:.1f} GB detected "
                f"({limits['description']})"
            )
            logger.info(
                f"Vision max dimension: {VISION_MAX_DIMENSION}px "
                f"(auto-configured for safety)"
            )
            logger.info(
                f"Override with VISION_MAX_DIMENSION env var if needed"
            )
            return VISION_MAX_DIMENSION

    # Fallback for systems < 8 GB
    VISION_MAX_DIMENSION = 512
    logger.warning(
        f"System RAM: {system_ram_gb:.1f} GB is below recommended minimum (8 GB)"
    )
    logger.warning(
        f"Vision max dimension: {VISION_MAX_DIMENSION}px (conservative limit)"
    )
    return VISION_MAX_DIMENSION


def auto_resize_image(data: bytes, max_dim: int) -> Tuple[bytes, dict]:
    """Resize image if needed, preserving aspect ratio.

    Args:
        data: Original image bytes
        max_dim: Maximum dimension (width or height)

    Returns:
        Tuple of (resized_bytes, metadata)
            metadata contains:
                - original_dimensions: (width, height)
                - resized: bool
                - final_dimensions: (width, height)
                - reason: str or None

    Raises:
        InvalidImageError: If image cannot be processed
    """
    from PIL import Image
    import io

    try:
        # Open image
        img = Image.open(io.BytesIO(data))
        width, height = img.size

        metadata = {
            "original_dimensions": (width, height),
            "resized": False,
            "final_dimensions": (width, height),
            "reason": None
        }

        # Check if resize needed
        if width > max_dim or height > max_dim:
            # Calculate new dimensions (preserve aspect ratio)
            scale = min(max_dim / width, max_dim / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize with high-quality Lanczos resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to bytes
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=True)
            resized_data = output.getvalue()

            metadata.update({
                "resized": True,
                "final_dimensions": (new_width, new_height),
                "reason": f"Memory optimization (max {max_dim}px)"
            })

            logger.info(
                f"Auto-resized image: {width}x{height} → {new_width}x{new_height} "
                f"(reduction: {100 * (1 - len(resized_data)/len(data)):.1f}%)"
            )

            return resized_data, metadata

        return data, metadata

    except Exception as e:
        raise InvalidImageError(f"Failed to resize image: {e}")


# Initialize resize cache (if enabled)
_resize_cache = None
if RESIZE_CACHE_ENABLED:
    from .resize_cache import TimedResizeCache
    _resize_cache = TimedResizeCache(
        max_size=RESIZE_CACHE_SIZE,
        ttl_seconds=RESIZE_CACHE_TTL
    )


def get_resize_cache():
    """Get the global resize cache instance."""
    return _resize_cache


# ============================================================================


def decode_data_url(data_url: str) -> Tuple[bytes, str]:
    """
    Decode data URL format: data:image/jpeg;base64,/9j/4AAQ...

    Args:
        data_url: Data URL string

    Returns:
        Tuple of (image_bytes, format) e.g. (b'\\xff\\xd8...', 'jpeg')

    Raises:
        InvalidImageError: If data URL format is invalid
    """
    # Match data URL pattern: data:image/{format};base64,{data}
    match = re.match(r'^data:image/([a-zA-Z]+);base64,(.+)$', data_url)

    if not match:
        raise InvalidImageError(
            "Invalid data URL format. Expected: data:image/{format};base64,{data}"
        )

    format_str = match.group(1).lower()  # jpeg, png, webp, etc.
    base64_data = match.group(2)

    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        raise InvalidImageError(f"Failed to decode base64 data: {e}")

    # Validate size
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ImageTooLargeError(
            f"Image too large: {len(image_bytes)} bytes (max={MAX_IMAGE_SIZE})"
        )

    logger.debug(f"Decoded data URL: {len(image_bytes)} bytes, format={format_str}")
    return (image_bytes, format_str)


def is_safe_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL doesn't point to internal/private resources (SSRF protection).

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_safe, error_message)

    Security: Opus 4.5 Critical Fix C1
    - Blocks private IP ranges (10.x, 172.16.x, 192.168.x)
    - Blocks localhost (127.x)
    - Blocks link-local / cloud metadata (169.254.x)
    - Blocks IPv6 private ranges
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return (False, "URL missing hostname")

        # Resolve hostname to IP addresses (both IPv4 and IPv6)
        try:
            results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        except socket.gaierror as e:
            return (False, f"Cannot resolve hostname: {e}")

        # Check each resolved IP against blocked ranges
        for family, _, _, _, sockaddr in results:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)

                # Check against blocked ranges
                for blocked_range in BLOCKED_IP_RANGES:
                    if ip in blocked_range:
                        return (
                            False,
                            f"URL resolves to blocked IP range: {ip} in {blocked_range} "
                            f"(SSRF protection - private/internal network access blocked)"
                        )
            except ValueError:
                # Invalid IP address format
                continue

        return (True, None)

    except Exception as e:
        return (False, f"URL validation error: {e}")


def resolve_and_validate_url(url: str) -> Tuple[str, str, int]:
    """
    Resolve URL hostname and validate against SSRF, returning pinned IP and port.

    This function performs DNS resolution and SSRF validation in one step,
    returning the resolved IP for DNS pinning to prevent DNS rebinding attacks.

    S1 Security Fix (Opus review):
    - Prevents DNS rebinding attacks
    - Returns actual port for proper --resolve pinning
    - Handles IPv4-mapped IPv6 addresses
    - Validates scheme is http/https only

    Args:
        url: URL to resolve and validate

    Returns:
        Tuple of (hostname, pinned_ip, port) for use with curl --resolve

    Raises:
        InvalidImageError: If URL is unsafe or cannot be resolved
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise InvalidImageError("URL missing hostname")

    # S1 Fix (Opus): Validate scheme first
    if parsed.scheme not in ('http', 'https'):
        raise InvalidImageError(f"URL scheme must be http or https, got: {parsed.scheme}")

    # S1 Fix (Opus): Extract actual port
    if parsed.port:
        port = parsed.port
    elif parsed.scheme == 'https':
        port = 443
    else:
        port = 80

    # Resolve hostname to IP
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
    except socket.gaierror as e:
        raise InvalidImageError(f"Cannot resolve hostname {hostname}: {e}")

    if not results:
        raise InvalidImageError(f"No IP addresses found for hostname: {hostname}")

    # Get first resolved IP and validate it
    pinned_ip = results[0][4][0]

    try:
        ip = ipaddress.ip_address(pinned_ip)

        # S1 Fix (Opus): Handle IPv4-mapped IPv6 addresses (e.g., ::ffff:169.254.169.254)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped  # Extract and validate the IPv4 portion

        for blocked_range in BLOCKED_IP_RANGES:
            if ip in blocked_range:
                raise InvalidImageError(
                    f"URL resolves to blocked IP range: {pinned_ip} in {blocked_range} "
                    f"(SSRF protection - private/internal network access blocked)"
                )
    except ValueError as e:
        raise InvalidImageError(f"Invalid IP address format: {pinned_ip}: {e}")

    return (hostname, pinned_ip, port)


async def fetch_image(url: str) -> Tuple[bytes, str]:
    """
    Download image from HTTP/HTTPS URL.

    Args:
        url: Image URL (http:// or https://)

    Returns:
        Tuple of (image_bytes, format) e.g. (b'\\xff\\xd8...', 'jpeg')

    Raises:
        ImageDownloadError: If download fails
        ImageTooLargeError: If image exceeds size limit
        InvalidImageError: If URL scheme invalid
    """
    # Validate URL scheme
    if not url.startswith(('http://', 'https://')):
        raise InvalidImageError(f"Invalid URL scheme: {url}. Only http:// and https:// supported")

    # S1 SSRF Protection: Resolve DNS and validate IP, get pinned IP for DNS rebinding protection
    # This validates SSRF AND returns the IP to pin curl's DNS resolution
    hostname, pinned_ip, port = resolve_and_validate_url(url)

    try:
        # M5: Improved timeout handling with separate connect/total timeouts
        # Use asyncio subprocess to run curl with timeout
        # This avoids adding httpx/aiohttp dependencies
        #
        # S1 Security Fix (Opus review):
        # - Removed '-L' (no redirects) - prevents redirect-based SSRF bypass
        # - Added '--max-redirs 0' - explicitly reject any redirects
        # - Added '--resolve' with actual port - DNS pinning prevents DNS rebinding
        process = await asyncio.create_subprocess_exec(
            'curl',
            '--max-redirs', '0',  # S1: Block redirects (SSRF bypass prevention)
            '--resolve', f'{hostname}:{port}:{pinned_ip}',  # S1: DNS pin with actual port
            '-s',  # Silent
            '-f',  # Fail on HTTP errors
            '--connect-timeout', '5',  # Connection timeout (separate from transfer)
            '--max-time', str(int(URL_TIMEOUT_SECONDS)),  # Total transfer time
            '--max-filesize', str(MAX_IMAGE_SIZE),  # Size limit
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=URL_TIMEOUT_SECONDS + 1  # Small buffer for process cleanup
        )

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace').strip()
            # S1: Detect redirect rejection (curl exit code 47 or redirect-related errors)
            if 'redirect' in error_msg.lower() or process.returncode == 47:
                raise InvalidImageError(
                    f"URL attempted redirect (blocked for SSRF protection): {url}. "
                    f"Direct image URLs only - no redirects allowed."
                )
            raise ImageDownloadError(
                f"Failed to download image from {url}: {error_msg}"
            )

        image_bytes = stdout

        if len(image_bytes) == 0:
            raise ImageDownloadError(f"Empty response from {url}")

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise ImageTooLargeError(
                f"Image too large: {len(image_bytes)} bytes (max={MAX_IMAGE_SIZE})"
            )

        # Detect format from magic bytes
        format_str = detect_image_format(image_bytes)

        logger.debug(f"Downloaded image from {url}: {len(image_bytes)} bytes, format={format_str}")
        return (image_bytes, format_str)

    except asyncio.TimeoutError:
        # M5: Properly kill process on timeout
        try:
            process.kill()
            await process.wait()
        except Exception:
            pass  # Process may already be dead
        raise ImageDownloadError(f"Timeout downloading image from {url} after {URL_TIMEOUT_SECONDS}s")
    except ImageDownloadError:
        raise  # Re-raise our custom errors
    except ImageTooLargeError:
        raise
    except Exception as e:
        raise ImageDownloadError(f"Failed to download image from {url}: {e}")


def load_local_file(file_path: str) -> Tuple[bytes, str]:
    """
    Load image from local file path.

    Args:
        file_path: Path to local image file

    Returns:
        Tuple of (image_bytes, format)

    Raises:
        InvalidImageError: If file doesn't exist or can't be read
        ImageTooLargeError: If file exceeds size limit
    """
    path = Path(file_path)

    if not path.exists():
        raise InvalidImageError(f"File not found: {file_path}")

    if not path.is_file():
        raise InvalidImageError(f"Not a file: {file_path}")

    # Check size before reading
    file_size = path.stat().st_size
    if file_size > MAX_IMAGE_SIZE:
        raise ImageTooLargeError(
            f"File too large: {file_size} bytes (max={MAX_IMAGE_SIZE})"
        )

    try:
        image_bytes = path.read_bytes()
    except Exception as e:
        raise InvalidImageError(f"Failed to read file {file_path}: {e}")

    # Detect format
    format_str = detect_image_format(image_bytes)

    logger.debug(f"Loaded local file {file_path}: {len(image_bytes)} bytes, format={format_str}")
    return (image_bytes, format_str)


def detect_image_format(data: bytes) -> str:
    """
    Detect image format from magic bytes.

    Args:
        data: Image bytes

    Returns:
        Format string: 'jpeg', 'png', 'webp', 'gif', 'bmp', or 'unknown'
    """
    if len(data) < 2:
        return 'unknown'

    # Check magic bytes
    # JPEG: FF D8
    if data[:2] == b'\xff\xd8':
        return 'jpeg'

    # BMP: BM
    elif data[:2] == b'BM':
        return 'bmp'

    # GIF: GIF87a or GIF89a (need at least 3 bytes)
    elif len(data) >= 3 and data[:3] == b'GIF':
        return 'gif'

    # PNG: 89 50 4E 47 0D 0A 1A 0A (need 8 bytes)
    elif len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'

    # WebP: RIFF....WEBP (need 12 bytes)
    elif len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'

    else:
        return 'unknown'


def validate_image(data: bytes) -> bool:
    """
    Validate image data using PIL/Pillow.

    Opus Fix C3: Added dimension limits to prevent image bomb attacks.
    Large dimensions can cause excessive memory usage during decompression.

    Args:
        data: Image bytes

    Returns:
        True if valid image

    Raises:
        InvalidImageError: If image is corrupted or unsupported format
        ImageTooLargeError: If image dimensions exceed limits
        ImportError: If Pillow not installed (vision models require it)
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        raise ImportError(
            "Pillow is required for vision/multimodal support. "
            "Install: pip install pillow"
        )

    try:
        # Try to open and verify image
        img = Image.open(io.BytesIO(data))
        img.verify()  # Checks for corruption

        # Re-open to get dimensions (verify() invalidates the Image object)
        img = Image.open(io.BytesIO(data))
        width, height = img.size

        # Opus C3: Enforce dimension limits (prevents decompression bombs)
        if width > MAX_IMAGE_DIMENSIONS[0] or height > MAX_IMAGE_DIMENSIONS[1]:
            raise ImageTooLargeError(
                f"Image dimensions too large: {width}x{height} "
                f"(max: {MAX_IMAGE_DIMENSIONS[0]}x{MAX_IMAGE_DIMENSIONS[1]})"
            )

        logger.debug(f"Valid image: {img.format} {img.size} {img.mode}")
        return True

    except ImageTooLargeError:
        raise  # Re-raise our custom error
    except Exception as e:
        raise InvalidImageError(f"Invalid or corrupted image: {e}")


async def prepare_images(content_blocks: list, bridge=None) -> list:
    """
    Convert API content blocks to IPC ImageData messages.

    Handles:
    - Data URLs (base64): decode and validate
    - HTTP/HTTPS URLs: download, validate
    - Local file paths: load and validate
    - Auto-resize with caching (v3.1.0)
    - Size-based routing: inline (<500KB) vs shared memory (≥500KB)

    Args:
        content_blocks: List of ContentBlock objects from API
        bridge: Optional SharedMemoryBridge for large images

    Returns:
        List of ImageData objects ready for IPC

    Raises:
        ImageProcessingError: On any image processing failure
    """
    from ..ipc.messages import ImageData

    image_data_list = []
    total_size = 0

    for block in content_blocks:
        # Only process image_url blocks
        if not hasattr(block, 'type') or block.type != 'image_url':
            continue

        # Enforce max images limit
        if len(image_data_list) >= MAX_TOTAL_IMAGES:
            raise ImageProcessingError(
                f"Too many images: max {MAX_TOTAL_IMAGES} per request"
            )

        # Extract URL from block
        image_url_dict = block.image_url
        url = image_url_dict.get('url', '')

        if not url:
            raise InvalidImageError("Missing 'url' in image_url block")

        # Route based on URL type
        if url.startswith('data:image/'):
            # Data URL: decode base64
            image_bytes, format_str = decode_data_url(url)
        elif url.startswith(('http://', 'https://')):
            # HTTP URL: download
            image_bytes, format_str = await fetch_image(url)
        elif url.startswith('/') or url.startswith('./'):
            # Local file path
            image_bytes, format_str = load_local_file(url)
        else:
            raise InvalidImageError(
                f"Unsupported URL format: {url[:50]}... "
                "Expected: data:image/..., http://, https://, or /path/to/file"
            )

        # Validate image (before processing)
        validate_image(image_bytes)

        # Auto-resize with caching (v3.1.0)
        resize_metadata = None
        if AUTO_RESIZE_ENABLED and VISION_MAX_DIMENSION is not None:
            # Keep reference to original for cache key
            original_image_bytes = image_bytes

            # Check cache first
            cache = get_resize_cache()
            if cache:
                cached_resized = cache.get(original_image_bytes)
                if cached_resized:
                    logger.debug(f"Using cached resized image ({len(cached_resized)} bytes)")
                    image_bytes = cached_resized
                    resize_metadata = {"cached": True}
                else:
                    # Cache miss - resize and cache
                    resized_bytes, resize_metadata = auto_resize_image(
                        original_image_bytes,
                        VISION_MAX_DIMENSION
                    )
                    if resize_metadata["resized"]:
                        # Cache using original as key, resized as value
                        cache.put(original_image_bytes, resized_bytes)
                        resize_metadata["cached"] = False
                    image_bytes = resized_bytes
            else:
                # Cache disabled - just resize
                image_bytes, resize_metadata = auto_resize_image(
                    original_image_bytes,
                    VISION_MAX_DIMENSION
                )

        # Validate again after resize (paranoid check)
        validate_image(image_bytes)

        # Check total size
        total_size += len(image_bytes)
        if total_size > MAX_IMAGE_SIZE * MAX_TOTAL_IMAGES:
            raise ImageTooLargeError(
                f"Total images too large: {total_size} bytes "
                f"(max={MAX_IMAGE_SIZE * MAX_TOTAL_IMAGES})"
            )

        # Route to inline or shared memory based on size
        if len(image_bytes) < INLINE_THRESHOLD:
            # Small image: inline base64
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            image_data = ImageData(
                type='inline',
                data=base64_data,
                format=format_str
            )
            logger.debug(f"Image {len(image_data_list)}: inline, {len(image_bytes)} bytes")
        else:
            # Large image: shared memory
            if bridge is None:
                raise ImageProcessingError(
                    "Large image requires shared memory bridge, but none provided"
                )

            # C3 fix: write_image now returns generation for stale read detection
            offset, length, generation = bridge.write_image(image_bytes)
            image_data = ImageData(
                type='shmem',
                offset=offset,
                length=length,
                generation=generation,  # C3: Include generation for worker validation
                format=format_str
            )
            logger.debug(
                f"Image {len(image_data_list)}: shmem, {len(image_bytes)} bytes "
                f"at offset {offset}, generation={generation}"
            )

        image_data_list.append(image_data)

    logger.info(
        f"Prepared {len(image_data_list)} images, total {total_size} bytes"
    )
    return image_data_list
