"""Image preprocessing utilities for vision/multimodal requests.

Handles:
- Base64 data URL decoding
- HTTP/HTTPS image downloading
- Local file path loading
- Image validation with PIL
- Size limit enforcement
- SSRF protection (blocking private IPs)
"""

import base64
import re
import logging
import socket
import ipaddress
from typing import Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
import asyncio

logger = logging.getLogger(__name__)

# Configuration (from plan)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB per image
MAX_TOTAL_IMAGES = 5  # Max images per request
MAX_IMAGE_DIMENSIONS = (4096, 4096)  # Max width/height (Opus C3 fix - prevents image bombs)
URL_TIMEOUT_SECONDS = 10.0  # HTTP download timeout
INLINE_THRESHOLD = 10 * 1024 * 1024  # 10MB threshold - inline all images (no shmem needed for typical use)

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

    # SSRF Protection: Block private IP ranges (Opus 4.5 Critical Fix C1)
    is_safe, error_msg = is_safe_url(url)
    if not is_safe:
        raise InvalidImageError(f"URL blocked by SSRF protection: {error_msg}")

    try:
        # M5: Improved timeout handling with separate connect/total timeouts
        # Use asyncio subprocess to run curl with timeout
        # This avoids adding httpx/aiohttp dependencies
        process = await asyncio.create_subprocess_exec(
            'curl',
            '-L',  # Follow redirects
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

        # Validate image
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

            offset, length = bridge.write_image(image_bytes)
            image_data = ImageData(
                type='shmem',
                offset=offset,
                length=length,
                format=format_str
            )
            logger.debug(
                f"Image {len(image_data_list)}: shmem, {len(image_bytes)} bytes "
                f"at offset {offset}"
            )

        image_data_list.append(image_data)

    logger.info(
        f"Prepared {len(image_data_list)} images, total {total_size} bytes"
    )
    return image_data_list
