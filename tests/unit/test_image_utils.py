"""Unit tests for image preprocessing utilities (Phase 2)."""

import pytest
import base64
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.orchestrator.image_utils import (
    decode_data_url,
    fetch_image,
    load_local_file,
    detect_image_format,
    validate_image,
    prepare_images,
    ImageProcessingError,
    ImageTooLargeError,
    ImageDownloadError,
    InvalidImageError,
    MAX_IMAGE_SIZE,
    MAX_TOTAL_IMAGES,
)


# Test fixtures - minimal valid images

# 1x1 red PNG (67 bytes)
TINY_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
)

# 1x1 red JPEG (631 bytes)
TINY_JPEG = base64.b64decode(
    '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKKAP/2Q=='
)

# 1x1 red WebP (26 bytes - invalid, for testing)
TINY_WEBP_INVALID = b'RIFF\x1a\x00\x00\x00WEBP' + b'\x00' * 16


class TestDecodeDataUrl:
    """Tests for decode_data_url()."""

    def test_decode_valid_png_data_url(self):
        """Test decoding valid PNG data URL."""
        base64_png = base64.b64encode(TINY_PNG).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_png}"

        image_bytes, format_str = decode_data_url(data_url)

        assert image_bytes == TINY_PNG
        assert format_str == 'png'

    def test_decode_valid_jpeg_data_url(self):
        """Test decoding valid JPEG data URL."""
        base64_jpeg = base64.b64encode(TINY_JPEG).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_jpeg}"

        image_bytes, format_str = decode_data_url(data_url)

        assert image_bytes == TINY_JPEG
        assert format_str == 'jpeg'

    def test_decode_invalid_format(self):
        """Test decode_data_url rejects invalid format."""
        with pytest.raises(InvalidImageError, match="Invalid data URL format"):
            decode_data_url("not-a-data-url")

    def test_decode_missing_base64(self):
        """Test decode_data_url rejects missing base64 prefix."""
        with pytest.raises(InvalidImageError, match="Invalid data URL format"):
            decode_data_url("data:image/png,notbase64")

    def test_decode_invalid_base64(self):
        """Test decode_data_url rejects invalid base64."""
        data_url = "data:image/png;base64,!!!invalid!!!"

        with pytest.raises(InvalidImageError, match="Failed to decode base64"):
            decode_data_url(data_url)

    def test_decode_too_large(self):
        """Test decode_data_url rejects images exceeding size limit."""
        # Create fake base64 that decodes to >10MB
        large_data = b'\x00' * (MAX_IMAGE_SIZE + 1)
        base64_large = base64.b64encode(large_data).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_large}"

        with pytest.raises(ImageTooLargeError, match="Image too large"):
            decode_data_url(data_url)


class TestDetectImageFormat:
    """Tests for detect_image_format()."""

    def test_detect_png(self):
        """Test PNG detection."""
        assert detect_image_format(TINY_PNG) == 'png'

    def test_detect_jpeg(self):
        """Test JPEG detection."""
        assert detect_image_format(TINY_JPEG) == 'jpeg'

    def test_detect_webp(self):
        """Test WebP detection."""
        webp_header = b'RIFF\x1a\x00\x00\x00WEBP'
        assert detect_image_format(webp_header) == 'webp'

    def test_detect_gif(self):
        """Test GIF detection."""
        gif_header = b'GIF89a'
        assert detect_image_format(gif_header) == 'gif'

    def test_detect_bmp(self):
        """Test BMP detection."""
        bmp_header = b'BM' + b'\x00' * 10
        assert detect_image_format(bmp_header) == 'bmp'

    def test_detect_unknown(self):
        """Test unknown format."""
        assert detect_image_format(b'\x00\x01\x02\x03') == 'unknown'

    def test_detect_too_short(self):
        """Test data too short."""
        assert detect_image_format(b'\x00') == 'unknown'


class TestValidateImage:
    """Tests for validate_image() with PIL."""

    def test_validate_valid_png(self):
        """Test valid PNG passes validation."""
        # Pillow is optional, skip if not installed
        pytest.importorskip("PIL")

        assert validate_image(TINY_PNG) is True

    def test_validate_valid_jpeg(self):
        """Test valid JPEG passes validation."""
        pytest.importorskip("PIL")

        assert validate_image(TINY_JPEG) is True

    def test_validate_corrupted_image(self):
        """Test corrupted image fails validation."""
        pytest.importorskip("PIL")

        corrupted = b'\xff\xd8\xff\xe0' + b'\x00' * 100  # JPEG header but truncated

        with pytest.raises(InvalidImageError, match="Invalid or corrupted"):
            validate_image(corrupted)

    def test_validate_non_image_data(self):
        """Test non-image data fails validation."""
        pytest.importorskip("PIL")

        with pytest.raises(InvalidImageError, match="Invalid or corrupted"):
            validate_image(b'This is not an image')


class TestFetchImage:
    """Tests for fetch_image() with URL download."""

    @pytest.mark.asyncio
    async def test_fetch_image_success(self):
        """Test successful image download."""
        url = "https://example.com/test.png"

        # Mock asyncio.create_subprocess_exec
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(TINY_PNG, b''))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            image_bytes, format_str = await fetch_image(url)

        assert image_bytes == TINY_PNG
        assert format_str == 'png'

    @pytest.mark.asyncio
    async def test_fetch_image_invalid_scheme(self):
        """Test fetch_image rejects invalid URL schemes."""
        with pytest.raises(InvalidImageError, match="Invalid URL scheme"):
            await fetch_image("ftp://example.com/image.png")

    @pytest.mark.asyncio
    async def test_fetch_image_timeout(self):
        """Test fetch_image handles timeout."""
        url = "https://example.com/slow.png"

        # Mock timeout
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with pytest.raises(ImageDownloadError, match="Timeout downloading"):
                await fetch_image(url)

    @pytest.mark.asyncio
    async def test_fetch_image_http_error(self):
        """Test fetch_image handles HTTP errors."""
        url = "https://example.com/404.png"

        # Mock HTTP 404
        mock_process = AsyncMock()
        mock_process.returncode = 22  # curl error code for HTTP 404
        mock_process.communicate = AsyncMock(return_value=(b'', b'HTTP 404 Not Found'))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with pytest.raises(ImageDownloadError, match="Failed to download"):
                await fetch_image(url)


class TestLoadLocalFile:
    """Tests for load_local_file()."""

    def test_load_existing_file(self, tmp_path):
        """Test loading existing local file."""
        # Create temp file
        test_file = tmp_path / "test.png"
        test_file.write_bytes(TINY_PNG)

        image_bytes, format_str = load_local_file(str(test_file))

        assert image_bytes == TINY_PNG
        assert format_str == 'png'

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file fails."""
        with pytest.raises(InvalidImageError, match="File not found"):
            load_local_file("/nonexistent/path/image.png")

    def test_load_directory(self, tmp_path):
        """Test loading directory fails."""
        with pytest.raises(InvalidImageError, match="Not a file"):
            load_local_file(str(tmp_path))

    def test_load_file_too_large(self, tmp_path):
        """Test loading file exceeding size limit fails."""
        # Create file larger than limit
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b'\x00' * (MAX_IMAGE_SIZE + 1))

        with pytest.raises(ImageTooLargeError, match="File too large"):
            load_local_file(str(large_file))


class TestPrepareImages:
    """Tests for prepare_images() orchestrator."""

    @pytest.mark.asyncio
    async def test_prepare_single_inline_image(self):
        """Test preparing single small image (inline)."""
        pytest.importorskip("PIL")

        # Mock content block
        mock_block = Mock()
        mock_block.type = 'image_url'
        base64_png = base64.b64encode(TINY_PNG).decode('utf-8')
        mock_block.image_url = {'url': f'data:image/png;base64,{base64_png}'}

        result = await prepare_images([mock_block], bridge=None)

        assert len(result) == 1
        assert result[0].type == 'inline'
        assert result[0].format == 'png'
        assert result[0].data is not None

    @pytest.mark.asyncio
    async def test_prepare_multiple_images(self):
        """Test preparing multiple images."""
        pytest.importorskip("PIL")

        # Create 3 small images
        blocks = []
        for i in range(3):
            mock_block = Mock()
            mock_block.type = 'image_url'
            base64_png = base64.b64encode(TINY_PNG).decode('utf-8')
            mock_block.image_url = {'url': f'data:image/png;base64,{base64_png}'}
            blocks.append(mock_block)

        result = await prepare_images(blocks, bridge=None)

        assert len(result) == 3
        assert all(img.type == 'inline' for img in result)

    @pytest.mark.asyncio
    async def test_prepare_too_many_images(self):
        """Test preparing >5 images fails."""
        pytest.importorskip("PIL")

        # Create 6 images
        blocks = []
        for i in range(MAX_TOTAL_IMAGES + 1):
            mock_block = Mock()
            mock_block.type = 'image_url'
            base64_png = base64.b64encode(TINY_PNG).decode('utf-8')
            mock_block.image_url = {'url': f'data:image/png;base64,{base64_png}'}
            blocks.append(mock_block)

        with pytest.raises(ImageProcessingError, match="Too many images"):
            await prepare_images(blocks, bridge=None)

    @pytest.mark.asyncio
    async def test_prepare_large_image_shmem(self):
        """Test preparing large image uses shared memory."""
        pytest.importorskip("PIL")

        # Create fake large image (>500KB)
        large_png = TINY_PNG * 10000  # Repeat to make larger

        mock_block = Mock()
        mock_block.type = 'image_url'
        base64_large = base64.b64encode(large_png).decode('utf-8')
        mock_block.image_url = {'url': f'data:image/png;base64,{base64_large}'}

        # Mock bridge
        mock_bridge = Mock()
        mock_bridge.write_image = Mock(return_value=(1024, len(large_png)))

        result = await prepare_images([mock_block], bridge=mock_bridge)

        assert len(result) == 1
        assert result[0].type == 'shmem'
        assert result[0].offset == 1024
        assert result[0].length == len(large_png)
        mock_bridge.write_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_skips_text_blocks(self):
        """Test prepare_images skips non-image blocks."""
        pytest.importorskip("PIL")

        # Mix of text and image blocks
        text_block = Mock()
        text_block.type = 'text'
        text_block.text = "Hello"

        image_block = Mock()
        image_block.type = 'image_url'
        base64_png = base64.b64encode(TINY_PNG).decode('utf-8')
        image_block.image_url = {'url': f'data:image/png;base64,{base64_png}'}

        result = await prepare_images([text_block, image_block], bridge=None)

        # Should only process image block
        assert len(result) == 1
        assert result[0].type == 'inline'
