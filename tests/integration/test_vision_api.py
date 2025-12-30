"""Integration tests for vision/multimodal support.

Tests the image preprocessing pipeline:
- Image decoding (data URLs, HTTP URLs)
- Image validation
- Security limits
- IPC message creation
"""

import pytest
import base64

from src.orchestrator.image_utils import (
    decode_data_url,
    validate_image,
    ImageProcessingError,
    ImageTooLargeError,
    InvalidImageError
)

# Test data: 1x1 red pixel PNG (valid, tiny image)
RED_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)

# Test data: 1x1 green pixel PNG (different from red for testing multiple images)
GREEN_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class TestImageDecoding:
    """Test image decoding from data URLs."""

    def test_decode_valid_png(self):
        """Test decoding valid PNG data URL."""
        data_url = f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}"

        image_bytes, format_str = decode_data_url(data_url)

        assert len(image_bytes) > 0
        assert format_str == "png"
        # PNG magic bytes: 89 50 4E 47
        assert image_bytes[:4] == b'\x89PNG'
        print(f"✅ Decoded PNG data URL ({len(image_bytes)} bytes)")

    def test_decode_invalid_url_format(self):
        """Test that invalid URL formats are rejected."""
        invalid_urls = [
            "not-a-data-url",
            "data:text/plain;base64,abc",  # Not image/*
            "data:image/png,notbase64",  # Missing base64 encoding
        ]

        for invalid_url in invalid_urls:
            with pytest.raises(InvalidImageError):
                decode_data_url(invalid_url)

        print(f"✅ Invalid URL formats correctly rejected")

    def test_decode_invalid_base64(self):
        """Test that invalid base64 is rejected."""
        invalid_url = "data:image/png;base64,!!!INVALID!!!"

        with pytest.raises(InvalidImageError):
            decode_data_url(invalid_url)

        print(f"✅ Invalid base64 correctly rejected")


class TestImageValidation:
    """Test image validation with PIL."""

    def test_validate_valid_png(self):
        """Test that valid images pass validation."""
        data_url = f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}"
        image_bytes, _ = decode_data_url(data_url)

        # Should not raise
        result = validate_image(image_bytes)
        assert result == True
        print(f"✅ Valid PNG passed validation")

    def test_validate_corrupted_image(self):
        """Test that corrupted images are rejected."""
        # Truncated/corrupted image data
        corrupted_data = b'\x89PNG\r\n\x1a\n' + b'corrupted'

        with pytest.raises(InvalidImageError):
            validate_image(corrupted_data)

        print(f"✅ Corrupted image correctly rejected")

    def test_validate_non_image_data(self):
        """Test that non-image data is rejected."""
        non_image = b"This is not an image"

        with pytest.raises(InvalidImageError):
            validate_image(non_image)

        print(f"✅ Non-image data correctly rejected")


class TestSecurityLimits:
    """Test security limits are enforced."""

    def test_reject_oversized_base64(self):
        """Test that oversized base64 data is rejected."""
        # Create ~15MB of base64 data (should exceed 10MB limit after decode)
        large_data = base64.b64encode(b'X' * (15 * 1024 * 1024)).decode('ascii')
        data_url = f"data:image/png;base64,{large_data}"

        with pytest.raises(ImageTooLargeError):
            decode_data_url(data_url)

        print(f"✅ Oversized image correctly rejected")

    def test_multiple_small_images_ok(self):
        """Test that multiple small images are accepted."""
        # Multiple 1x1 pixel images should be fine
        images = []
        for i in range(3):
            data_url = f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}"
            image_bytes, format_str = decode_data_url(data_url)
            images.append((image_bytes, format_str))

        assert len(images) == 3
        print(f"✅ Multiple small images accepted")


class TestImageProcessing:
    """Test end-to-end image processing."""

    def test_process_single_image(self):
        """Test processing a single image through the pipeline."""
        data_url = f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}"

        # Decode
        image_bytes, format_str = decode_data_url(data_url)
        assert format_str == "png"

        # Validate
        result = validate_image(image_bytes)
        assert result == True

        print(f"✅ Single image processed successfully")

    def test_process_multiple_images(self):
        """Test processing multiple images."""
        data_urls = [
            f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}",
            f"data:image/png;base64,{GREEN_PIXEL_PNG_BASE64}",
        ]

        processed = []
        for data_url in data_urls:
            image_bytes, format_str = decode_data_url(data_url)
            validate_image(image_bytes)
            processed.append((image_bytes, format_str))

        assert len(processed) == 2
        print(f"✅ Multiple images processed successfully")


# Summary test to verify all components work together
def test_vision_pipeline_summary():
    """Summary test: Verify full vision processing pipeline."""
    print("\n" + "="*60)
    print("VISION PROCESSING PIPELINE VALIDATION")
    print("="*60)

    steps_passed = 0
    total_steps = 6

    # Step 1: Decode data URL
    try:
        data_url = f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}"
        image_bytes, format_str = decode_data_url(data_url)
        assert len(image_bytes) > 0
        steps_passed += 1
        print(f"✅ Step 1/6: Data URL decoding")
    except Exception as e:
        print(f"❌ Step 1/6 FAILED: {e}")

    # Step 2: Format detection
    try:
        assert format_str == "png"
        assert image_bytes[:4] == b'\x89PNG'
        steps_passed += 1
        print(f"✅ Step 2/6: Format detection")
    except Exception as e:
        print(f"❌ Step 2/6 FAILED: {e}")

    # Step 3: Image validation
    try:
        validate_image(image_bytes)
        steps_passed += 1
        print(f"✅ Step 3/6: Image validation")
    except Exception as e:
        print(f"❌ Step 3/6 FAILED: {e}")

    # Step 4: Invalid image rejection
    try:
        with pytest.raises(InvalidImageError):
            validate_image(b"not an image")
        steps_passed += 1
        print(f"✅ Step 4/6: Invalid image rejection")
    except Exception as e:
        print(f"❌ Step 4/6 FAILED: {e}")

    # Step 5: Size limit enforcement
    try:
        large_data = base64.b64encode(b'X' * (15 * 1024 * 1024)).decode('ascii')
        with pytest.raises(ImageTooLargeError):
            decode_data_url(f"data:image/png;base64,{large_data}")
        steps_passed += 1
        print(f"✅ Step 5/6: Size limit enforcement")
    except Exception as e:
        print(f"❌ Step 5/6 FAILED: {e}")

    # Step 6: Multiple images
    try:
        for data_url in [
            f"data:image/png;base64,{RED_PIXEL_PNG_BASE64}",
            f"data:image/png;base64,{GREEN_PIXEL_PNG_BASE64}",
        ]:
            image_bytes, _ = decode_data_url(data_url)
            validate_image(image_bytes)
        steps_passed += 1
        print(f"✅ Step 6/6: Multiple images")
    except Exception as e:
        print(f"❌ Step 6/6 FAILED: {e}")

    # Summary
    print("="*60)
    print(f"RESULT: {steps_passed}/{total_steps} steps passed")
    if steps_passed == total_steps:
        print("✅ VISION PIPELINE FULLY OPERATIONAL")
    else:
        print(f"⚠️ VISION PIPELINE PARTIALLY OPERATIONAL ({steps_passed}/{total_steps})")
    print("="*60 + "\n")

    assert steps_passed == total_steps, f"Pipeline validation failed: {steps_passed}/{total_steps} passed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
