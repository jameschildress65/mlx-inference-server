"""
Unit tests for C3 fix: Image buffer generation counter.

Tests that write_image() returns a generation counter and read_image()
validates it to detect stale reads when buffer is reset.
"""

import pytest

try:
    import posix_ipc
    HAS_POSIX_IPC = True
except ImportError:
    HAS_POSIX_IPC = False

pytestmark = pytest.mark.skipif(not HAS_POSIX_IPC, reason="posix_ipc not available")


class TestImageBufferGenerationCounter:
    """Tests for C3 fix: generation counter in image buffer."""

    def test_write_image_returns_generation(self):
        """Verify write_image returns (offset, length, generation) tuple."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_gen_return", is_server=True)

        result = bridge.write_image(b"test_image_data")

        assert len(result) == 3, "write_image should return 3-tuple"
        offset, length, generation = result
        assert isinstance(offset, int)
        assert isinstance(length, int)
        assert isinstance(generation, int)
        assert generation == 0  # Initial generation

        bridge.close()

    def test_generation_starts_at_zero(self):
        """Verify initial generation counter is 0."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_gen_zero", is_server=True)

        _, _, generation = bridge.write_image(b"x" * 100)
        assert generation == 0

        bridge.close()

    def test_generation_increments_on_reset(self):
        """Verify generation counter increments when buffer resets."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_gen_incr", is_server=True)

        # Fill buffer to trigger reset
        # Buffer is 16MB minus 8 byte header = ~16MB effective
        chunk_size = 1024 * 1024  # 1MB chunks
        generations = []

        for i in range(20):  # 20MB total > 16MB buffer
            _, _, gen = bridge.write_image(b"x" * chunk_size)
            generations.append(gen)

        # Should have seen at least one generation increment
        assert max(generations) >= 1, "Generation should have incremented on buffer reset"

        bridge.close()

    def test_read_with_correct_generation_succeeds(self):
        """Verify read succeeds when generation matches."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_read_ok", is_server=True)

        test_data = b"test_image_data_for_reading"
        offset, length, generation = bridge.write_image(test_data)

        # Read with correct generation should succeed
        read_data = bridge.read_image(offset, length, generation)
        assert read_data == test_data

        bridge.close()

    def test_read_with_stale_generation_fails(self):
        """Verify read fails with ValueError when generation mismatches."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_read_stale", is_server=True)

        # Write initial data
        offset, length, old_generation = bridge.write_image(b"x" * 1000)

        # Fill buffer to trigger reset (increment generation)
        for _ in range(20):
            bridge.write_image(b"y" * (1024 * 1024))

        # Read with old generation should fail
        with pytest.raises(ValueError, match="generation"):
            bridge.read_image(offset, length, old_generation)

        bridge.close()

    def test_stale_read_error_message_is_helpful(self):
        """Verify stale read error message includes generation info."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_error_msg", is_server=True)

        offset, length, old_gen = bridge.write_image(b"x" * 1000)

        # Trigger reset
        for _ in range(20):
            bridge.write_image(b"y" * (1024 * 1024))

        try:
            bridge.read_image(offset, length, old_gen)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "generation" in error_msg.lower()
            assert "stale" in error_msg.lower()

        bridge.close()

    def test_multiple_resets_increment_generation(self):
        """Verify each buffer reset increments generation."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_multi_reset", is_server=True)

        chunk_size = 8 * 1024 * 1024  # 8MB - will reset after 2 writes
        seen_generations = set()

        for _ in range(10):
            _, _, gen = bridge.write_image(b"x" * chunk_size)
            seen_generations.add(gen)

        # Should have seen multiple generations
        assert len(seen_generations) >= 3, f"Expected multiple generations, got {seen_generations}"

        bridge.close()


class TestImageDataGeneration:
    """Tests for ImageData model with generation field."""

    def test_image_data_has_generation_field(self):
        """Verify ImageData model accepts generation field."""
        from src.ipc.messages import ImageData

        img = ImageData(
            type='shmem',
            offset=100,
            length=5000,
            generation=42,
            format='jpeg'
        )

        assert img.generation == 42

    def test_image_data_generation_optional(self):
        """Verify generation is optional for backward compatibility."""
        from src.ipc.messages import ImageData

        # Inline images don't need generation
        img = ImageData(
            type='inline',
            data='base64encodeddata',
            format='png'
        )

        assert img.generation is None

    def test_shmem_image_serialization(self):
        """Verify shmem ImageData serializes with generation."""
        from src.ipc.messages import ImageData

        img = ImageData(
            type='shmem',
            offset=0,
            length=1024,
            generation=5,
            format='jpeg'
        )

        serialized = img.model_dump()
        assert serialized['generation'] == 5
        assert serialized['type'] == 'shmem'
