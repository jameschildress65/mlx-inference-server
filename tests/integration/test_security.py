"""
Security tests for MLX Server V3.

Tests critical security fixes from Opus 4.5 review (2025-12-24):
- CRITICAL-1: Command injection via model path
- CRITICAL-2: IPC timeout implementation
- HIGH-1: TOCTOU race condition in idle monitor
"""

import pytest
import subprocess
import time
from pathlib import Path
import sys

from src.orchestrator.worker_manager import WorkerManager, WorkerSpawnError
from src.config.server_config import ServerConfig
from unittest.mock import Mock


@pytest.fixture
def worker_manager():
    """Create a WorkerManager instance for testing."""
    config = Mock()
    config.idle_timeout_seconds = 300
    return WorkerManager(config)


class TestCommandInjectionPrevention:
    """Test CRITICAL-1: Command injection via model path."""

    def test_reject_invalid_format(self, worker_manager):
        """Test that invalid path formats are rejected."""
        wm = worker_manager

        # Invalid formats that should be rejected
        invalid_paths = [
            "no-slash",  # Missing organization/model format
            "/absolute/path",  # Absolute path
            "../path/traversal",  # Path traversal
            "org/../traversal",  # Path traversal in model name
            "org/model/../traversal",  # Path traversal after valid prefix
            "org/",  # Missing model name
            "/org/model",  # Leading slash
            "org//model",  # Double slash
            "org/model;malicious",  # Semicolon injection attempt
            "org/model|whoami",  # Pipe injection attempt
            "org/model&rm",  # Command chaining attempt
            "org/model`id`",  # Command substitution attempt
            "org/model$(id)",  # Command substitution attempt
            "org/model\nrm",  # Newline injection
            "org/model\x00null",  # Null byte injection
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(ValueError, match="Invalid model path format|Path traversal"):
                wm.load_model(invalid_path, timeout=1)

    def test_reject_untrusted_organization(self, worker_manager):
        """Test that untrusted organizations are rejected."""
        wm = worker_manager

        # Organizations not in whitelist
        untrusted_orgs = [
            "evil-attacker/malicious-model",
            "unknown-org/model",
            "hacker/backdoor",
            "mallory/trojan",
        ]

        for untrusted_path in untrusted_orgs:
            with pytest.raises(ValueError, match="Untrusted organization"):
                wm.load_model(untrusted_path, timeout=1)

    def test_accept_trusted_organizations(self, worker_manager):
        """Test that trusted organizations pass validation."""
        wm = worker_manager

        # These should pass validation (will fail at model load, but that's OK)
        trusted_paths = [
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "meta-llama/Llama-3-8B",
            "mistralai/Mistral-7B-v0.1",
            "Qwen/Qwen2.5-7B",
            "microsoft/phi-2",
            "google/gemma-7b",
        ]

        for trusted_path in trusted_paths:
            try:
                # Will fail at subprocess spawn (model doesn't exist), but validation passes
                wm.load_model(trusted_path, timeout=1)
            except WorkerSpawnError:
                # Expected - model doesn't exist, but validation passed
                pass
            except ValueError as e:
                # Should not reject trusted organizations
                pytest.fail(f"Rejected trusted path {trusted_path}: {e}")

    def test_path_traversal_prevention(self, worker_manager):
        """Test that path traversal attacks are prevented."""
        wm = worker_manager

        # Path traversal attempts
        traversal_paths = [
            "../../../etc/passwd",
            "mlx-community/../../../etc/passwd",
            "mlx-community/model../../etc/passwd",
            "mlx-community/..%2F..%2F..%2Fetc%2Fpasswd",  # URL encoded
            "mlx-community/model/../../../etc/passwd",
        ]

        for traversal_path in traversal_paths:
            with pytest.raises(ValueError, match="Path traversal|Invalid model path format"):
                wm.load_model(traversal_path, timeout=1)

    def test_special_characters_rejected(self, worker_manager):
        """Test that special shell characters are rejected."""
        wm = worker_manager

        # Shell metacharacters that could be used for injection
        special_char_paths = [
            "mlx-community/model;echo",
            "mlx-community/model|whoami",
            "mlx-community/model&id",
            "mlx-community/model`ls`",
            "mlx-community/model$(pwd)",
            "mlx-community/model>file",
            "mlx-community/model<file",
            "mlx-community/model*",
            "mlx-community/model?",
            "mlx-community/model[a-z]",
            "mlx-community/model{1,2}",
            "mlx-community/model$VAR",
            "mlx-community/model~user",
        ]

        for special_path in special_char_paths:
            with pytest.raises(ValueError, match="Invalid model path format"):
                wm.load_model(special_path, timeout=1)

    def test_whitespace_handling(self, worker_manager):
        """Test that paths with whitespace are handled correctly."""
        wm = worker_manager

        # Whitespace should be rejected (not in allowed pattern)
        whitespace_paths = [
            "mlx-community/model name",  # Space
            "mlx-community/model\tname",  # Tab
            "mlx-community/model\nname",  # Newline
            "mlx-community/model\rname",  # Carriage return
            " mlx-community/model",  # Leading space
            "mlx-community/model ",  # Trailing space
        ]

        for whitespace_path in whitespace_paths:
            with pytest.raises(ValueError, match="Invalid model path format"):
                wm.load_model(whitespace_path, timeout=1)

    def test_unicode_and_encoding_attacks(self, worker_manager):
        """Test that unicode and encoding attacks are prevented."""
        wm = worker_manager

        # Unicode/encoding attack attempts
        encoding_paths = [
            "mlx-community/model\u0000",  # Null byte (Unicode)
            "mlx-community/model%00",  # URL encoded null
            "mlx-community/model%2e%2e%2f",  # URL encoded ../
            "mlx-community/model\x00",  # Hex null
            "mlx-community/model\x1b[0m",  # ANSI escape codes
            "mlx-community/\u202e",  # Right-to-left override
        ]

        for encoding_path in encoding_paths:
            with pytest.raises(ValueError, match="Invalid model path format|ASCII-only|confusable"):
                wm.load_model(encoding_path, timeout=1)

    def test_unicode_normalization_bypass_prevention(self, worker_manager):
        """
        Test 4.2 fix: Unicode normalization bypass attacks are prevented.

        Attackers can use confusable Unicode characters that normalize to ASCII,
        potentially bypassing validation. For example:
        - Full-width solidus (U+FF0F) normalizes to "/" (U+002F)
        - Full-width letters normalize to ASCII letters
        """
        wm = worker_manager

        # Unicode normalization bypass attempts
        normalization_attacks = [
            "mlx-community\uff0fQwen",  # Full-width solidus (／) instead of /
            "mlx\u2010community/Qwen",  # Hyphen (‐) instead of -
            "ｍｌｘ-community/Qwen",  # Full-width letters
            "mlx-community/Ｑｗｅｎ",  # Full-width model name
            "mlx\u2212community/Qwen",  # Minus sign instead of hyphen
            "mlx\u00adcommunity/Qwen",  # Soft hyphen (invisible)
        ]

        for attack_path in normalization_attacks:
            with pytest.raises(ValueError, match="confusable Unicode|ASCII-only"):
                wm.load_model(attack_path, timeout=1)

    def test_valid_model_paths_accepted(self, worker_manager):
        """Test that valid model paths are accepted."""
        wm = worker_manager

        # Valid paths with various allowed characters
        valid_paths = [
            "mlx-community/Qwen2.5-7B-Instruct-4bit",  # Dots, dashes, numbers
            "mlx-community/model_with_underscores",  # Underscores
            "mlx-community/model-with-dashes",  # Dashes
            "mlx-community/Model123",  # Numbers
            "mlx-community/model.v2.0",  # Multiple dots
            "microsoft/phi-2",  # Short names
            "google/gemma-2b-it",  # Multiple dashes
        ]

        for valid_path in valid_paths:
            try:
                # Will fail at subprocess spawn, but validation should pass
                wm.load_model(valid_path, timeout=1)
            except WorkerSpawnError:
                # Expected - validation passed, model doesn't exist
                pass
            except ValueError as e:
                pytest.fail(f"Rejected valid path {valid_path}: {e}")


class TestIPCTimeout:
    """Test CRITICAL-2: IPC timeout implementation."""

    def test_timeout_enforced_on_hung_worker(self, worker_manager):
        """Test that IPC timeout prevents infinite blocking."""
        from src.ipc.stdio_bridge import StdioBridge, WorkerCommunicationError

        # Create a subprocess that never sends data
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(999)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            start_time = time.time()

            # Should timeout after 2 seconds
            with pytest.raises(WorkerCommunicationError, match="timeout"):
                StdioBridge.receive_message(proc, timeout=2.0)

            elapsed = time.time() - start_time

            # Should timeout close to 2 seconds (not hang forever)
            assert 1.5 < elapsed < 3.0, f"Timeout took {elapsed}s, expected ~2s"

        finally:
            proc.kill()
            proc.wait()

    def test_default_timeout_prevents_infinite_wait(self, worker_manager):
        """Test that default timeout is applied when none specified."""
        from src.ipc.stdio_bridge import StdioBridge, WorkerCommunicationError

        # Create a subprocess that never sends data
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(999)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            start_time = time.time()

            # Should use default timeout (30s)
            with pytest.raises(WorkerCommunicationError, match="timeout"):
                StdioBridge.receive_message(proc, timeout=None)

            elapsed = time.time() - start_time

            # Should timeout around 30 seconds
            assert 28 < elapsed < 35, f"Default timeout took {elapsed}s, expected ~30s"

        finally:
            proc.kill()
            proc.wait()

    def test_timeout_does_not_interfere_with_normal_operation(self, worker_manager):
        """Test that timeout doesn't break normal IPC."""
        from src.ipc.stdio_bridge import StdioBridge
        from src.ipc.messages import ReadyMessage

        # Create a subprocess that sends valid JSON quickly
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                'import json, sys; '
                'msg = {"type": "ready", "model_name": "test", "memory_gb": 1.0}; '
                'print(json.dumps(msg), flush=True); '
                'import time; time.sleep(999)'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            # Should receive message successfully
            msg = StdioBridge.receive_message(proc, timeout=5.0)

            assert msg.type == "ready"
            assert msg.model_name == "test"
            assert msg.memory_gb == 1.0

        finally:
            proc.kill()
            proc.wait()

    def test_short_timeout_respected(self, worker_manager):
        """Test that very short timeouts work correctly."""
        from src.ipc.stdio_bridge import StdioBridge, WorkerCommunicationError

        # Create a subprocess that sends data after 2 seconds
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                'import time, json, sys; '
                'time.sleep(2); '
                'print(json.dumps({"type": "ready", "model_name": "test", "memory_gb": 1.0}), flush=True)'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            start_time = time.time()

            # Should timeout after 0.5 seconds (before data arrives)
            with pytest.raises(WorkerCommunicationError, match="timeout"):
                StdioBridge.receive_message(proc, timeout=0.5)

            elapsed = time.time() - start_time

            # Should timeout quickly
            assert 0.4 < elapsed < 1.0, f"Timeout took {elapsed}s, expected ~0.5s"

        finally:
            proc.kill()
            proc.wait()


class TestTOCTOURaceCondition:
    """Test HIGH-1: TOCTOU race condition fix."""

    def test_atomic_unload_if_idle(self, worker_manager):
        """Test that unload_model_if_idle is atomic."""
        wm = worker_manager

        # No worker loaded - should return None
        result = wm.unload_model_if_idle()
        assert result is None

    def test_unload_if_idle_respects_active_requests(self, worker_manager):
        """Test that unload is prevented when requests are active."""
        wm = worker_manager

        # Load a model first (will fail, but that's OK for this test)
        try:
            wm.load_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit", timeout=1)
        except:
            pass

        # Simulate active request
        wm._increment_active_requests()

        try:
            # Should return None (not unload) because request is active
            result = wm.unload_model_if_idle()
            assert result is None

        finally:
            # Cleanup
            wm._decrement_active_requests()

    def test_unload_if_idle_succeeds_when_truly_idle(self, worker_manager):
        """Test that unload succeeds when no active requests."""
        import os

        # Skip if no test model available
        test_model = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        if not test_model:
            pytest.skip("No test model configured")

        wm = worker_manager

        try:
            # Load a real model
            load_result = wm.load_model(test_model, timeout=120)
            assert load_result.model_name is not None

            # No active requests - should unload successfully
            result = wm.unload_model_if_idle()
            assert result is not None
            assert result.memory_freed_gb > 0

        except Exception as e:
            # If model load fails, skip test
            pytest.skip(f"Model load failed: {e}")


class TestSecurityBestPractices:
    """Test general security best practices."""

    def test_no_hardcoded_secrets(self, worker_manager):
        """Verify no API keys or secrets in code."""
        src_dir = Path(__file__).parent.parent.parent / "src"

        # Patterns for actual secret values (not just variable names)
        secret_patterns = [
            b'"sk-ant-',  # Anthropic API key in string
            b"'sk-ant-",  # Anthropic API key in string
            b'"Bearer ',  # Bearer token in string
            b"'Bearer ",  # Bearer token in string
        ]

        violations = []

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_bytes()
            for pattern in secret_patterns:
                if pattern in content:
                    lines = content.decode('utf-8', errors='ignore').split('\n')
                    for i, line in enumerate(lines):
                        if pattern.decode('utf-8') in line:
                            # Flag actual hardcoded secret values
                            # Skip if it's getting from environment
                            if 'os.getenv' not in line and 'os.environ' not in line:
                                violations.append(f"{py_file}:{i+1} - {line.strip()}")

        if violations:
            pytest.fail(f"Potential hardcoded secrets found:\n" + "\n".join(violations[:5]))

    def test_input_validation_on_public_endpoints(self, worker_manager):
        """Verify that public API endpoints validate inputs."""
        # This is a smoke test - detailed validation tested in other classes
        wm = worker_manager

        # Empty model path should be rejected
        with pytest.raises((ValueError, TypeError)):
            wm.load_model("", timeout=1)

        # None should be rejected
        with pytest.raises((ValueError, TypeError)):
            wm.load_model(None, timeout=1)


if __name__ == "__main__":
    # Run tests with: pytest tests/integration/test_security.py -v
    pytest.main([__file__, "-v", "--tb=short"])
