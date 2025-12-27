"""
Server Configuration with Auto-Detection

Production-grade hardware detection and configuration for MLX Inference Server.
Automatically detects RAM and chip type, then applies appropriate configuration.

Developed with Claude Code and refined through iterative testing.
Works on any Apple Silicon Mac (M1/M2/M3/M4).

Configuration:
- Main API port: 11440
- Admin API port: 11441
- Process isolation architecture
- Auto-tuning based on available RAM
"""

import os
import subprocess
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging


logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """
    Server configuration with automatic hardware detection.

    Detects hardware via sysctl and system_profiler, then applies
    appropriate configuration based on available RAM.

    Works on any Apple Silicon Mac (M1/M2/M3/M4).
    All settings overridable via environment variables.
    """

    # Network
    main_port: int
    admin_port: int
    host: str

    # Timeouts (seconds)
    idle_timeout_seconds: int
    request_timeout_seconds: int

    # Memory (GB)
    memory_threshold_gb: int

    # Paths
    cache_dir: str
    log_dir: str

    # PHASE 2: IPC Method (Opus 4.5 Performance Optimization)
    use_shared_memory: bool  # True = POSIX shared memory (fast), False = stdin/stdout (fallback)

    # Machine detection
    machine_type: str  # "high-memory", "medium-memory", "low-memory"

    # Hardware info (for logging/debugging)
    total_ram_gb: int
    chip_model: str
    model_name: str

    @staticmethod
    def _get_ram_gb() -> int:
        """
        Get total RAM in GB via sysctl.

        Returns:
            Total RAM in GB (rounded)
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True
            )
            ram_bytes = int(result.stdout.strip())
            ram_gb = round(ram_bytes / (1024**3))
            return ram_gb
        except Exception as e:
            logger.warning(f"Failed to detect RAM: {e}, defaulting to 32GB")
            return 32

    @staticmethod
    def _get_chip_model() -> str:
        """
        Get chip model (e.g., "Apple M4 Max", "Apple M1").

        Returns:
            Chip model string or "Unknown"
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            chip = result.stdout.strip()
            return chip if chip else "Unknown"
        except Exception as e:
            logger.warning(f"Failed to detect chip: {e}")
            return "Unknown"

    @staticmethod
    def _get_model_name() -> str:
        """
        Get model name (e.g., "Mac Studio", "MacBook Air").

        Uses system_profiler for accurate model detection.

        Returns:
            Model name or "Unknown Mac"
        """
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            # Parse "Model Name: Mac Studio"
            match = re.search(r"Model Name:\s+(.+)", result.stdout)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.warning(f"Failed to detect model name: {e}")

        return "Unknown Mac"

    @staticmethod
    def _detect_machine_profile(
        ram_gb: int,
        model_name: str,
        chip_model: str
    ) -> Tuple[str, Dict]:
        """
        Detect appropriate machine profile based on available RAM.

        Auto-configures timeouts and memory thresholds based on system RAM.
        Works on any Apple Silicon Mac (M1/M2/M3/M4).

        Profiles (RAM-based tiers):
        - high-memory: ≥64GB RAM (server/production workloads)
        - medium-memory: 24-64GB RAM (developer/power user)
        - low-memory: <24GB RAM (conservative settings)

        Args:
            ram_gb: Total RAM in GB
            model_name: Model name (for logging only)
            chip_model: Chip model (for logging only)

        Returns:
            Tuple of (profile_name, config_dict)
        """
        # Default cache directory (overridable via HF_HOME env var)
        default_cache = os.path.expanduser("~/.cache/huggingface")

        # High-memory tier: ≥64GB (server/production)
        if ram_gb >= 64:
            return "high-memory", {
                "idle_timeout_seconds": 600,      # 10 minutes
                "request_timeout_seconds": 600,   # 10 minutes (long contexts)
                "memory_threshold_gb": ram_gb - 20,  # Leave 20GB for system
                "cache_dir": default_cache,
                "description": f"High-memory configuration ({ram_gb}GB RAM)"
            }

        # Medium-memory tier: 24-63GB (developer/power user)
        elif ram_gb >= 24:
            return "medium-memory", {
                "idle_timeout_seconds": 300,      # 5 minutes
                "request_timeout_seconds": 300,   # 5 minutes
                "memory_threshold_gb": ram_gb - 8,  # Leave 8GB for system
                "cache_dir": default_cache,
                "description": f"Medium-memory configuration ({ram_gb}GB RAM)"
            }

        # Low-memory tier: <24GB (conservative)
        else:
            return "low-memory", {
                "idle_timeout_seconds": 180,      # 3 minutes
                "request_timeout_seconds": 180,   # 3 minutes
                "memory_threshold_gb": max(8, ram_gb - 4),  # Leave 4GB for system
                "cache_dir": default_cache,
                "description": f"Low-memory configuration ({ram_gb}GB RAM)"
            }

    @classmethod
    def auto_detect(cls) -> 'ServerConfig':
        """
        Auto-detect hardware and load appropriate configuration.

        Detection process:
        1. Detect total RAM via sysctl
        2. Detect chip model via sysctl
        3. Detect model name via system_profiler
        4. Apply RAM-based configuration tier
        5. Allow environment variable overrides

        Environment variable overrides:
        - MLX_IDLE_TIMEOUT: Override idle timeout (seconds)
        - MLX_REQUEST_TIMEOUT: Override request timeout (seconds)
        - MLX_ADMIN_PORT: Override admin port (default: 11441)
        - HF_HOME: Override model cache directory
        - MLX_LOG_DIR: Override log directory
        - MLX_MACHINE_TYPE: Force specific profile tier

        Returns:
            ServerConfig instance with detected/configured values
        """
        # Detect hardware
        ram_gb = cls._get_ram_gb()
        chip_model = cls._get_chip_model()
        model_name = cls._get_model_name()

        logger.info(f"Hardware detected: {model_name}, {chip_model}, {ram_gb}GB RAM")

        # Detect machine profile
        machine_type, default_config = cls._detect_machine_profile(
            ram_gb, model_name, chip_model
        )

        # Allow manual override
        machine_type = os.getenv("MLX_MACHINE_TYPE", machine_type)

        logger.info(
            f"Machine profile: {machine_type} - {default_config.get('description', 'N/A')}"
        )

        # Apply environment variable overrides
        idle_timeout = int(
            os.getenv("MLX_IDLE_TIMEOUT", str(default_config["idle_timeout_seconds"]))
        )
        request_timeout = int(
            os.getenv("MLX_REQUEST_TIMEOUT", str(default_config["request_timeout_seconds"]))
        )
        admin_port = int(os.getenv("MLX_ADMIN_PORT", "11441"))  # V3 Admin API port
        cache_dir = os.getenv("HF_HOME", default_config["cache_dir"])
        log_dir = os.getenv(
            "MLX_LOG_DIR",
            os.path.join(os.getcwd(), "logs")
        )

        # PHASE 2: Shared memory IPC (Opus 4.5 optimization)
        # Enable by default, can disable for debugging with MLX_USE_STDIO=1
        use_shared_memory = os.getenv("MLX_USE_STDIO", "0") != "1"

        # Create configuration
        config = cls(
            main_port=11440,  # V3 Main API port (parallel with V2 on 11436)
            admin_port=admin_port,
            host="0.0.0.0",
            idle_timeout_seconds=idle_timeout,
            request_timeout_seconds=request_timeout,
            memory_threshold_gb=default_config["memory_threshold_gb"],
            cache_dir=cache_dir,
            log_dir=log_dir,
            use_shared_memory=use_shared_memory,  # PHASE 2: Shared memory IPC
            machine_type=machine_type,
            total_ram_gb=ram_gb,
            chip_model=chip_model,
            model_name=model_name
        )

        # Ensure directories exist
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Set environment variables for HuggingFace
        os.environ["HF_HOME"] = config.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = config.cache_dir

        logger.info(f"Configuration applied: {config.machine_type}")
        logger.info(f"Cache directory: {config.cache_dir}")
        logger.info(f"Idle timeout: {config.idle_timeout_seconds}s")
        logger.info(f"Memory threshold: {config.memory_threshold_gb}GB")
        logger.info(f"IPC method: {'Shared Memory' if config.use_shared_memory else 'stdio'} (Phase 2)")

        return config

    def __str__(self) -> str:
        """Human-readable configuration display."""
        return f"""
MLX Server V2 Configuration (NASA-Grade Detection)
===================================================
Hardware Detected:
  Model Name:       {self.model_name}
  Chip:             {self.chip_model}
  Total RAM:        {self.total_ram_gb} GB

Profile Applied:
  Machine Type:     {self.machine_type.upper()}

Network:
  Main Port:        {self.main_port}
  Admin Port:       {self.admin_port}
  Host:             {self.host}

Timeouts:
  Idle Timeout:     {self.idle_timeout_seconds}s ({self.idle_timeout_seconds/60:.1f} min)
  Request Timeout:  {self.request_timeout_seconds}s

Memory:
  Threshold:        {self.memory_threshold_gb} GB

Paths:
  Cache Directory:  {self.cache_dir}
  Log Directory:    {self.log_dir}
===================================================
        """.strip()

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "machine_type": self.machine_type,
            "main_port": self.main_port,
            "admin_port": self.admin_port,
            "host": self.host,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "memory_threshold_gb": self.memory_threshold_gb,
            "cache_dir": self.cache_dir,
            "log_dir": self.log_dir,
            "use_shared_memory": self.use_shared_memory,
            "total_ram_gb": self.total_ram_gb,
            "chip_model": self.chip_model,
            "model_name": self.model_name
        }
