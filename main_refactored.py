"""
Refactored main.py - Clean orchestration using application layer.
All the complexity is now abstracted into services and configuration.
"""

import sys
from pathlib import Path
from config import AppConfig
from application import ApplicationOrchestrator
from utils import log, log_level

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_training(config_path: str | None = None) -> None:
    """
    Execute training pipeline with given configuration.

    Args:
        config_path: Path ke YAML config file. Jika None, gunakan default config.
    """

    # Load configuration
    if config_path:
        log(
            domain="Main",
            msg=f"Loading configuration dari: {config_path}",
            level=log_level.INFO,
        )
        config = AppConfig.from_yaml(config_path)
    else:
        log(
            domain="Main",
            msg="Using default configuration",
            level=log_level.INFO,
        )
        config = AppConfig()

    # Log configuration
    log(
        domain="Main",
        msg=f"Device: {config.training.device}",
        level=log_level.INFO,
    )
    log(
        domain="Main",
        msg=f"Output directory: {config.output.model_dir}",
        level=log_level.INFO,
    )
    log(
        domain="Main",
        msg=f"Data path: {config.data.data_path}",
        level=log_level.INFO,
    )

    # Execute pipeline
    orchestrator = ApplicationOrchestrator(config)

    try:
        results = orchestrator.run_pipeline()
        log(
            domain="Main",
            msg=f"Training completed successfully. Test accuracy: {results['evaluation']['accuracy']:.4f}",
            level=log_level.INFO,
        )
    except Exception as e:
        log(
            domain="Main",
            msg=f"Error during training: {str(e)}",
            level=log_level.ERROR,
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="POS Tagging Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()
    run_training(args.config)
