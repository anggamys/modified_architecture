"""
Central configuration management untuk entire application.
Implements configuration hierarchy: YAML -> Config Classes -> Usage
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch


@dataclass
class DataConfig:
    """Configuration untuk data loading & preprocessing"""

    data_path: str
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Character vocabulary
    min_char_freq: int = 5
    include_emoji: bool = False
    max_word_len: int = 50
    max_seq_len: int = 512

    # Data splits stratification
    stratify_by_mlu: bool = True
    mlu_threshold_short: float = 5.0
    mlu_threshold_medium: float = 8.0


@dataclass
class ModelConfig:
    """Configuration untuk model architecture"""

    # BERT
    bert_model: str = "dafqi/IndoBertTweet"
    bert_freeze_layers: int = 2

    # Character encoder
    char_emb_dim: int = 32
    char_num_filters: int = 96
    char_kernel_sizes: tuple = field(default_factory=lambda: (2, 3, 4, 5))
    char_output_dim: int = 128
    char_cnn_dropout: float = 0.35

    # Optional BiLSTM
    use_word_bilstm: bool = False
    bilstm_hidden_dim: int = 128
    bilstm_num_layers: int = 2
    bilstm_dropout: float = 0.3

    # CRF
    use_crf: bool = True

    # Loss function
    use_focal_loss: bool = True
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # Loss weights
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"crf": 0.60, "focal": 0.25, "cross_entropy": 0.15}
    )

    # Class weights normalization
    class_weight_min: float = 0.5
    class_weight_max: float = 2.0


@dataclass
class TrainingConfig:
    """Configuration untuk training loop"""

    epochs: int = 20
    batch_size: int = 16
    learning_rate_bert: float = 1e-5
    learning_rate_other: float = 1e-3
    learning_rate_crf: float = 5e-3
    weight_decay: float = 0.05

    # Warmup & scheduling
    warmup_ratio: float = 0.1

    # Early stopping
    patience: int = 3

    # Gradient
    gradient_clip_norm: float = 1.0

    # Device
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OutputConfig:
    """Configuration untuk output & checkpointing"""

    output_dir: str = "outputs"
    model_name: str = "M6"
    save_best_only: bool = True
    save_vocab: bool = True
    save_class_mapping: bool = True
    save_test_results: bool = True
    save_classification_report: bool = True

    @property
    def model_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name

    def create_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration untuk logging"""

    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    def create_dirs(self) -> None:
        self.log_path.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Root configuration container"""

    data: DataConfig = field(default_factory=lambda: DataConfig(data_path=""))
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    output: OutputConfig = field(default_factory=lambda: OutputConfig())
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        """Load configuration dari YAML file"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Load configuration dari dictionary"""
        config = cls()

        # Update data config
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        # Update model config
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        # Update training config
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)

        # Update output config
        if "output" in config_dict:
            for key, value in config_dict["output"].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)

        # Update logging config
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration ke dictionary"""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "output": asdict(self.output),
            "logging": asdict(self.logging),
        }

    def setup(self) -> None:
        """Initialize directories & setup required resources"""
        self.output.create_dirs()
        self.logging.create_dirs()


# Default configuration (dapat di-override via YAML atau programmatically)
DEFAULT_CONFIG = AppConfig()
