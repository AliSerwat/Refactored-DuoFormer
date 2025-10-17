"""
🔧 Refactored DuoFormer Model Configuration

Centralized configuration management for DuoFormer models.
Refactored for general medical imaging applications.

All hyperparameters, paths, and settings are managed here.

Original work: https://github.com/xiaoyatang/duoformer_TCGA
Paper: https://arxiv.org/abs/2506.12982
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import yaml
import json
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


# Validation functions (inline to avoid circular imports)
def validate_positive_number(value, name, allow_zero=False):
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value}")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def validate_range(value, min_val, max_val, name):
    """Validate that a value is within a range."""
    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


@dataclass
class BackboneConfig:
    """Configuration for backbone network."""

    name: str = "resnet50"  # resnet50, resnet18, resnet50_swav
    pretrained: bool = True
    freeze: bool = True
    weights: Optional[str] = "DEFAULT"  # 'DEFAULT', 'IMAGENET1K_V1', etc.

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_backbones = ["resnet50", "resnet18", "resnet50_swav"]
        if self.name not in valid_backbones:
            raise ValueError(
                f"Invalid backbone: {self.name}. Choose from {valid_backbones}"
            )


@dataclass
class TransformerConfig:
    """Configuration for transformer architecture."""

    depth: int = 12  # Number of transformer blocks
    embed_dim: int = 768  # Embedding dimension
    num_heads: int = 12  # Number of attention heads
    mlp_ratio: float = 4.0  # MLP expansion ratio
    qkv_bias: bool = True
    qk_norm: bool = False
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    init_values: Optional[float] = None  # Layer scale init values

    def __post_init__(self) -> None:
        """Validate configuration."""
        validate_positive_number(self.depth, "depth")
        validate_positive_number(self.embed_dim, "embed_dim")
        validate_positive_number(self.num_heads, "num_heads")
        validate_positive_number(self.mlp_ratio, "mlp_ratio")
        validate_range(self.attn_drop_rate, 0.0, 1.0, "attn_drop_rate")
        validate_range(self.proj_drop_rate, 0.0, 1.0, "proj_drop_rate")
        validate_range(self.drop_path_rate, 0.0, 1.0, "drop_path_rate")

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"Embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale processing."""

    num_layers: int = 2  # Number of scales (2, 3, or 4)
    num_patches: int = 49  # Number of patches (7x7 = 49)
    proj_dim: int = 768  # Projection dimension
    scale_token: str = "random"  # 'random' or 'channel'
    patch_attn: bool = True  # Whether to use patch attention

    def __post_init__(self) -> None:
        """Validate configuration."""
        validate_range(self.num_layers, 2, 4, "num_layers")
        validate_positive_number(self.num_patches, "num_patches")
        validate_positive_number(self.proj_dim, "proj_dim")

        if self.scale_token not in ["random", "channel"]:
            raise ValueError(
                f"scale_token must be 'random' or 'channel', got {self.scale_token}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd'
    momentum: float = 0.9  # For SGD

    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', 'onecycle', None
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training
    epochs: int = 100
    batch_size: int = 32
    num_workers: Optional[int] = None  # None = auto-detect based on platform
    pin_memory: Optional[bool] = None  # None = auto-detect based on platform

    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # Checkpointing
    save_freq: int = 5  # Save every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        """Validate configuration."""
        validate_positive_number(self.learning_rate, "learning_rate")
        validate_positive_number(self.weight_decay, "weight_decay")
        validate_positive_number(self.epochs, "epochs")
        validate_positive_number(self.batch_size, "batch_size")
        validate_positive_number(self.warmup_epochs, "warmup_epochs", allow_zero=True)
        validate_positive_number(self.patience, "patience", allow_zero=True)
        validate_positive_number(self.save_freq, "save_freq")
        validate_positive_number(self.keep_last_n, "keep_last_n")

        validate_range(self.momentum, 0.0, 1.0, "momentum")
        validate_range(self.label_smoothing, 0.0, 1.0, "label_smoothing")
        validate_range(self.mixup_alpha, 0.0, 1.0, "mixup_alpha")
        validate_range(self.cutmix_alpha, 0.0, 1.0, "cutmix_alpha")

        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. Must be 'adam', 'adamw', or 'sgd'"
            )

        if self.scheduler not in ["cosine", "step", "onecycle", None]:
            raise ValueError(
                f"Invalid scheduler: {self.scheduler}. Must be 'cosine', 'step', 'onecycle', or None"
            )


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Dataset
    dataset_name: str = "medical_imaging"  # General medical imaging dataset
    data_dir: Path = Path("./data")
    num_classes: int = 10  # Adjust based on your classification task

    # Preprocessing
    image_size: int = 224
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation
    use_augmentation: bool = True
    random_flip: bool = True
    random_rotation: int = 10
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2

    # Splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    def __post_init__(self) -> None:
        """Validate configuration."""
        validate_positive_number(self.num_classes, "num_classes")
        validate_positive_number(self.image_size, "image_size")
        validate_positive_number(
            self.random_rotation, "random_rotation", allow_zero=True
        )

        validate_range(self.train_split, 0.0, 1.0, "train_split")
        validate_range(self.val_split, 0.0, 1.0, "val_split")
        validate_range(self.test_split, 0.0, 1.0, "test_split")
        validate_range(self.brightness, 0.0, 1.0, "brightness")
        validate_range(self.contrast, 0.0, 1.0, "contrast")

        # Check that splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    # Logging
    log_dir: Path = Path("./logs")
    tensorboard_dir: Path = Path("./runs")
    log_interval: int = 10  # Log every N batches

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "refactored-duoformer"
    wandb_entity: Optional[str] = None

    # MLflow
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "refactored-duoformer"

    def __post_init__(self) -> None:
        """Validate configuration."""
        validate_positive_number(self.log_interval, "log_interval")

        # Validate paths exist or can be created
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create logging directories: {e}")


@dataclass
class ModelConfig:
    """Complete model configuration."""

    # Model components
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Experiment
    exp_name: str = "duoformer_experiment"
    seed: int = 42
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'cuda:0', 'mps', etc.
    mixed_precision: bool = False  # Use AMP
    gpu_ids: Optional[str] = None  # Comma-separated GPU IDs (e.g., '0,1,2')

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        # Parse nested configurations
        backbone = BackboneConfig(**config_dict.get("backbone", {}))
        transformer = TransformerConfig(**config_dict.get("transformer", {}))
        multiscale = MultiScaleConfig(**config_dict.get("multiscale", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        data = DataConfig(**config_dict.get("data", {}))

        # Convert string paths back to Path objects before creating LoggingConfig
        logging_dict = config_dict.get("logging", {})
        if isinstance(logging_dict.get("log_dir"), str):
            logging_dict["log_dir"] = Path(logging_dict["log_dir"])
        if isinstance(logging_dict.get("tensorboard_dir"), str):
            logging_dict["tensorboard_dir"] = Path(logging_dict["tensorboard_dir"])

        logging = LoggingConfig(**logging_dict)

        return cls(
            backbone=backbone,
            transformer=transformer,
            multiscale=multiscale,
            training=training,
            data=data,
            logging=logging,
            exp_name=config_dict.get("exp_name", "duoformer_experiment"),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
            mixed_precision=config_dict.get("mixed_precision", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()

        # Convert pathlib.Path objects to strings for YAML serialization
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif hasattr(obj, "__fspath__"):  # pathlib.Path objects
                return str(obj)
            else:
                return obj

        config_dict = convert_paths(config_dict)

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate_schema(self) -> None:
        """Validate the entire configuration schema.

        Performs comprehensive validation of all configuration parameters
        including cross-parameter dependencies and constraints.

        Raises:
            ValidationError: If any configuration parameter is invalid
            ConfigurationError: If configuration has logical inconsistencies
        """
        # Validate all sub-configurations
        self.backbone.__post_init__()
        self.transformer.__post_init__()
        self.multiscale.__post_init__()
        self.training.__post_init__()
        self.data.__post_init__()
        self.logging.__post_init__()

        # Cross-parameter validation
        if self.training.batch_size <= 0:
            raise ValueError("Training batch_size must be positive")

        if self.training.epochs <= 0:
            raise ValueError("Training epochs must be positive")

        if self.training.learning_rate <= 0:
            raise ValueError("Training learning_rate must be positive")

        # Validate model architecture constraints
        if self.multiscale.num_layers < 2 or self.multiscale.num_layers > 4:
            raise ValueError("Multi-scale num_layers must be between 2 and 4")

        # Validate transformer constraints
        if self.transformer.embed_dim % self.transformer.num_heads != 0:
            raise ValueError(
                f"Transformer embed_dim ({self.transformer.embed_dim}) "
                f"must be divisible by num_heads ({self.transformer.num_heads})"
            )

    def __str__(self) -> str:
        """String representation."""
        return yaml.dump(self.to_dict(), default_flow_style=False)


# Default configurations for common use cases
DEFAULT_CONFIG = ModelConfig()

LIGHTWEIGHT_CONFIG = ModelConfig(
    backbone=BackboneConfig(name="resnet18", freeze=False),
    transformer=TransformerConfig(depth=6, embed_dim=384, num_heads=6),
    multiscale=MultiScaleConfig(num_layers=2, proj_dim=384),
    training=TrainingConfig(batch_size=64),
)

PERFORMANCE_CONFIG = ModelConfig(
    backbone=BackboneConfig(name="resnet50", freeze=False),
    transformer=TransformerConfig(depth=12, embed_dim=768, num_heads=12),
    multiscale=MultiScaleConfig(num_layers=4, proj_dim=768),
    training=TrainingConfig(batch_size=16, epochs=200),
)

FINETUNE_CONFIG = ModelConfig(
    backbone=BackboneConfig(name="resnet50", freeze=True),
    transformer=TransformerConfig(depth=12, embed_dim=768, num_heads=12),
    multiscale=MultiScaleConfig(num_layers=2, proj_dim=768),
    training=TrainingConfig(
        learning_rate=1e-5, batch_size=32, epochs=50, scheduler="cosine"
    ),
)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("DuoFormer Configuration Examples")
    print("=" * 80)

    # Default configuration
    print("\n1. Default Configuration:")
    print(DEFAULT_CONFIG)

    # Lightweight configuration
    print("\n2. Lightweight Configuration (for quick experiments):")
    print(LIGHTWEIGHT_CONFIG)

    # Save configuration
    DEFAULT_CONFIG.to_yaml("config/default_config.yaml")
    print("\n✅ Default configuration saved to config/default_config.yaml")
