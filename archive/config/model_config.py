"""
🔧 Refactored DuoFormer Model Configuration

Centralized configuration management for DuoFormer models.
Refactored for general medical imaging applications.

All hyperparameters, paths, and settings are managed here.

Original work: https://github.com/xiaoyatang/duoformer_TCGA
Paper: https://arxiv.org/abs/2506.12982
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class BackboneConfig:
    """Configuration for backbone network."""

    name: str = "resnet50"  # resnet50, resnet18, resnet50_swav
    pretrained: bool = True
    freeze: bool = True
    weights: Optional[str] = "DEFAULT"  # 'DEFAULT', 'IMAGENET1K_V1', etc.

    def __post_init__(self):
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

    def __post_init__(self):
        """Validate configuration."""
        assert self.depth > 0, "Depth must be positive"
        assert self.embed_dim > 0, "Embedding dimension must be positive"
        assert self.num_heads > 0, "Number of heads must be positive"
        assert self.embed_dim % self.num_heads == 0, (
            "Embed_dim must be divisible by num_heads"
        )


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale processing."""

    num_layers: int = 2  # Number of scales (2, 3, or 4)
    num_patches: int = 49  # Number of patches (7x7 = 49)
    proj_dim: int = 768  # Projection dimension
    scale_token: str = "random"  # 'random' or 'channel'
    patch_attn: bool = True  # Whether to use patch attention

    def __post_init__(self):
        """Validate configuration."""
        assert 2 <= self.num_layers <= 4, "num_layers must be between 2 and 4"
        assert self.scale_token in [
            "random",
            "channel",
        ], "scale_token must be 'random' or 'channel'"


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


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    # Logging
    log_dir: Path = Path("./logs")
    tensorboard_dir: Path = Path("./runs")
    log_interval: int = 10  # Log every N batches

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "duoformer"
    wandb_entity: Optional[str] = None

    # MLflow
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "duoformer"


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
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
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
        logging = LoggingConfig(**config_dict.get("logging", {}))

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

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()

        # Convert pathlib.Path objects to strings for YAML serialization
        def convert_paths(obj):
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

    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

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
