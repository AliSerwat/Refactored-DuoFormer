#!/usr/bin/env python3
"""
🚀 Refactored DuoFormer Training Script

Professional training script for general medical image classification.
Refactored with configuration management and MLOps best practices.

Features:
- Configuration-based training
- Platform & hardware agnostic
- Automatic checkpointing
- TensorBoard logging
- Mixed precision training
- Early stopping

Usage:
    python train.py --config config/default_config.yaml
    python train.py --data_dir /path/to/data --epochs 100

Original work: https://github.com/xiaoyatang/duoformer_TCGA
Paper: https://arxiv.org/abs/2506.12982
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import random
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import build_model_no_extra_params, count_parameters
from utils import (
    Trainer,
    create_optimizer,
    create_scheduler,
    create_dataloaders,
    setup_device_environment,
    get_optimal_num_workers,
    print_device_info,
)
from config import ModelConfig, DEFAULT_CONFIG


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DuoFormer on medical images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (YAML or JSON)"
    )

    # Data
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--csv_file", type=str, help="Path to CSV file with data")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="r50",
        choices=["r50", "r18", "r50_Swav"],
        help="Backbone architecture",
    )
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument(
        "--embed_dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of scales (2-4)"
    )
    parser.add_argument(
        "--proj_dim", type=int, default=768, help="Projection dimension"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained backbone",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "onecycle", "none"],
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Mixed precision
    parser.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Experiment
    parser.add_argument(
        "--exp_name", type=str, default="duoformer_exp", help="Experiment name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cuda/cpu/cuda:0/mps) - auto selects best available",
    )
    parser.add_argument(
        "--gpu_ids", type=str, help="Comma-separated GPU IDs (e.g., 0,1,2)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set CUDA visible devices if specified
    if args.gpu_ids:
        from utils import set_cuda_visible_devices

        set_cuda_visible_devices(args.gpu_ids)
        print(f"🎯 Restricting to GPUs: {args.gpu_ids}")

    # Load configuration
    if args.config:
        print(f"📄 Loading configuration from {args.config}")
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = ModelConfig.from_yaml(args.config)
        elif args.config.endswith(".json"):
            config = ModelConfig.from_json(args.config)
        else:
            raise ValueError("Config file must be .yaml or .json")
    else:
        print("📄 Using default configuration")
        config = DEFAULT_CONFIG

    # Override config with command line arguments
    if args.data_dir:
        config.data.data_dir = Path(args.data_dir)
    if args.num_classes:
        config.data.num_classes = args.num_classes
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    # Set random seed
    set_seed(args.seed)
    print(f"🎲 Random seed set to {args.seed}")

    # Setup device (hardware-agnostic)
    print_device_info()
    device = setup_device_environment(
        preferred_device=args.device if args.device != "auto" else "auto",
        verbose=False,  # Already printed above
    )

    # Get optimal number of workers based on platform
    if args.num_workers == 4:  # Default value
        args.num_workers = get_optimal_num_workers(max_workers=8)
        print(f"📊 Optimal num_workers for this platform: {args.num_workers}")

    # Print configuration
    print("\n" + "=" * 80)
    print("⚙️  Configuration")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Depth: {args.depth}, Embed Dim: {args.embed_dim}, Heads: {args.num_heads}")
    print(f"Scales: {args.num_layers}, Proj Dim: {args.proj_dim}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"Optimizer: {args.optimizer}, Scheduler: {args.scheduler}")
    print(f"Mixed Precision: {args.amp}")
    print("=" * 80 + "\n")

    # Create data loaders
    print("📊 Creating data loaders...")

    # Interactive data path input if not provided
    if not args.data_dir and not args.csv_file:
        print("\n" + "=" * 80)
        print("📁 DATA DIRECTORY SETUP REQUIRED")
        print("=" * 80)
        print("\n🔍 No data directory or CSV file provided!")
        print("Please specify your data location using one of these methods:\n")

        print("📂 METHOD 1: Directory Structure (Recommended)")
        print("\n🌳 REQUIRED DATA STRUCTURE:")
        print("   ┌─────────────────────────────────────────────────────────┐")
        print("   │                    your_data_folder/                    │")
        print("   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐ │")
        print("   │  │   class_name_1  │  │   class_name_2  │  │ class_3 │ │")
        print("   │  │  ┌─────────────┐ │  │  ┌─────────────┐ │ │ ┌─────┐ │ │")
        print("   │  │  │ image1.png  │ │  │  │ image1.jpg  │ │ │ │img1 │ │ │")
        print("   │  │  │ image2.jpg  │ │  │  │ image2.tif  │ │ │ │img2 │ │ │")
        print("   │  │  │ image3.tif  │ │  │  │ image3.png  │ │ │ │img3 │ │ │")
        print("   │  │  └─────────────┘ │  │  └─────────────┘ │ │ └─────┘ │ │")
        print("   │  └─────────────────┘  └─────────────────┘  └─────────┘ │")
        print("   └─────────────────────────────────────────────────────────┘")
        print("\n📋 STRUCTURE RULES:")
        print("   ✅ Each class = separate folder")
        print("   ✅ Class names = descriptive (benign, malignant, normal)")
        print("   ✅ Images = PNG, JPG, JPEG, TIF, TIFF formats")
        print("   ✅ No spaces in class folder names (use underscores)")
        print("\n💡 EXAMPLE FOR MEDICAL IMAGING:")
        print("   medical_data/")
        print("   ├── benign/")
        print("   │   ├── patient_001.png")
        print("   │   └── patient_002.jpg")
        print("   ├── malignant/")
        print("   │   ├── tumor_001.tif")
        print("   │   └── tumor_002.png")
        print("   └── normal/")
        print("       └── healthy_001.jpg")

        print("\n📄 METHOD 2: CSV File")
        print("   Your CSV should have these columns:")
        print("   image_path,label")
        print("   /path/to/image1.png,class_0")
        print("   /path/to/image2.jpg,class_1")
        print("   /path/to/image3.tif,class_2")

        print("\n" + "=" * 80)

        # Get user input
        while True:
            print("\n🎯 Choose your data input method:")
            print("   1. Directory structure")
            print("   2. CSV file")
            print("   3. Exit and use demo notebook")

            choice = input("\nEnter your choice (1/2/3): ").strip()

            if choice == "1":
                data_path = input(
                    "\n📁 Enter the path to your data directory: "
                ).strip()
                if data_path:
                    # Validate directory exists
                    data_path_obj = Path(data_path)
                    if data_path_obj.exists() and data_path_obj.is_dir():
                        args.data_dir = data_path
                        print(f"✅ Using data directory: {data_path}")
                        break
                    else:
                        print(f"❌ Directory not found: {data_path}")
                        print("   Please check the path and try again.")
                        continue
                else:
                    print("❌ Please enter a valid path.")
                    continue

            elif choice == "2":
                csv_path = input("\n📄 Enter the path to your CSV file: ").strip()
                if csv_path:
                    # Validate CSV exists
                    csv_path_obj = Path(csv_path)
                    if csv_path_obj.exists() and csv_path_obj.is_file():
                        args.csv_file = csv_path
                        print(f"✅ Using CSV file: {csv_path}")
                        break
                    else:
                        print(f"❌ CSV file not found: {csv_path}")
                        print("   Please check the path and try again.")
                        continue
                else:
                    print("❌ Please enter a valid path.")
                    continue

            elif choice == "3":
                print("\n🔬 For demo purposes, try:")
                print("   jupyter notebook demo_duoformer.ipynb")
                return
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                continue

        # Ask for number of classes if not specified
        if not args.num_classes or args.num_classes == 10:
            print(f"\n📊 Current number of classes: {args.num_classes}")
            num_classes_input = input(
                "Enter number of classes (or press Enter to keep current): "
            ).strip()
            if num_classes_input.isdigit():
                args.num_classes = int(num_classes_input)
                print(f"✅ Number of classes set to: {args.num_classes}")

        print("\n" + "=" * 80)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            csv_file=args.csv_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("\n💡 Please check:")
        if args.data_dir:
            print(f"   • Data directory exists: {args.data_dir}")
            print("   • Directory contains class subfolders")
            print("   • Each class folder contains image files")
        if args.csv_file:
            print(f"   • CSV file exists: {args.csv_file}")
            print("   • CSV has 'image_path' and 'label' columns")
        return
    except ValueError as e:
        print(f"❌ Data structure error: {e}")
        print("\n💡 Data structure requirements:")
        if args.data_dir:
            print("   📂 Directory structure:")
            print("   ┌─────────────────────────────────────────────────────────┐")
            print("   │                    your_data_folder/                    │")
            print("   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐ │")
            print("   │  │   class_name_1  │  │   class_name_2  │  │ class_3 │ │")
            print("   │  │  ┌─────────────┐ │  │  ┌─────────────┐ │ │ ┌─────┐ │ │")
            print("   │  │  │ image1.png  │ │  │  │ image1.jpg  │ │ │ │img1 │ │ │")
            print("   │  │  │ image2.jpg  │ │  │  │ image2.tif  │ │ │ │img2 │ │ │")
            print("   │  │  │ image3.tif  │ │  │  │ image3.png  │ │ │ │img3 │ │ │")
            print("   │  │  └─────────────┘ │  │  └─────────────┘ │ │ └─────┘ │ │")
            print("   │  └─────────────────┘  └─────────────────┘  └─────────┘ │")
            print("   └─────────────────────────────────────────────────────────┘")
        if args.csv_file:
            print("   📄 CSV format:")
            print("   image_path,label")
            print("   /path/to/image1.png,class_name_1")
            print("   /path/to/image2.jpg,class_name_2")
        return
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("\n💡 Common issues:")
        print("   • Invalid image formats (use PNG, JPG, JPEG, TIF, TIFF)")
        print("   • Missing 'image_path' or 'label' columns in CSV")
        print("   • Empty directories or no images found")
        print("   • Corrupted image files")
        return

    # Build model
    print("🏗️  Building model...")
    model = build_model_no_extra_params(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        num_patches=49,  # 7x7 patches
        proj_dim=args.proj_dim,
        freeze_backbone=args.freeze_backbone,
        backbone=args.backbone,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    # Count parameters
    trainable_params, total_params = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"   Trainable parameters: {trainable_params:.2f}M")
    print(f"   Total parameters: {total_params:.2f}M")
    print(f"   Model name: {model.name}\n")

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model, optimizer_name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        config={"exp_name": args.exp_name},
        checkpoint_dir=Path(args.checkpoint_dir) / args.exp_name,
        use_amp=args.amp,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\n✨ Training complete!")
    print(f"💾 Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"📊 TensorBoard logs: {trainer.writer.log_dir}")
    print("\nTo view tensorboard:")
    print(f"   tensorboard --logdir={trainer.writer.log_dir}")


if __name__ == "__main__":
    main()
