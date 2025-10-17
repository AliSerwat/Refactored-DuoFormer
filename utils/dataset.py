"""
📊 Refactored DuoFormer Dataset Utilities

Dataset handling for general medical image classification tasks.
Works with various medical imaging modalities: histopathology, radiology, dermatology, etc.

Features:
- General medical image dataset support
- Flexible dataset loader (directory or CSV)
- Medical image augmentation pipeline
- Automatic train/val/test splitting

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings


class MedicalImageDataset(Dataset):
    """
    General Medical Image Dataset.

    Supports various medical imaging modalities:
    - Histopathology (H&E, IHC, whole slide images)
    - Radiology (X-ray, CT, MRI)
    - Dermatology (skin lesions, dermoscopy)
    - Ophthalmology (retinal images, OCT)
    - And more...

    Expected directory structure:
        data_dir/
            class_0/
                image1.png
                image2.png
                ...
            class_1/
                image1.png
                ...

    Or CSV format:
        image_path, label
        /path/to/image1.png, 0
        /path/to/image2.png, 1
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        csv_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        """
        Initialize medical image dataset.

        Args:
            data_dir: Root directory with class subfolders
            csv_file: CSV file with image paths and labels
            transform: Image transformations
            target_transform: Label transformations
            image_size: Target image size
        """
        self.transform = transform or self._default_transform(image_size)
        self.target_transform = target_transform
        self.image_size = image_size

        # Load data
        if csv_file is not None:
            self.data = self._load_from_csv(csv_file)
        elif data_dir is not None:
            self.data = self._load_from_directory(data_dir)
        else:
            raise ValueError("Either data_dir or csv_file must be provided")

        # Class information
        self.classes = sorted(set(self.data["label"]))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        print(f"📊 Dataset loaded:")
        print(f"   Total samples: {len(self.data)}")
        print(f"   Number of classes: {self.num_classes}")
        print(f"   Classes: {self.classes}")

    def _load_from_csv(self, csv_file: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(csv_file)
        assert "image_path" in df.columns, "CSV must have 'image_path' column"
        assert "label" in df.columns, "CSV must have 'label' column"
        return df

    def _load_from_directory(self, data_dir: str) -> pd.DataFrame:
        """Load dataset from directory structure."""
        data_dir = Path(data_dir)
        data = {"image_path": [], "label": []}

        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff",
                ]:
                    data["image_path"].append(str(img_path))
                    data["label"].append(class_name)

        return pd.DataFrame(data)

    def _default_transform(self, image_size: int) -> transforms.Compose:
        """Default image transformations."""
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        row = self.data.iloc[idx]

        # Load image
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (self.image_size, self.image_size))

        # Get label
        label = row["label"]
        if isinstance(label, str):
            label = self.class_to_idx[label]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes."""
        return dict(self.data["label"].value_counts())


class MedicalImageAugmentation:
    """
    Medical image augmentation pipeline.

    Includes augmentations suitable for various medical imaging modalities:
    - Histopathology: rotation, flip, color jitter
    - Radiology: rotation, contrast, brightness
    - Dermatology: flip, rotation, color
    - General: configurable augmentation strategy
    """

    @staticmethod
    def train_transform(
        image_size: int = 224,
        random_flip: bool = True,
        random_rotation: int = 10,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ) -> transforms.Compose:
        """
        Training augmentation pipeline.

        Args:
            image_size: Target image size
            random_flip: Apply random horizontal flip
            random_rotation: Max rotation degrees
            color_jitter: Apply color jitter
            brightness: Brightness jitter
            contrast: Contrast jitter

        Returns:
            Composed transformations
        """
        transform_list = [transforms.Resize((image_size, image_size))]

        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if random_rotation > 0:
            transform_list.append(transforms.RandomRotation(degrees=random_rotation))

        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast)
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return transforms.Compose(transform_list)

    @staticmethod
    def val_transform(image_size: int = 224) -> transforms.Compose:
        """
        Validation/test augmentation pipeline.

        Args:
            image_size: Target image size

        Returns:
            Composed transformations
        """
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_dataloaders(
    data_dir: Optional[str] = None,
    csv_file: Optional[str] = None,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    use_augmentation: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.

    Args:
        data_dir: Root directory with class subfolders
        csv_file: CSV file with image paths and labels
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        image_size: Target image size
        use_augmentation: Use data augmentation for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, (
        "Splits must sum to 1.0"
    )

    # Auto-detect optimal num_workers if not specified
    if num_workers is None:
        from .device_utils import get_optimal_num_workers

        num_workers = get_optimal_num_workers()
        print(
            f"📊 Auto-detected num_workers: {num_workers} (based on platform and CPU cores)"
        )

    # Create transforms
    if use_augmentation:
        train_transform = MedicalImageAugmentation.train_transform(image_size)
    else:
        train_transform = MedicalImageAugmentation.val_transform(image_size)

    val_transform = MedicalImageAugmentation.val_transform(image_size)

    # Load full dataset with train transform initially
    full_dataset = MedicalImageDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=train_transform,
        image_size=image_size,
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Random split
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(
        range(total_size), [train_size, val_size, test_size], generator=generator
    )

    # Create datasets with appropriate transforms
    train_dataset = Subset(full_dataset, train_indices)

    # For val/test, create new datasets with val transform
    val_dataset_full = MedicalImageDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=val_transform,
        image_size=image_size,
    )
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(val_dataset_full, test_indices)

    # Get platform-specific settings if pin_memory not explicitly set
    if pin_memory is True:
        from .platform_utils import get_platform_specific_settings

        platform_settings = get_platform_specific_settings()
        pin_memory = platform_settings["pin_memory"]
        persistent_workers = (
            platform_settings.get("persistent_workers", False)
            if num_workers > 0
            else False
        )
    else:
        persistent_workers = False

    # Create data loaders with platform-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(f"\n📦 Data Loaders Created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}\n")

    return train_loader, val_loader, test_loader


def build_dataset(
    dataset_name: str, params: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataset (for backward compatibility).

    Args:
        dataset_name: Name of dataset ('medical_imaging', 'TCGA', etc.)
        params: Dictionary of parameters

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Support various dataset names for flexibility
    supported = ["MEDICAL_IMAGING", "TCGA", "HISTOPATHOLOGY", "RADIOLOGY", "MEDICAL"]

    if dataset_name.upper() in supported:
        return create_dataloaders(
            data_dir=params.get("data_dir"),
            csv_file=params.get("csv_file"),
            batch_size=params.get("batch_size", 32),
            num_workers=params.get("num_workers", None),  # Auto-detect
            use_augmentation=params.get("dataAug", True),
        )
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not implemented. "
            f"Supported: {', '.join(supported)}"
        )
