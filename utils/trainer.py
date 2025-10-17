"""
🎯 Refactored DuoFormer Training Utilities

Professional training pipeline with logging, checkpointing, and monitoring.
Refactored for general medical imaging applications.

Features:
- Automatic checkpointing with best model saving
- TensorBoard integration
- Progress bars and metrics tracking
- Mixed precision training (AMP)
- Gradient clipping
- Early stopping

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, Any
from tqdm import tqdm
import time
import json
from datetime import datetime
import numpy as np


class Trainer:
    """
    Professional trainer for DuoFormer models.

    Features:
    - Automatic checkpointing
    - Early stopping
    - Mixed precision training
    - TensorBoard logging
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        checkpoint_dir: Optional[Path] = None,
        use_amp: bool = False,
        gradient_clip_val: float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            checkpoint_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision
            gradient_clip_val: Max gradient norm for clipping
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup TensorBoard
        log_dir = Path(self.config.get("log_dir", "./runs")) / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        print(f"💾 Checkpoint directory: {self.checkpoint_dir}")
        print(f"📊 TensorBoard log directory: {log_dir}")
        print(f"⚡ Mixed precision training: {use_amp}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_clip_val
                    )

                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Log to TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar("Train/BatchLoss", loss.item(), self.global_step)
                self.writer.add_scalar("Train/BatchAcc", acc, self.global_step)

            self.global_step += 1

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc="Validation", leave=False)

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 20,
        save_best_only: bool = True,
    ):
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            patience: Patience for early stopping
            save_best_only: Only save best models
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 Starting Training")
        print(f"{'=' * 80}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Early stopping patience: {patience}")
        print(f"{'=' * 80}\n")

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Update history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)

            # Log to TensorBoard
            self.writer.add_scalar("Epoch/TrainLoss", train_loss, epoch)
            self.writer.add_scalar("Epoch/TrainAcc", train_acc, epoch)
            self.writer.add_scalar("Epoch/ValLoss", val_loss, epoch)
            self.writer.add_scalar("Epoch/ValAcc", val_acc, epoch)
            self.writer.add_scalar("Epoch/LearningRate", current_lr, epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n📅 Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")

            # Check if best model
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = True
                self.patience_counter = 0
                print(f"   ✨ New best validation accuracy: {val_acc:.2f}%")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Save checkpoint
            if improved or not save_best_only:
                self.save_checkpoint(
                    epoch=epoch, val_loss=val_loss, val_acc=val_acc, is_best=improved
                )

            # Early stopping
            if not improved:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                    break

        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"🎉 Training Complete!")
        print(f"{'=' * 80}")
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'=' * 80}\n")

        # Save final history
        self.save_history()

        # Close TensorBoard writer
        self.writer.close()

    def save_checkpoint(
        self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": self.config,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"   💾 Best model saved to {best_path}")

        # Save epoch-specific checkpoint
        if (epoch + 1) % self.config.get("save_freq", 5) == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint.get("history", self.history)

        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        print(f"   Resuming from epoch {self.current_epoch + 1}")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 Training history saved to {history_path}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs,
) -> optim.Optimizer:
    """
    Create optimizer.

    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    **kwargs,
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler
        epochs: Total epochs
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance or None
    """
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    if scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=kwargs.get("min_lr", 1e-6)
        )
    elif scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_name.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            verbose=True,
        )
    elif scheduler_name.lower() == "onecycle":
        steps_per_epoch = kwargs.get("steps_per_epoch", 100)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", optimizer.defaults["lr"]),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
