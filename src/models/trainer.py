"""
Model training utilities.
"""

from typing import Optional, Dict, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR,
)
from .wrappers import ModelWrapper


class ModelTrainer:
    """
    Trainer class for classifier models.
    
    Handles training loop, validation, early stopping, and checkpointing.
    
    Args:
        model: PyTorch model to train
        model_name: Name for checkpointing
        device: Device to train on
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        
    Example:
        >>> model = get_model('resnet50')
        >>> trainer = ModelTrainer(model, 'resnet50')
        >>> history = trainer.train(train_loader, val_loader, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = DEVICE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        # Wrap model for consistent interface
        if not isinstance(model, ModelWrapper):
            self.model = ModelWrapper(model).to(device)
        else:
            self.model = model.to(device)
        
        self.model_name = model_name
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        print("=" * 60)
        print(f"Training: {self.model_name.upper()}")
        print("=" * 60)
        print(f"Device: {self.device} | Epochs: {num_epochs}")
        print()
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint()
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break
            
            print()
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n✅ Training completed in {elapsed:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint_path = CHECKPOINT_DIR / f"{self.model_name}_best.pth"
        
        # Save the inner model's state dict
        if hasattr(self.model, 'model'):
            torch.save(self.model.model.state_dict(), checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)
    
    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Evaluate model with full metrics.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
        }
        
        return metrics


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    **kwargs
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a model.
    
    Args:
        model_name: Name of model architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        **kwargs: Additional arguments for ModelTrainer
        
    Returns:
        Tuple of (trained model, training history)
    """
    from .factory import get_model
    
    model = get_model(model_name, pretrained=True)
    trainer = ModelTrainer(model, model_name, **kwargs)
    history = trainer.train(train_loader, val_loader, num_epochs)
    
    return trainer.model, history
