# ============================================================================
# 파일: trainer.py
# 설명: 모델 학습 및 평가 (콜백 기반)
# ============================================================================

import copy
import logging
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

EpochCallback = Callable[[dict], None]
StopCheck = Callable[[], bool]


class Trainer:
    """모델 학습 관리자

    아티팩트(model.pt + scaler + meta) 저장은 호출자(main.py 또는 FastAPI 서비스) 책임.
    Trainer는 best state_dict를 메모리에서 보관하며 외부에 노출한다.
    """

    def __init__(self, model: nn.Module, config) -> None:
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )

        self.history: dict[str, list[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        self.best_val_acc: float = 0.0
        self.best_val_loss: float = float('inf')
        self.best_state_dict: Optional[dict] = None
        self.best_epoch: int = -1
        self._epochs_since_improve = 0

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return running_loss / len(train_loader), 100.0 * correct / total

    def evaluate(self, loader: DataLoader) -> tuple[float, float, list, list]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: list = []
        all_labels: list = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return running_loss / len(loader), 100.0 * correct / total, all_preds, all_labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        on_epoch_end: Optional[EpochCallback] = None,
        should_stop: Optional[StopCheck] = None,
    ) -> None:
        """전체 학습 프로세스. early stopping + 외부 cancel + epoch 콜백 지원."""
        logger.info('Training on %s', self.device)
        patience = self.config.EARLY_STOPPING_PATIENCE
        min_delta = self.config.EARLY_STOPPING_MIN_DELTA

        for epoch in range(epochs):
            if should_stop and should_stop():
                logger.info('Training cancelled by external request at epoch %d', epoch + 1)
                break

            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            elapsed_ms = int((time.time() - t0) * 1000)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            logger.info(
                'Epoch [%d/%d] - Train Loss: %.4f, Train Acc: %.2f%% - Val Loss: %.4f, Val Acc: %.2f%%',
                epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc,
            )
            if new_lr != prev_lr:
                logger.info('  → Learning rate reduced: %.2e → %.2e', prev_lr, new_lr)

            improved_loss = val_loss < (self.best_val_loss - min_delta)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                logger.info('  → New best (Val Acc: %.2f%%)', val_acc)

            if improved_loss:
                self.best_val_loss = val_loss
                self._epochs_since_improve = 0
            else:
                self._epochs_since_improve += 1

            if on_epoch_end is not None:
                on_epoch_end({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': new_lr,
                    'elapsed_ms': elapsed_ms,
                })

            if self._epochs_since_improve >= patience:
                logger.info('Early stopping at epoch %d (no val_loss improvement for %d epochs)',
                            epoch + 1, patience)
                break

        logger.info('Training complete (best val_acc=%.2f%% @ epoch %d).',
                    self.best_val_acc, self.best_epoch)

    def load_best_into_model(self) -> None:
        """학습 후 best state_dict를 다시 self.model에 로드"""
        if self.best_state_dict is None:
            raise RuntimeError('No best state_dict available — train() never produced an improvement')
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
