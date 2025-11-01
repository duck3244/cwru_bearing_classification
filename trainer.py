# ============================================================================
# 파일: trainer.py
# 설명: 모델 학습 및 평가
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os


class Trainer:
    """모델 학습 관리자"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)

        # 손실 함수 및 옵티마이저
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True
        )

        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0.0

    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self, test_loader):
        """모델 평가"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(test_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs):
        """전체 학습 프로세스"""
        print(f'Training on {self.device}')
        print('=' * 60)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            self.scheduler.step(val_loss)

            print(f'Epoch [{epoch + 1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # 최고 성능 모델 저장
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f'  → Best model saved (Val Acc: {val_acc:.2f}%)')

        print('=' * 60)
        print('Training complete!')

    def save_model(self, filename):
        """모델 저장"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filename):
        """모델 로드"""
        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        self.model.load_state_dict(torch.load(filepath))