# ============================================================================
# 파일: visualizer.py
# 설명: 결과 시각화
# ============================================================================

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class Visualizer:
    """결과 시각화 클래스"""

    def __init__(self, config) -> None:
        self.config = config
        os.makedirs(config.RESULTS_SAVE_PATH, exist_ok=True)

    def plot_training_history(self, history: dict) -> None:
        """학습 히스토리 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.config.RESULTS_SAVE_PATH, 'training_history.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info('Training history saved to %s', filepath)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """혼동 행렬 시각화"""
        cm = confusion_matrix(y_true, y_pred)

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES,
            cbar_kws={'label': 'Count'},
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        filepath = os.path.join(self.config.RESULTS_SAVE_PATH, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info('Confusion matrix saved to %s', filepath)
