# ============================================================================
# 파일: main.py
# 설명: CLI 진입점 — 학습을 실행하고 아티팩트를 디렉토리로 저장
# ============================================================================

import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from artifact import ModelMeta, compute_arch_hash, load_artifact, save_artifact
from config import Config
from data_loader import CWRUDataLoader
from dataset import BearingDataset
from inference import predict_signal  # noqa: F401  (스모크 임포트, 모듈 등록)
from model import BearingClassifier
from trainer import Trainer
from utils import set_seed, setup_logging
from visualizer import Visualizer

logger = logging.getLogger(__name__)


def main() -> None:
    config = Config()
    setup_logging(config.LOG_LEVEL)
    set_seed(config.RANDOM_STATE)

    logger.info('=' * 60)
    logger.info('CWRU Bearing Fault Classification with PyTorch')
    logger.info('=' * 60)

    logger.info('[1/6] Loading data...')
    data_loader = CWRUDataLoader(config)
    X, y = data_loader.load_data()
    logger.info('  Data shape: %s, Labels shape: %s', X.shape, y.shape)

    logger.info('[2/6] Splitting data (train/val/test)...')
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    logger.info('  Train: %s, Val: %s, Test: %s', X_train.shape, X_val.shape, X_test.shape)

    logger.info('[3/6] Normalizing data...')
    X_train, X_val, X_test = data_loader.normalize_data(X_train, X_val, X_test)

    logger.info('[4/6] Creating datasets and dataloaders...')
    train_dataset = BearingDataset(X_train, y_train)
    val_dataset = BearingDataset(X_val, y_val)
    test_dataset = BearingDataset(X_test, y_test)

    loader_generator = torch.Generator().manual_seed(config.RANDOM_STATE)
    common_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=loader_generator, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    logger.info('[5/6] Creating model...')
    input_size = X_train.shape[1]
    model = BearingClassifier(
        input_size=input_size,
        hidden_sizes=config.HIDDEN_SIZES,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    )
    logger.info('  Input size: %d', input_size)
    logger.info('  Hidden sizes: %s', config.HIDDEN_SIZES)
    logger.info('  Number of classes: %d', config.NUM_CLASSES)
    logger.info('  Model parameters: %s', f'{sum(p.numel() for p in model.parameters()):,}')

    logger.info('[6/6] Training model...')
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, config.EPOCHS)

    if trainer.best_state_dict is None:
        raise RuntimeError('Training produced no best model — aborting artifact save')

    artifact_id = uuid4().hex
    artifact_dir = os.path.join(config.MODEL_SAVE_PATH, artifact_id)
    meta = ModelMeta(
        artifact_id=artifact_id,
        arch_hash=compute_arch_hash(input_size, list(config.HIDDEN_SIZES),
                                    config.NUM_CLASSES, config.DROPOUT),
        input_size=input_size,
        hidden_sizes=list(config.HIDDEN_SIZES),
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        sampling_rate=config.SAMPLING_RATE,
        window_size=config.WINDOW_SIZE,
        overlap=config.OVERLAP,
        class_names=list(config.CLASS_NAMES),
        label_map=dict(config.LABEL_MAP),
        val_acc=trainer.best_val_acc,
        val_loss=trainer.best_val_loss,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_artifact(artifact_dir, trainer.best_state_dict, data_loader.scaler, meta)
    logger.info('Artifact saved → %s', artifact_dir)

    logger.info('=' * 60)
    logger.info('Final Evaluation (held-out test set)')
    logger.info('=' * 60)
    loaded = load_artifact(artifact_dir, device=config.DEVICE)
    eval_trainer = Trainer(loaded.model, config)
    _, test_acc, y_pred, y_true = eval_trainer.evaluate(test_loader)

    logger.info('Test Accuracy: %.2f%%', test_acc)
    logger.info('Classification Report:\n%s',
                classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    logger.info('Generating visualizations...')
    visualizer = Visualizer(config)
    visualizer.plot_training_history(trainer.history)
    visualizer.plot_confusion_matrix(y_true, y_pred)

    logger.info('=' * 60)
    logger.info('All done!')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
