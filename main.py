# ============================================================================
# 파일: main.py
# 설명: 메인 실행 파일
# ============================================================================

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


def main():
    # 설정 로드
    from config import Config
    config = Config()

    print('=' * 60)
    print('CWRU Bearing Fault Classification with PyTorch')
    print('=' * 60)

    # 데이터 로드
    print('\n[1/6] Loading data...')
    from data_loader import CWRUDataLoader
    data_loader = CWRUDataLoader(config)
    X, y = data_loader.load_data()
    print(f'  Data shape: {X.shape}, Labels shape: {y.shape}')

    # 데이터 분할
    print('\n[2/6] Splitting data...')
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')

    # 데이터 정규화
    print('\n[3/6] Normalizing data...')
    X_train, X_test = data_loader.normalize_data(X_train, X_test)

    # 데이터셋 및 데이터로더 생성
    print('\n[4/6] Creating datasets and dataloaders...')
    from dataset import BearingDataset
    train_dataset = BearingDataset(X_train, y_train)
    test_dataset = BearingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 모델 생성
    print('\n[5/6] Creating model...')
    from model import BearingClassifier
    input_size = X_train.shape[1]
    model = BearingClassifier(
        input_size=input_size,
        hidden_sizes=config.HIDDEN_SIZES,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    print(f'  Input size: {input_size}')
    print(f'  Hidden sizes: {config.HIDDEN_SIZES}')
    print(f'  Number of classes: {config.NUM_CLASSES}')
    print(f'  Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 학습
    print('\n[6/6] Training model...')
    from trainer import Trainer
    trainer = Trainer(model, config)
    trainer.train(train_loader, test_loader, config.EPOCHS)

    # 최고 성능 모델 로드 및 평가
    print('\n' + '=' * 60)
    print('Final Evaluation')
    print('=' * 60)
    trainer.load_model('best_model.pth')
    _, test_acc, y_pred, y_true = trainer.evaluate(test_loader)

    print(f'\nTest Accuracy: {test_acc:.2f}%')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    # 시각화
    print('\nGenerating visualizations...')
    from visualizer import Visualizer
    visualizer = Visualizer(config)
    visualizer.plot_training_history(trainer.history)
    visualizer.plot_confusion_matrix(y_true, y_pred)

    print('\n' + '=' * 60)
    print('All done!')
    print('=' * 60)


if __name__ == '__main__':
    main()