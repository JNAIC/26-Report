"""训练 MNIST CNN：验证集选模型，训练完成后评估独立测试集。"""
import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn

from cnn_model import DigitCNN
from prepare_cnn_data import load_data


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一轮：每一批都计算梯度并更新参数。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('损失出现无效值，请检查训练设置。')
        loss.backward()
        optimizer.step()

        # 损失按图片数加权，避免最后一个小批次影响平均值的计算。
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        if batch_index % 100 == 0 or batch_index == len(loader):
            print(f'  Batch {batch_index}/{len(loader)} | '
                  f'loss={total_loss / total:.4f} | acc={correct / total:.2%}', flush=True)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_examples=False):
    """验证或测试：只预测，不计算梯度，也不更新参数。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    examples = {'correct': [], 'wrong': []}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('评估损失出现无效值。')
        predictions = logits.argmax(dim=1)
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        if collect_examples:
            # 按测试集顺序收集前 10 个正确和错误案例，便于复查。
            for image, truth, prediction in zip(images.cpu(), labels.cpu(), predictions.cpu()):
                key = 'correct' if truth.item() == prediction.item() else 'wrong'
                if len(examples[key]) < 10:
                    examples[key].append((image, truth.item(), prediction.item()))
    return total_loss / total, correct / total, correct, total, examples


def save_plots(history, examples, output):
    epochs = [row['epoch'] for row in history]
    for metric, title, filename in (
        ('loss', 'Cross-entropy loss', 'loss_curve.png'),
        ('accuracy', 'Accuracy', 'accuracy_curve.png'),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [row[f'train_{metric}'] for row in history], 'o-', label='Train')
        ax.plot(epochs, [row[f'val_{metric}'] for row in history], 'o-', label='Validation')
        ax.set(xlabel='Epoch', ylabel=title, title=title, xticks=epochs)
        if metric == 'accuracy':
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)

    for key, title in (('correct', 'Correct test predictions'), ('wrong', 'Incorrect test predictions')):
        items = examples[key]
        rows = max(1, math.ceil(len(items) / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(10, 2.5 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (image, truth, prediction) in zip(axes.flat, items):
            ax.imshow(image[0], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'True: {truth} | Pred: {prediction}')
        if not items:
            axes[0, 2].text(0.5, 0.5, 'No examples', ha='center')
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output / f'{key}_examples.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs、batch-size 必须为正整数，lr 必须为有限正数。')

    torch.manual_seed(42)
    # 小型 CNN 在 CPU 上限制线程数量，避免线程过多带来额外开销。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).resolve().parent
    output = root / 'results_cnn' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)  # 每次运行独立保存，不覆盖旧结果。
    config = {**vars(args), 'seed': 42, 'device': str(device),
              'torch_version': torch.__version__, 'pixel_range': [0, 1],
              'split_indices': str(root / 'data_check' / 'split_indices.json')}
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train_loader, val_loader, test_loader = load_data(args.batch_size)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_accuracy, best_epoch = -1.0, 0
    checkpoint = output / 'best_model.pt'
    print(f'Device: {device}\nResults: {output}', flush=True)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}', flush=True)
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _, _ = evaluate(model, val_loader, criterion, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                        'val_loss': val_loss, 'val_accuracy': val_accuracy})
        # 每轮写入记录；中途停止时也保留已经完成的轮次。
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f'Train loss={train_loss:.4f}, acc={train_accuracy:.2%} | '
              f'Val loss={val_loss:.4f}, acc={val_accuracy:.2%}', flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(model.state_dict(), checkpoint)
            print('Saved best model.', flush=True)

    # 只根据验证集选模型，全部训练结束后才做最终测试。
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, test_accuracy, correct, total, examples = evaluate(
        model, test_loader, criterion, device, collect_examples=True)
    summary = {'best_epoch': best_epoch, 'best_val_accuracy': best_accuracy,
               'test_loss': test_loss, 'test_accuracy': test_accuracy,
               'test_correct': correct, 'test_wrong': total - correct, 'test_total': total,
               'elapsed_seconds': time.perf_counter() - start}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save_plots(history, examples, output)
    print(f'\nBest epoch: {best_epoch} | Test accuracy: {test_accuracy:.2%} '
          f'({correct}/{total})\nAll results saved to: {output}', flush=True)


if __name__ == '__main__':
    main()
