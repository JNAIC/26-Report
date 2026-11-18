"""两层卷积的 MNIST 分类网络。直接运行本文件可检查网络。"""
import json
from pathlib import Path

import torch
from torch import nn


class DigitCNN(nn.Module):
    """输入 [批大小, 1, 28, 28]，输出 [批大小, 10] 的原始分数。"""

    def __init__(self):
        super().__init__()
        # padding=1 让 3×3 卷积保持图片的高和宽。
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28×28 → 14×14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # 保留批次维度：32×7×7 = 1568
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        # CrossEntropyLoss 直接接收原始分数，输出层无需加 Softmax。

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def check_model():
    """用真实训练数据验证尺寸、梯度及一次更新，不进行完整训练。"""
    from prepare_cnn_data import load_data

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitCNN().to(device)
    train_loader, _, _ = load_data()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    print(model)
    print('Device:', device)
    shapes = []
    x = images
    # 逐层显示实际尺寸，不额外改变模型结构。
    with torch.no_grad():
        print('Input:', list(x.shape))
        for layer in [*model.features, *model.classifier]:
            x = layer(x)
            shapes.append({'layer': type(layer).__name__, 'shape': list(x.shape)})
            print(type(layer).__name__, '->', list(x.shape))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    logits = model(images)
    assert logits.shape == (len(images), 10)
    assert torch.isfinite(logits).all()
    loss = nn.CrossEntropyLoss()(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f'Missing gradient: {name}'
        assert torch.isfinite(parameter.grad).all(), f'Invalid gradient: {name}'
    before = model.features[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.features[0].weight.detach())
    # 同时验证单张图片输入，避免网络只能处理固定批大小。
    with torch.no_grad():
        assert model(images[:1]).shape == (1, 10)

    summary = {
        'device': str(device),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'input_shape': list(images.shape),
        'layer_shapes': shapes,
        'output_shape': list(logits.shape),
        'initial_batch_loss': loss.item(),
        'finite_gradients': True,
        'parameter_update_verified': True,
        'single_image_verified': True,
        'note': 'One batch update for verification only; no trained model saved.',
    }
    output = Path(__file__).resolve().parent / 'data_check' / 'cnn_model_check.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Parameters:', summary['parameter_count'])
    print('Initial batch loss:', summary['initial_batch_loss'])
    print('PASS: batch/single-image outputs, finite gradients and parameter update.')
    print('This is a model check, not a completed training run.')


if __name__ == '__main__':
    check_model()
