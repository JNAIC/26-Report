"""Prepare and verify MNIST for the CNN project; no model training."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'data_check'


def load_data(batch_size=64):
    """Return reproducible train/validation/test loaders (pixels in [0, 1])."""
    transform = transforms.ToTensor()
    full = datasets.MNIST(str(ROOT / 'data'), train=True,
                          download=True, transform=transform)
    test = datasets.MNIST(str(ROOT / 'data'), train=False,
                          download=True, transform=transform)
    RESULTS.mkdir(exist_ok=True)
    split_path = RESULTS / 'split_indices.json'
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding='utf-8'))
    else:
        indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(42)).tolist()
        split = {'seed': 42, 'train': indices[:50000], 'validation': indices[50000:]}
        split_path.write_text(json.dumps(split), encoding='utf-8')
    train_ids, val_ids = split['train'], split['validation']
    assert len(train_ids) == 50000 and len(val_ids) == 10000
    assert len(set(train_ids)) == 50000 and len(set(val_ids)) == 10000
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(len(full)))
    train, val = Subset(full, train_ids), Subset(full, val_ids)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(42), num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def main():
    loaders = load_data()
    summary = {
        'python': sys.version.split()[0], 'python_executable': sys.executable,
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'matplotlib': matplotlib.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'split_seed': 42, 'batch_size': 64, 'datasets': {},
    }
    for name, loader in zip(('train', 'validation', 'test'), loaders):
        images, labels = next(iter(loader))
        assert images.shape == (64, 1, 28, 28)
        assert images.dtype == torch.float32 and labels.dtype == torch.int64
        assert torch.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
        assert 0 <= labels.min() <= labels.max() <= 9
        counts = torch.zeros(10, dtype=torch.int64)
        if isinstance(loader.dataset, Subset):
            targets = loader.dataset.dataset.targets[loader.dataset.indices]
        else:
            targets = loader.dataset.targets
        counts += torch.bincount(targets, minlength=10)
        assert (counts > 0).all()
        summary['datasets'][name] = {
            'size': len(loader.dataset), 'batch_shape': list(images.shape),
            'image_dtype': str(images.dtype), 'label_dtype': str(labels.dtype),
            'class_counts_0_to_9': counts.tolist(),
        }
    fig, axes = plt.subplots(4, 5, figsize=(9, 7))
    train = loaders[0].dataset
    for i, ax in enumerate(axes.flat):
        picture, label = train[i]
        ax.imshow(picture[0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'True label: {label}')
        ax.axis('off')
    fig.suptitle('MNIST training samples (ground-truth labels)')
    fig.tight_layout()
    fig.savefig(RESULTS / 'cnn_mnist_samples.png', dpi=150)
    plt.close(fig)
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('PASS: split integrity, input shapes, types, value range and labels.')


if __name__ == '__main__':
    main()
