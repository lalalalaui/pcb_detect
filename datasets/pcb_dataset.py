from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if path.exists():
        return path

    project_root = Path(__file__).resolve().parents[1]
    project_path = project_root / data_dir
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"Data directory not found: {data_dir}")


def _make_classification_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def get_classification_dataloaders(
    data_dir: str = "data/processed/pcb_cls",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Build DataLoaders for the supervised PCB defect classification dataset."""
    root = _resolve_data_dir(data_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing classification split directory: {split_dir}")

    train_transform, eval_transform = _make_classification_transforms(image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names


def _make_anomaly_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class PCBAnomalyDataset(Dataset):
    """Patch dataset for PCB anomaly detection.

    Each item returns image, label, path. label=0 means normal and label=1 means anomaly.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = _resolve_data_dir(data_dir)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No anomaly samples found for split '{split}' in {self.root}")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing anomaly split directory: {split_dir}")

        normal_samples = [(path, 0) for path in _list_images(split_dir / "normal")]
        if self.split == "train":
            return normal_samples

        anomaly_samples = [(path, 1) for path in _list_images(split_dir / "anomaly")]
        return normal_samples + anomaly_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as exc:
            raise OSError(f"Failed to read image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(image_path)


def get_anomaly_dataloaders(
    data_dir: str = "data/processed/pcb_anomaly",
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Build DataLoaders for the PCB anomaly-detection patch dataset."""
    transform = _make_anomaly_transform(image_size)

    train_dataset = PCBAnomalyDataset(data_dir=data_dir, split="train", transform=transform)
    val_dataset = PCBAnomalyDataset(data_dir=data_dir, split="val", transform=transform)
    test_dataset = PCBAnomalyDataset(data_dir=data_dir, split="test", transform=transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def _print_batch_info(name: str, loader: DataLoader) -> None:
    images, labels, paths = next(iter(loader))
    label_counts = {}
    if hasattr(loader.dataset, "samples"):
        for _, label in loader.dataset.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
    print(f"{name} batch image shape: {tuple(images.shape)}")
    print(f"{name} label example: {labels[:8].tolist()}")
    print(f"{name} label counts: {label_counts}")
    print(f"{name} path example: {paths[0]}")


def main() -> None:
    print("=" * 80)
    print("PCB Dataset Loader Smoke Test")
    print("=" * 80)

    cls_train_loader, cls_val_loader, cls_test_loader, class_names = get_classification_dataloaders()
    cls_images, cls_labels = next(iter(cls_train_loader))
    print(f"Classification class names: {class_names}")
    print(f"Classification train batch image shape: {tuple(cls_images.shape)}")
    print(f"Classification train label example: {cls_labels[:8].tolist()}")
    print(f"Classification val batches: {len(cls_val_loader)}")
    print(f"Classification test batches: {len(cls_test_loader)}")

    anomaly_train_loader, anomaly_val_loader, anomaly_test_loader = get_anomaly_dataloaders()
    _print_batch_info("Anomaly train", anomaly_train_loader)
    _print_batch_info("Anomaly val", anomaly_val_loader)
    _print_batch_info("Anomaly test", anomaly_test_loader)

    print("\nDataset loader smoke test finished.")


if __name__ == "__main__":
    main()
