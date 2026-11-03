import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from torch.utils.data import Subset, Dataset

import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

SEED = 123


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        X, y = self.subset[index]
        if self.transform:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.subset)


class MultiScaleDataset(Dataset):
    def __init__(self, base_dataset, input_sizes):
        self.base_dataset = base_dataset
        self.input_sizes = input_sizes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        images = []
        for size in self.input_sizes:
            if img.shape[1] == size and img.shape[2] == size:
                images.append(img)
            else:
                resized = F.interpolate(
                    img.unsqueeze(0),
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                images.append(resized)

        return images, label


def multi_scale_collate_fn(batch):
    num_scales = len(batch[0][0])

    images_per_scale = [[] for _ in range(num_scales)]
    labels = []

    for images, label in batch:
        for i, img in enumerate(images):
            images_per_scale[i].append(img)
        labels.append(label)

    batched_images = [torch.stack(imgs) for imgs in images_per_scale]
    batched_labels = torch.tensor(labels)

    return batched_images, batched_labels


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None

    max_size = max(args.input_sizes) if hasattr(args, "input_sizes") else args.img_size
    train_transform = transforms.Compose(
        [
            # transforms.Resize((max_size, max_size)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    full_dataset = datasets.ImageFolder(root=args.data_root, transform=None)

    targets = np.array(full_dataset.targets)
    indices = np.arange(len(full_dataset))

    # Division 80% train, 10% val, 10% test with stratification
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=SEED
    )

    temp_targets = targets[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_targets, random_state=SEED
    )

    train_subset = TransformedSubset(
        Subset(full_dataset, train_idx), transform=train_transform
    )
    val_subset = TransformedSubset(
        Subset(full_dataset, val_idx), transform=val_transform
    )
    test_subset = TransformedSubset(
        Subset(full_dataset, test_idx), transform=test_transform
    )

    # Se input_sizes está definido, usa MultiScaleDataset
    if hasattr(args, "input_sizes") and len(args.input_sizes) > 1:
        train_set = MultiScaleDataset(train_subset, args.input_sizes)
        val_set = MultiScaleDataset(val_subset, args.input_sizes)
        test_set = MultiScaleDataset(test_subset, args.input_sizes)
        collate_fn = multi_scale_collate_fn
    else:
        train_set = train_subset
        val_set = val_subset
        test_set = test_subset
        collate_fn = None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def _seed_worker(worker_id):
    import random

    """Seed each DataLoader worker deterministically based on the base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_imagenet100(args):
    import json
    from pathlib import Path

    labels_path = Path(args.data_root) / "Labels.json"
    labels = json.loads(labels_path.read_text())
    labels = {k: v.split(",")[0] for k, v in labels.items()}

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transforms = [
        transforms.RandomResizedCrop(224),
    ]

    if args.use_color_jitter:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            )
        )

    if args.use_random_augmentation:
        train_transforms.append(
            transforms.RandAugment(num_ops=2, magnitude=9)
        )

    train_transforms.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if args.use_random_erasing:
        train_transforms.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
            )
        )

    train_transforms = transforms.Compose(train_transforms)

    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    data_root = Path(args.data_root)

    import pickle

    cache_file = data_root / "imagenet100_index_cache.pkl"
    force_rebuild = getattr(args, "rebuild_dataset_cache", False)

    def scan_split(split_dirs):
        samples = []
        for split_dir in split_dirs:
            print(split_dir)
            if not split_dir.exists():
                continue
            for folder in split_dir.iterdir():
                if not folder.is_dir() or folder.name not in labels:
                    continue
                cls_name = folder.name
                for img_path in folder.iterdir():
                    if img_path.suffix.lower().lstrip(".") in ("jpg", "jpeg", "png"):
                        samples.append((str(img_path), cls_name))
        return samples

    if cache_file.exists() and not force_rebuild:
        print(f"=> Loading cached dataset index from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        class_to_idx = cache["class_to_idx"]
        train_samples = cache["train_samples"]  # list of (path, class_idx)
        test_samples = cache["test_samples"]
        train_idx = cache["train_idx"]
        val_idx = cache["val_idx"]
    else:
        print("=> Building dataset index (first run, this may take a while)...")
        train_dirs = sorted(data_root.glob("train.X*"))
        val_dirs   = sorted(data_root.glob("val.X*"))

        train_raw = scan_split(train_dirs)
        test_raw = scan_split(val_dirs)

        classes = sorted({cls for _, cls in train_raw} | {cls for _, cls in test_raw})
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        train_samples = [(p, class_to_idx[c]) for p, c in train_raw]
        test_samples = [(p, class_to_idx[c]) for p, c in test_raw]

        stratify_targets = np.array([t for _, t in train_samples])
        indices = list(range(len(train_samples)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.12, stratify=stratify_targets, random_state=SEED
        )

        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "class_to_idx": class_to_idx,
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                },
                f,
            )
        print(f"=> Saved dataset index cache to {cache_file}")

    loader_fn = datasets.folder.default_loader

    class CachedImageDataset(Dataset):
        def __init__(self, samples, loader, transform=None):
            self.samples = samples
            self.loader = loader
            self.transform = transform
            self.targets = [t for _, t in samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, target = self.samples[idx]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

    train_full = CachedImageDataset(train_samples, loader_fn, transform=train_transforms)
    test_dataset = CachedImageDataset(test_samples, loader_fn, transform=eval_transform)

    train_dataset = Subset(train_full, train_idx)
    val_dataset   = Subset(train_full, val_idx)

    g = torch.Generator()
    g.manual_seed(getattr(args, "seed", SEED))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader

def get_dataloader_distech(args, resize=False, transform=None):
    from torch.utils.data import DataLoader
    from data.datamodule import DistechDataDatamodule

    datamodule = DistechDataDatamodule(
        batch_size=args.batch_size,
        resize=resize,
        transform=transform,
        img_size=args.img_size,
    )

    datamodule.setup()

    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    g = torch.Generator()
    g.manual_seed(getattr(args, "seed", SEED))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=_seed_worker,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader, test_loader


def get_dataloader_cifar10(args):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    full_train_aug = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )
    full_train_eval = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=eval_transform
    )

    total = len(full_train_aug)
    val_len = 5000 if getattr(args, "use_valid", False) else 0
    train_len = total - val_len

    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_idx, val_idx = torch.utils.data.random_split(
        range(total), [train_len, val_len], generator=gen
    )

    train_dataset = torch.utils.data.Subset(full_train_aug, train_idx.indices)
    val_dataset = torch.utils.data.Subset(full_train_eval, val_idx.indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    if getattr(args, "use_valid", False):
        return train_loader, val_loader, test_loader

    return train_loader, test_loader, test_loader


def get_dataloader_cifar100(args):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    )

    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # eval_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    
    train_transforms = [
        transforms.RandomCrop(32, padding=4)
    ]
    
    if args.use_color_jitter:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            )
        )

    if args.use_random_augmentation:
        train_transforms.append(
            transforms.RandAugment(num_ops=2, magnitude=9)
        )
    
    train_transforms.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    if args.use_random_erasing:
        train_transforms.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
            )
        )

    train_transforms = transforms.Compose(train_transforms)
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    full_train_aug = datasets.CIFAR100(
        root=args.data_root, train=True, download=True, transform=train_transforms
    )
    full_train_eval = datasets.CIFAR100(
        root=args.data_root, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=eval_transform
    )

    total = len(full_train_aug)
    val_len = 5000 if args.use_valid else 0
    train_len = total - val_len

    gen = torch.Generator()
    gen.manual_seed(123)
    train_idx, val_idx = torch.utils.data.random_split(
        range(total), [train_len, val_len], generator=gen
    )

    train_dataset = torch.utils.data.Subset(full_train_aug, train_idx.indices)
    val_dataset = torch.utils.data.Subset(full_train_eval, val_idx.indices)

    #train_sampler = DistributedSampler(train_dataset, shuffle=True)
    use_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        #shuffle=True,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    if args.use_valid:
        return train_loader, val_loader, test_loader

    return train_loader, test_loader, test_loader 

def get_dataloader_tinyimagenet(args):
    root = args.data_root    
    if not os.path.isdir(os.path.join(root, "train")):
        candidate = os.path.join(root, "tiny-imagenet-200")
        if os.path.isdir(candidate):
            root = candidate
        else:
            raise FileNotFoundError(
                f"Cannot find tiny-imagenet-200 in {args.data_root}. "
                f"Expected {root}/train/ or {candidate}/train/ to exist."
            )

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    _restructure_train(train_dir)
    _restructure_val(val_dir)

    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2770, 0.2691, 0.2821],
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    if getattr(args, "use_valid", False):
        total = len(full_train)
        val_len = 10000  # 50 per class
        train_len = total - val_len

        gen = torch.Generator()
        gen.manual_seed(SEED)
        train_idx, val_split_idx = torch.utils.data.random_split(
            range(total), [train_len, val_len], generator=gen
        )

        train_dataset = torch.utils.data.Subset(full_train, train_idx.indices)
        val_split_dataset = torch.utils.data.Subset(full_train_eval, val_split_idx.indices)
        test_dataset = val_dataset
    else:
        train_dataset = full_train
        val_split_dataset = val_dataset
        test_dataset = val_dataset

    g = torch.Generator()
    g.manual_seed(getattr(args, "seed", SEED))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        generator=g,
    )

    val_loader = DataLoader(
        val_split_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader