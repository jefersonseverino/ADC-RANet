import json

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms as T

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


class ImageNet10Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./image-net/classes/",
        batch_size: int = 32,
        img_size: int = 64,
        aug: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.train_transform = None
        if aug:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomResizedCrop(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare_data(self):
        datasets.ImageFolder(root=self.data_dir, transform=None)

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=None)

        print(f"Classes: {full_dataset.classes}")
        print(f"Total images: {len(full_dataset)}")
        print(f"Samples per class: {torch.bincount(torch.tensor(full_dataset.targets))}")

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

        self.train_dataset = TransformedSubset(
            Subset(full_dataset, train_idx), transform=self.train_transform
        )
        self.val_dataset = TransformedSubset(
            Subset(full_dataset, val_idx), transform=self.val_transform
        )
        self.test_dataset = TransformedSubset(
            Subset(full_dataset, test_idx), transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)


import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class DistechDataset(Dataset):
    def __init__(self, annotations, split, base_path, img_size, resize=False, transform=None):
        self.samples = []
        self.base_path = base_path
        self.split = split
        self.img_size = img_size
        self.resize = resize
        self.transform = transform

        count_dict = self.get_count_per_class(annotations)

        for n_people, seq_ids in count_dict.items():
            for seq_id in seq_ids:
                self.samples.append((seq_id, n_people))

    def get_count_per_class(self, annotations):
        count_per_class = {}

        for image_id, data in annotations.items():
            n_people = len(data["boxes"])
            if 0 < n_people <= 3:
                if n_people not in count_per_class:
                    count_per_class[n_people] = []
                count_per_class[n_people].append(image_id)

        return count_per_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, label = self.samples[idx]
        path = os.path.join(self.base_path, self.split, f"{seq_id}.npy")

        im = np.load(path)

        # scale to consistent range (preserve existing scaling behavior)
        im = (im.max() - im) / 760.0

        # ensure shape is C,H,W
        if im.ndim == 2:
            im = np.stack((im,) * 3, axis=0)
        elif im.ndim == 3:
            if im.shape[-1] == 3:
                im = np.transpose(im, (2, 0, 1))

        im = torch.from_numpy(im).float()

        if self.resize:
            height = int(self.img_size * 0.75)
            im = T.Resize((self.img_size, height))(im)

        if self.transform is not None:
            im = self.transform(im)

        return im, torch.tensor(label, dtype=torch.long) - 1


class DistechDataDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/export/livia/home/vision/Jaraujo/projects/ETS/distech/distech/v18/v18",
        batch_size=32,
        num_workers=4,
        img_size=224,
        resize=False,
        transform=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.resize = resize
        self.transform = transform

    def load_annotations(self, path):
        with open(path, "r") as f:
            annotations = json.load(f)

        categories = annotations["categories"]
        annot = {}

        for image in annotations["images"]:
            image_id = image["id"]
            annot[image_id] = {
                "boxes": [],
                "labels": [],
            }

        for a in annotations["annotations"]:
            image_id = a["image_id"]
            annot[image_id]["boxes"].append(a["bbox"])
            annot[image_id]["labels"].append(categories[a["category_id"]]["id"])

        return annot

    def setup(self, stage=None):
        train_annotations = self.load_annotations(
            os.path.join(self.data_dir, "train", "annotations.json")
        )
        val_annotations = self.load_annotations(
            os.path.join(self.data_dir, "va" \
            "l", "annotations.json")
        )
        test_annotations = self.load_annotations(
            os.path.join(self.data_dir, "test", "annotations.json")
        )

        train_transform = None
        if self.transform is not None:
            train_transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(degrees=15),
                ]
            )

        self.train_dataset = DistechDataset(
            train_annotations,
            "train",
            self.data_dir,
            self.img_size,
            resize=self.resize,
            transform=train_transform,
        )

        self.val_dataset = DistechDataset(
            val_annotations, "val", self.data_dir, self.img_size, resize=self.resize
        )
        self.test_dataset = DistechDataset(
            test_annotations, "test", self.data_dir, self.img_size, resize=self.resize
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

class CIFAR100Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        img_size: int = 32,
        use_valid: bool = True,
        val_size: int = 5000,
        resize: bool = False,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_valid = use_valid
        self.val_size = val_size
        self.resize = resize

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )

        train_ops = []
        eval_ops = []

        if self.resize and self.img_size != 32:
            train_ops.append(transforms.Resize((self.img_size, self.img_size)))
            eval_ops.append(transforms.Resize((self.img_size, self.img_size)))

        train_ops.extend([
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        if transform is not None:
            train_ops.extend([
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
            ])

        train_ops.extend([transforms.ToTensor(), normalize])
        eval_ops.extend([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose(train_ops)
        self.eval_transform = transforms.Compose(eval_ops)

    def prepare_data(self):
        datasets.CIFAR100(root=self.data_dir, train=True, download=False)
        datasets.CIFAR100(root=self.data_dir, train=False, download=False)

    def setup(self, stage=None):
        full_train_aug = datasets.CIFAR100(
            root=self.data_dir, train=True, transform=self.train_transform
        )
        full_train_eval = datasets.CIFAR100(
            root=self.data_dir, train=True, transform=self.eval_transform
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir, train=False, transform=self.eval_transform
        )

        if self.use_valid:
            total = len(full_train_aug)
            val_len = self.val_size
            train_len = total - val_len

            gen = torch.Generator().manual_seed(SEED)
            train_idx, val_idx = torch.utils.data.random_split(
                range(total), [train_len, val_len], generator=gen
            )

            self.train_dataset = Subset(full_train_aug, train_idx.indices)
            self.val_dataset = Subset(full_train_eval, val_idx.indices)
        else:
            self.train_dataset = full_train_aug
            self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )