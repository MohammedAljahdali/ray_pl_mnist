from typing import Optional
from warnings import warn
import platform
from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningDataModule

from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = '',
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            seed: starting seed for RNG.
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)
        if num_workers and platform.system() == "Windows":
            # see: https://stackoverflow.com/a/59680818
            warn(
                f"You have requested num_workers={num_workers} on Windows,"
                " but currently recommended is 0, so we set it for you"
            )
            num_workers = 0

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.test_transforms = self.default_transforms

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Saves MNIST files to `data_dir`"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""
        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        train_length = len(dataset)
        self.dataset_train, self.dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])

    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split"""
        extra = dict(transform=self.test_transforms) if self.test_transforms else {}
        dataset = MNIST(self.data_dir, train=False, download=False, **extra)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @property
    def default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5, ), std=(0.5, ))
            ])
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms