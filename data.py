from torch.utils.data import Dataset, dataloader
from torchvision import transforms as trans
import torchvision
import os
import torch
import numpy as np
from PIL import Image
from ast import literal_eval

class CustomDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mask_transform=None):

        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])

        path = data_path.split('/')[:-1]
        mask_dir = '/'.join(path)
        name = data_path.split('/')[-1]
        # print(data_path)
        if '_l' in name:
            mask_name = 'mask_l.jpg'
        elif '_r' in name:
            mask_name = 'mask_r.jpg'
        else:
            mask_name = 'mask.jpg'
        image = Image.open(data_path).convert('RGB')
        mask  = Image.open(mask_dir + '/' + mask_name).convert('L')
        cat   = np.array(literal_eval(self.df.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, cat


def get_train_dataset(img_folder, df_train, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = trans.Compose([
        trans.Resize(img_size, Image.BILINEAR),
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std)
    ])
    mask_transform = trans.Compose([
        trans.Resize(img_size, Image.NEAREST),
        trans.ToTensor()
    ])

    ds_train = CustomDataset(df_train, img_folder, transform=train_transform, mask_transform=mask_transform)


    return ds_train


def get_val_test_dataset(img_folder, df, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_transform = trans.Compose([
        trans.Resize(img_size, Image.BILINEAR),
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std)
    ])
    mask_transform = trans.Compose([
        trans.Resize(img_size, Image.NEAREST),
        trans.ToTensor()
    ])

    ds = CustomDataset(df, img_folder, transform=img_transform, mask_transform=mask_transform)
    return ds


def create_data_loaders(dataset_train, dataset_valid, batch_size, num_workers):
    """
        Function to build the data loaders.
    """

    train_loader = dataloader.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    valid_loader = dataloader.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    return train_loader, valid_loader


def create_test_loader(dataset_test, batch_size, num_workers):
    """
        Function to build the data loaders.
    """
    test_loader = dataloader.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    return test_loader