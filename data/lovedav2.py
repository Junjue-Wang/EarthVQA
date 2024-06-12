import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler
from ever.api.data import distributed
import numpy as np
import logging

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
    Playground=(165,0,165),
    Pond=(0,185,246),
)






class LoveDADataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms


    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        raw_image = image.copy()
        mask=None
        if len(self.cls_filepath_list) > 0:
            mask = imread(self.cls_filepath_list[idx]).astype(np.int64) -1
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
            return image, dict(mask=mask, imagen=os.path.basename(self.rgb_filepath_list[idx]), raw_image=raw_image)
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']
            return image, dict(imagen=os.path.basename(self.rgb_filepath_list[idx]), raw_image=raw_image)


    def __len__(self):
        return len(self.rgb_filepath_list)


@er.registry.DATALOADER.register()
class LoveDALoaderV2(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = LoveDADataset(self.config.image_dir, self.config.mask_dir, self.config.transforms)

        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
            dataset)

        super(LoveDALoaderV2, self).__init__(dataset,
                                           self.config.batch_size,
                                           sampler=sampler,
                                           num_workers=self.config.num_workers,
                                           pin_memory=True)
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))
