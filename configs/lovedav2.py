from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, RandomBrightnessContrast, Resize, Transpose
import ever as er
from ever.api.preprocess.albu import RandomDiscreteScale


data = dict(
    train=dict(
        type='LoveDALoaderV2',
        params=dict(
            image_dir='./EarthVQA/Train/images_png',
            mask_dir='./EarthVQA/Train/masks_png',
            transforms=Compose([
                RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
                RandomCrop(512, 512),
                RandomBrightnessContrast(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()
            ]),
            CV=dict(k=10, i=-1),
            training=True,
            batch_size=16,
            num_workers=4,
        ),
    ),
    test=dict(
        type='LoveDALoaderV2',
        params=dict(
            #image_dir='./EarthVQA/Train/images_png',
            image_dir='./EarthVQA/Test/images_png',
            mask_dir=None,
            transforms=Compose([
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=4,
            num_workers=4,
        ),
    ),
)

optimizer = dict(
    type='adamw',
    params=dict(
        lr=1e-4, 
        betas=(0.9, 0.999), 
        weight_decay=0.05,
    ),
)

learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=1e-4,
        power=0.9,
        max_iters=15000,
    ))
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=30,
    eval_interval_epoch=30,
)

test = dict(

)
