from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Compose, Transpose
from module.utils import RandomDropChannel



data = dict(
    train=dict(
        type='EarthVQALoader',
        params=dict(
            image_feat_dir='./log/sfpnr50/train_features',
            qa_path='./EarthVQA/Train_QA.json',
            feat_load=True,
            pred_mask_load=True,
            mask_load=False,
            image_load=False,
            encoding=True,
            image_transforms=None,
            vqa_transforms=Compose([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Transpose(p=0.5),
                RandomRotate90(p=0.5),
                RandomDropChannel(p=0.5, drop_rate=0.1),
                RandomRotate90(p=0.5),
                RandomRotate90(p=0.5)
            ], is_check_shapes=False),
            training=True,
            batch_size=16,
            num_workers=6,
        ),
    ),
    test=dict(
        type='EarthVQALoader',
        params=dict(
            image_feat_dir='./log/sfpnr50/test_features',
            qa_path='./EarthVQA/Test_QA.json',
            feat_load=True,
            pred_mask_load=True,
            mask_load=False,
            image_load=False,
            encoding=True,
            image_transforms=None,
            vqa_transforms=None,
            training=False,
            batch_size=32,
            num_workers=6,
        ),
    ),
)


optimizer = dict(
    type='adamw',
    params=dict(
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)

learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=5e-5,
        power=0.9,
        max_iters=40000,
    ))
train = dict(
    forward_times=1,
    num_iters=40000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1,
    eval_interval_epoch=1,
)

test = dict(

)
