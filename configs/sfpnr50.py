from configs.lovedav2 import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='SemanticFPN',
        params=dict(
            encoder=dict(
                name='resnet50',
                weights='imagenet',
                in_channels=3,
                output_stride=32,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                num_groups_gn=None
            ),
            classes=8,
            loss=dict(
                ignore_index=-1,
            )
        )),
        data=data,
        optimizer=optimizer,
        learning_rate=learning_rate,
        train=train,
        test=test
    )
