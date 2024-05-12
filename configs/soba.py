from configs.earthvqa import train, test, data, optimizer, learning_rate
from data.earthvqa import EarthVQADataset

dim=384

config = dict(
    model=dict(
        type='SOBA',
        params=dict(
            ques_embed=dict(
                vocab_size=len(EarthVQADataset.QUESTION_VOC),
                emb_size=256,
                hidden_size=dim,
                num_layers=2,
                qu_feature_size=dim,
                dropout=0.,
            ),
            classes=len(EarthVQADataset.ANSWER_VOC),
            mask_emb_dim=128,
            coformer=dict(
                inchannel=2048+128,
                hidden_size=dim,
                num_head=8,
                ff_size=dim,
                dropout=0.1,
                num_layer=3,
                flat_mlp_size=dim,
                flat_glimpses=1,
                flat_out_size=dim,
                cam_ratio=16,
                use_lang_pos_emb=True,
                use_img_pos_emb=True,
            ),
            nl=dict(
                uper_numer_idx=17,
                alpha=1.,
                gamma=0.5,
            ),
            
    )),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
