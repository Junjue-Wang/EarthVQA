import ever as er
from data.earthvqa import EarthVQADataset
import torch
from module.transformer import BCA, CLSMLP
from module.loss import NumericalLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import torch.nn as nn
from module.position_emb import positionalencoding1d

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, dropout=0.):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        score = self.sigmoid(out)
        x = x * score.view(score.size(0), score.size(1), 1, 1)
        x = self.dropout(x)

        return x


class SemanticAwareVQA(nn.Module):
    def __init__(self, inchannel, hidden_size, num_head, ff_size, dropout, num_layer,
                 flat_mlp_size, flat_glimpses, flat_out_size,
                 use_lang_pos_emb=False, use_img_pos_emb=False, img_feat_size=(32, 32),
                 sentence_len=len(EarthVQADataset.QUESTION_VOC), cam_ratio=16):
        super(SemanticAwareVQA, self).__init__()
        self.transformer = BCA(hidden_size, num_head, ff_size, dropout, num_layer)
        self.pre_proj = nn.Linear(inchannel, hidden_size)
        self.attflat = CLSMLP(hidden_size, flat_mlp_size, flat_glimpses, flat_out_size, dropout)
        self.img_attflat = CLSMLP(hidden_size, flat_mlp_size, flat_glimpses, flat_out_size, dropout)
        self.proj_norm = nn.LayerNorm(flat_out_size)
        self.hidden_size = hidden_size
        self.use_lang_pos_emb = use_lang_pos_emb
        self.use_img_pos_emb = use_img_pos_emb

        self.use_lang_pos_emb = use_lang_pos_emb
        self.use_img_pos_emb = use_img_pos_emb
        # image pos emb
        self.lang_pos_emb =None
        if use_lang_pos_emb:
            self.sentence_len = sentence_len
            self.lang_pos_emb = positionalencoding1d(hidden_size, self.sentence_len).to(device)
        self.img_pos_emb=None
        if use_img_pos_emb:
            self.img_feat_size = img_feat_size
            self.img_pos_emb = positionalencoding1d(hidden_size, self.img_feat_size[0] * self.img_feat_size[1]).to(device)


        self.pre_cam = ChannelAttention(inchannel, cam_ratio, dropout=0.)


    def forward(self, img_feat, question, question_feat):
        bs, c, h, w = img_feat.shape
        #print(bs, c, h, w)
        img_feat = img_feat.to(torch.float32) # feat 12 x 256 x 40 x 40
        img_feat = self.pre_cam(img_feat) + img_feat

        img_feat = img_feat.permute([0, 2, 3, 1]).reshape(bs, -1, c) # 12 x 1600 x 256
        img_feat = self.pre_proj(img_feat)
        if self.use_img_pos_emb:
            img_feat = img_feat + self.img_pos_emb
        # img_feat bmc
        question = question.unsqueeze(2)
        lang_feat_mask = self.make_mask(question)
        img_feat_mask = self.make_mask(img_feat)
        if self.use_lang_pos_emb:
            question_feat = question_feat + self.lang_pos_emb

        x, y, self_att_list, cross_att_list  = self.transformer(img_feat, question_feat, img_feat_mask, lang_feat_mask)
        # flat attention
        x = self.img_attflat(x, img_feat_mask)
        y = self.attflat(y, lang_feat_mask)
        fused_xy = self.proj_norm(x+y)
        return fused_xy, self_att_list, cross_att_list

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


@er.registry.MODEL.register()
class SOBA(er.ERModule):

    def __init__(self, config):
        super(SOBA, self).__init__(config)
        self.embedding = nn.Embedding(
            num_embeddings=self.config.ques_embed.vocab_size,
            embedding_dim=self.config.ques_embed.emb_size,
        )
        self.lstm = nn.LSTM(
            input_size=self.config.ques_embed.emb_size,
            hidden_size=self.config.ques_embed.hidden_size,
            num_layers=self.config.ques_embed.num_layers,
            batch_first=True
        )
        self.coformer =  SemanticAwareVQA(**self.config.coformer)
        self.answer_classifier = nn.Linear(self.config.coformer.flat_out_size, self.config.classes)
        self.numerical_loss = NumericalLoss(**self.config.nl)
        
        self.mask_emb = nn.Sequential(
            nn.Conv2d(8, self.config.mask_emb_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.mask_emb_dim),
            nn.ReLU()
        )


    def forward(self, x, y=None):
        ques, ques_len, ans, pred_mask = y['question'], y['question_len'], y['answer'], y['seg_mask']
        
        # mask embedding
        x = x.to(device)
        ques = ques.to(device)
        pred_mask = pred_mask.unsqueeze(1).to(device)
        pred_mask = F.interpolate(pred_mask.float(), size=x.shape[2:4], mode='nearest').squeeze(1).long()
        pred_mask[pred_mask<0] = 0
        pred_mask = F.one_hot(pred_mask, num_classes=8).permute([0, 3, 1, 2]).float()
        pred_mask = self.mask_emb(pred_mask)

        x = torch.cat([x, pred_mask], dim=1)
        
        ques_feature = self.embedding(ques)
        ques_feature, _ = self.lstm(ques_feature)

        ans_feat, self_att_list, cross_att_list = self.coformer(x, ques, ques_feature)
        ans_logits = self.answer_classifier(ans_feat)

        if self.training:
            loss_dict = self.numerical_loss(ans_logits, ans)
            return loss_dict

        else:
            return ans_logits.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            ques_embed=dict(
                vocab_size=len(EarthVQADataset.QUESTION_VOC),
                emb_size=256,
                hidden_size=512,
                num_layers=2,
                qu_feature_size=512,
                dropout=0.1,
            ),
            uper_numer_idx = 0,
            classes=len(EarthVQADataset.ANSWER_VOC),
            coformer=dict(
                inchannel=2048,
                hidden_size=512,
                num_head=8,
                ff_size=1024,
                dropout=0.1,
                num_layer=6,
                flat_mlp_size=512,
                flat_glimpses=1,
                flat_out_size=256,
            ),
            nl=dict(),
            mask_emb_dim=128
        ))
