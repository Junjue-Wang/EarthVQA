import torch.nn as nn
import torch.nn.functional as F
import torch, math


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_size, mid_size)
        self.linear2 = nn.Linear(mid_size, out_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)

        return x

class ProjMLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(ProjMLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class CLSMLP(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size, flat_glimpses, flat_out_size, dropout):
        super(CLSMLP, self).__init__()

        self.mlp = ProjMLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            dropout_r=dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )
        self.flat_glimpses = flat_glimpses

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e4
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# Multi-Head Attention

class MHAtt(nn.Module):
    def __init__(self, hidden_size, num_head):
        super(MHAtt, self).__init__()
        self.num_head = num_head
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, v, k, q, mask, q_pos=None, k_pos=None):
        if q_pos is not None:
            q = q + q_pos
        if k_pos is not None:
            k = k + k_pos

        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_head,
            self.hidden_size // self.num_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_head,
            self.hidden_size // self.num_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_head,
            self.hidden_size // self.num_head
        ).transpose(1, 2)

        atted, att_map = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)
        return atted, att_map

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value), att_map


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
        )

    def forward(self, x):
        return self.mlp(x)


# Visual Self Attention


class VSA(nn.Module):
    def __init__(self, hidden_size, num_head, ff_size, dropout):
        super(VSA, self).__init__()

        self.mhatt = MHAtt(hidden_size, num_head)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        nx = self.norm1(x)
        nx, att_map = self.mhatt(nx, nx, nx, x_mask)
        x = x + self.dropout1(nx)

        nx = self.norm2(x)
        x = x + self.dropout2(self.ffn(nx))

        return x, att_map


# Multi-modal Cross Attention
class MCA(nn.Module):
    def __init__(self, hidden_size, num_head, ff_size, dropout):
        super(MCA, self).__init__()

        self.mhatt2 = MHAtt(hidden_size, num_head)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        ny = self.norm2(y)
        ny, att_map = self.mhatt2(x, x, ny, x_mask)
        y = y + self.dropout2(ny)

        ny = self.norm3(y)
        y = y + self.dropout3(
            self.ffn(ny)
        )

        return y, att_map


class BCA(nn.Module):
    def __init__(self, hidden_size, num_head, ff_size, dropout, num_layer):
        super(BCA, self).__init__()

        self.enc_list = nn.ModuleList([VSA(hidden_size, num_head, ff_size, dropout) for _ in range(num_layer)])
        self.dec_list1 = nn.ModuleList([MCA(hidden_size, num_head, ff_size, dropout) for _ in range(num_layer)])
        self.dec_list2 = nn.ModuleList([MCA(hidden_size, num_head, ff_size, dropout) for _ in range(num_layer)])


    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        self_att_list = []
        cross_att_list = []
        for enc in self.enc_list:
            x, att_map = enc(x, x_mask)
            self_att_list.append(att_map)

        for dec in self.dec_list1:
            x, att_map = dec(y, x, y_mask, x_mask)
            cross_att_list.append(att_map)

        for dec in self.dec_list2:
            y, att_map = dec(x, y, x_mask, y_mask)
            cross_att_list.append(att_map)

        return x, y, self_att_list, cross_att_list

