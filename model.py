import torch
import torch.nn as nn
import torch.nn.functional as F
"""
基本模块
Embedding -> Encoder -> (VAE) -> Decoder -> BeamSearch
"""


class CustomEmbedding(nn.Module):
    def __init__(self, config):
        super(CustomEmbedding, self).__init__()
        self.config = config
        # +1 for UNK
        self.char_lookup = nn.Embedding(config.char_table_size + 2, config.char_dim, padding_idx=0)
        self.word_lookup = nn.Embedding(config.word_table_size + 2, config.word_dim, padding_idx=0)
        self.fc_c2h = nn.Linear(config.char_dim, config.hidden_dim, bias=False)
        self.fc_w2h = nn.Linear(config.word_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=config.dropout_level)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_dim)

    def forward(self, ids):
        if self.config.use_mix_embedding and len(ids) == 2:
            ids_char, ids_word = ids
            emb_char = self.fc_c2h(self.char_lookup(ids_char))
            emb_word = self.fc_w2h(self.word_lookup(ids_word))
            emb_mix = self.dropout(self.layer_norm(emb_char + emb_word))
            return emb_mix
        else:
            emb_char = self.fc_c2h(self.char_lookup(ids))
            emb_char = self.dropout(self.layer_norm(emb_char))
            return emb_char


class GatedCNN(nn.Module):
    def __init__(self, hidden_dim, dropout_level, kernel_size=3, dilation=1):
        super(GatedCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_dim,
                               out_channels=hidden_dim,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding=(dilation * (kernel_size - 1)) // 2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim,
                               out_channels=hidden_dim,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding=(dilation * (kernel_size - 1)) // 2)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, inputs):
        src, mask = inputs
        src1 = self.conv1(src)
        src2 = self.activation(self.dropout(self.conv2(src)))
        # src = (N, H, L), mask = (N, L)
        # mask.unsqueeze(1) = (N, 1, L)
        src = (src * (1 - src2) + src1 * src2) * mask.unsqueeze(1)
        return src, mask


class PoemModel(nn.Module):
    def __init__(self, config):
        super(PoemModel, self).__init__()
        self.embedding = CustomEmbedding(config)
        self.cnns = nn.Sequential(GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level),
                                  GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level),
                                  GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level))
        self.fc = nn.Linear(config.hidden_dim, config.char_table_size+1, bias=True)  # no UNK

    def forward(self, inputs):
        ids, mask = inputs
        emb = self.embedding(ids)
        ht, mask = self.cnns((emb.permute(0, 2, 1), mask))
        logit = self.fc(ht.permute(0, 2, 1))
        return logit


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        ...


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def forward(self):
        ...