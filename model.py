import math
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
    def __init__(self, hidden_dim, dropout_level, kernel_size=3, dilation=1, add_fc_layer=True):
        super(GatedCNN, self).__init__()
        self.add_fc_layer = add_fc_layer
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
        if add_fc_layer:
            self.fc1 = nn.Linear(in_features=hidden_dim, out_features=4 * hidden_dim, bias=False)
            self.fc2 = nn.Linear(in_features=4 * hidden_dim, out_features=hidden_dim, bias=False)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, inputs):
        src, mask = inputs
        src1 = self.conv1(src)
        src2 = self.activation(self.dropout(self.conv2(src)))
        # src = (N, H, L), mask = (N, L)
        # mask.unsqueeze(1) = (N, 1, L)
        src = (src * (1 - src2) + src1 * src2) * mask.unsqueeze(1)
        if self.add_fc_layer:
            src = self.fc2(F.relu(self.fc1(src.permute(0, 2, 1)))).permute(0, 2, 1)
        return src, mask


class CoupletSeqLabeling(nn.Module):
    def __init__(self, config):
        super(CoupletSeqLabeling, self).__init__()
        self.config = config
        self.embedding = CustomEmbedding(config)
        if config.model_type == "GCNN":
            self.encoder = nn.Sequential(GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level),
                                         GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level),
                                         GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level))
            self.fc = nn.Linear(config.hidden_dim, config.char_table_size + 2, bias=True)
        elif config.model_type == "LSTM":
            self.encoder = nn.LSTM(input_size=config.hidden_dim,
                                   hidden_size=config.hidden_dim,
                                   batch_first=True,
                                   bidirectional=True)
            self.fc = nn.Linear(config.hidden_dim * 2, config.char_table_size + 2, bias=True)
        else:
            raise Exception("Wrong Model Type")

    def forward(self, inputs):
        ids, mask, lengths = inputs
        emb = self.embedding(ids)
        if self.config.model_type == "GCNN":
            ht, mask = self.encoder((emb.permute(0, 2, 1), mask))
            logit = self.fc(ht.permute(0, 2, 1))
        elif self.config.model_type == "LSTM":
            emb = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
            ht = self.encoder(emb)
            ht, lengths = nn.utils.rnn.pad_packed_sequence(ht[0], batch_first=True)
            logit = self.fc(ht)
        else:
            raise Exception("Wrong Model Type")
        return logit


class CoupletSeq2SeqWithVAE(nn.Module):
    def __init__(self, config):
        super(CoupletSeq2SeqWithVAE, self).__init__()
        self.config = config
        self.embedding = CustomEmbedding(config)
        self.encoder = nn.Sequential(GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level, add_fc_layer=False),
                                     GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level, add_fc_layer=False))
        self.pooling = AttentionPooling1D(config.hidden_dim)
        self.fc_mean = nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim, bias=True)
        self.fc_log_var = nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim, bias=True)
        self.decoder = VAEDecoder(config)

    def forward(self, inputs):
        ids, mask, lengths = inputs
        emb = self.embedding(ids)
        ht, mask = self.encoder((emb.permute(0, 2, 1), mask))
        if self.config.use_attention:
            ht = self.pooling(ht.permute(0, 2, 1), mask)
        else:
            ht = torch.sum(ht.permute(0, 2, 1), dim=1) / ht.shape[2]
        z_mean = self.fc_mean(ht)
        z_log_var = self.fc_log_var(ht)
        z = self.reparameterize(z_mean, z_log_var)

        logits = self.decoder((z, mask))
        kld_loss = self.loss_function(z_mean, z_log_var)

        return logits, kld_loss

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(log_var / 2) * epsilon

    def loss_function(self, mean, log_var):
        kld_loss = -0.5 * torch.sum(1 + log_var - torch.square(mean) - torch.exp(log_var), dim=1)
        kld_loss = torch.mean(kld_loss, dim=0)
        return kld_loss

    def get_decoder(self):
        return self.decoder


class AttentionPooling1D(nn.Module):
    """
    soft attention layer
    公式: a = x * softmax(U*tanh(Wx))
    input:
        src: hidden state of t_step, shape = (N, L, H)
        mask: padding masks, shape = (N, L)
    output:
        attention value: shape = (N, H)
    """
    def __init__(self, hidden_size):
        super(AttentionPooling1D, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, 1, bias=False)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src, mask):
        # src = (N, L, H)
        # mask = (N, L)
        logit = self.U(self.activation(self.W(src)))  # (batch_size, length, 1)
        # softmax前做归一化，防止方差随着维度增加而增加，避免softmax的结果差距过大
        logit = logit.squeeze() / math.sqrt(src.shape[2])
        if mask is not None:
            logit = logit - (1 - mask) * 10000  # padding mask
        prob = self.softmax(logit).unsqueeze(-1)  # (batch_size, length, 1)
        attention_value = torch.sum(src * prob, dim=1)  # (batch_size, hidden_size)
        return attention_value


class VAEDecoder(nn.Module):
    def __init__(self, config):
        super(VAEDecoder, self).__init__()
        self.config = config
        self.fc_vae = nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim * (2 * config.max_length + 1))
        self.decoder = nn.Sequential(GatedCNN(hidden_dim=config.hidden_dim, dropout_level=config.dropout_level, add_fc_layer=False))
        self.fc_out = nn.Linear(in_features=config.hidden_dim, out_features=config.char_table_size + 2)

    def forward(self, inputs):
        z, mask = inputs
        hz = self.fc_vae(z)
        hz = hz.view(-1, 2 * self.config.max_length + 1, self.config.hidden_dim)
        hz, mask = self.decoder((hz.permute(0, 2, 1), mask))
        logits = self.fc_out(hz.permute(0, 2, 1))
        return logits
