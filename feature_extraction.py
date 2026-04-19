import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=32,
        num_filters=128,
        kernel_sizes=(3, 4, 5),
        output_dim=128,  # supaya match ke BERT
        dropout=0.3,
    ):
        super().__init__()

        # Embedding
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.embed_dropout = nn.Dropout(0.1)

        # Multi-kernel Conv1D
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes]
        )

        # Normalisasi (stabilitas training)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_filters) for _ in kernel_sizes]
        )

        # Output projection (penting untuk hybrid dengan BERT)
        self.proj = nn.Linear(num_filters * len(kernel_sizes), output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        """
        char_ids: (B, S, W)
        B = batch
        S = seq_len (jumlah token)
        W = max_word_len
        """
        B, S, W = char_ids.shape

        # flatten token
        x = char_ids.reshape(-1, W)  # (B*S, W)

        # embedding
        x = self.char_emb(x)  # (B*S, W, emb)
        x = self.embed_dropout(x)

        # Conv1D butuh (B, C, L)
        x = x.permute(0, 2, 1)  # (B*S, emb, W)

        conv_outputs = []

        for conv, bn in zip(self.convs, self.batch_norms):
            c = conv(x)  # (B*S, num_filters, L)
            c = bn(c)
            c = F.relu(c)

            # global max pooling
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            conv_outputs.append(p)

        # concat semua kernel
        x = torch.cat(conv_outputs, dim=1)  # (B*S, num_filters * K)

        # projection ke dimensi stabil
        x = self.proj(x)

        x = self.dropout(x)

        # reshape balik ke (B, S, F)
        x = x.reshape(B, S, -1)

        return x


class Bert(nn.Module):
    def __init__(self, model_name="indobert-base-p1", output_dim=128, dropout=0.3):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, S, H)

        x = self.proj(last_hidden_state)  # (B, S, output_dim)
        x = self.dropout(x)

        return x
