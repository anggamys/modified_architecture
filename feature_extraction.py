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
        output_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.embed_dropout = nn.Dropout(0.1)

        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes]
        )

        # LayerNorm setelah pooling untuk setiap conv
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(num_filters) for _ in kernel_sizes]
        )

        self.proj = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        B, S, W = char_ids.shape

        x = char_ids.reshape(-1, W)  # (B*S, W)
        x = self.char_emb(x)  # (B*S, W, emb)
        x = self.embed_dropout(x)

        x = x.permute(0, 2, 1)  # (B*S, emb, W)

        conv_outputs = []

        for conv, ln in zip(self.convs, self.layer_norms):
            c = conv(x)  # (B*S, F, L)
            c = F.relu(c)

            # pooling dulu
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)

            # baru normalize
            p = ln(p)

            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)  # (B*S, F*K)
        x = self.proj(x)
        x = self.dropout(x)

        return x.reshape(B, S, -1)


class Bert(nn.Module):
    def __init__(
        self, model_name="indobenchmark/indobert-base-p1", dropout=0.1, freeze_bert=True
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # optional: unfreeze layer atas
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        x = outputs.last_hidden_state  # (B, S, 768)
        x = self.dropout(x)

        return x
