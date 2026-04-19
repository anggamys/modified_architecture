import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel
from torchcrf import CRF  # type: ignore


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        num_filters: int = 96,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim  # penting untuk fusion

        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)

        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(num_filters) for _ in kernel_sizes]
        )

        self.proj = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: Tensor) -> Tensor:
        B, S, W = char_ids.shape

        x = char_ids.reshape(-1, W)  # (B*S, W)
        x = self.char_emb(x)  # (B*S, W, emb)
        x = self.embed_dropout(x)

        x = x.permute(0, 2, 1)  # (B*S, emb, W)

        conv_outputs = []

        for conv, ln in zip(self.convs, self.layer_norms):
            c = conv(x)
            c = F.relu(c)

            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            p = ln(p)

            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)
        x = self.proj(x)
        x = self.dropout(x)

        return x.reshape(B, S, -1)


class Bert(nn.Module):
    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        dropout: float = 0.1,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # unfreeze 2 layer terakhir
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        x = outputs.last_hidden_state  # (B,S,768)
        x = self.dropout(x)

        return x


class HybridModel(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        num_classes: int,
        fusion_dim: int = 256,
    ) -> None:
        super().__init__()

        # encoder
        self.char_cnn = CharCNN(vocab_size=char_vocab_size)
        self.bert = Bert()

        # ambil dimensi dinamis
        bert_dim = self.bert.bert.config.hidden_size
        char_dim = self.char_cnn.output_dim

        # fusion
        self.fusion = nn.Linear(bert_dim + char_dim, fusion_dim)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # CRF
        self.crf = CRF(num_classes, batch_first=True)

    def forward(
        self,
        char_ids: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ):
        bert_out = self.bert(input_ids, attention_mask)  # (B,S,768)
        char_out = self.char_cnn(char_ids)  # (B,S,128)

        x = torch.cat([bert_out, char_out], dim=-1)  # (B,S,896)
        x = F.relu(self.fusion(x))  # (B,S,256)
        x = self.dropout(x)

        emissions = self.classifier(x)  # (B,S,num_classes)

        mask = attention_mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss

        else:
            preds = self.crf.decode(emissions, mask=mask)
            return preds
