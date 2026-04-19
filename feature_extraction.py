import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


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


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        # transition matrix
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions, tags, mask):
        # negative log likelihood
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood

    def _compute_log_likelihood(self, emissions, tags, mask):
        # score dari path benar
        gold_score = self._score_sentence(emissions, tags, mask)

        # semua kemungkinan path
        partition = self._compute_partition(emissions, mask)

        return gold_score - partition

    def _score_sentence(self, emissions, tags, mask):
        B, S, _ = emissions.shape

        score = torch.zeros(B, device=emissions.device)

        for t in range(S):
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)

            if t > 0:
                trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            else:
                trans_score = 0

            score += (emit_score + trans_score) * mask[:, t]

        return score.sum()

    def _compute_partition(self, emissions, mask):
        B, S, C = emissions.shape

        alpha = emissions[:, 0]  # (B, C)

        for t in range(1, S):
            emit = emissions[:, t].unsqueeze(1)  # (B, 1, C)

            trans = self.transitions.unsqueeze(0)  # (1, C, C)

            score = alpha.unsqueeze(2) + trans + emit  # (B, C, C)

            alpha = torch.logsumexp(score, dim=1)

            alpha = alpha * mask[:, t].unsqueeze(1) + alpha * (
                ~mask[:, t].unsqueeze(1)
            )

        return torch.logsumexp(alpha, dim=1).sum()

    def decode(self, emissions, mask):
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        B, S, C = emissions.shape

        score = emissions[:, 0]
        history = []

        for t in range(1, S):
            broadcast_score = score.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)

            next_score = broadcast_score + broadcast_trans
            best_score, best_path = next_score.max(dim=1)

            score = best_score + emissions[:, t]
            history.append(best_path)

        best_last_score, best_last_tag = score.max(dim=1)

        paths = [best_last_tag]

        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            paths.insert(0, best_last_tag)

        paths = torch.stack(paths, dim=1)

        return paths


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
        self.crf = CRF(num_classes)

    def forward(
        self,
        char_ids: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ):
        bert_out = self.bert(input_ids, attention_mask)  # (B,S_bert,768)
        char_out = self.char_cnn(char_ids)  # (B,S_char,128)

        # Align sequence lengths between BERT and CharCNN outputs
        S_bert = bert_out.shape[1]
        S_char = char_out.shape[1]

        if S_bert != S_char:
            # Pad the shorter sequence or truncate the longer one
            target_seq_len = min(S_bert, S_char)

            if S_bert > target_seq_len:
                bert_out = bert_out[:, :target_seq_len, :]
                attention_mask = attention_mask[:, :target_seq_len]

            if S_char > target_seq_len:
                char_out = char_out[:, :target_seq_len, :]

        x = torch.cat([bert_out, char_out], dim=-1)  # (B,S,896)
        x = F.relu(self.fusion(x))  # (B,S,256)
        x = self.dropout(x)

        emissions = self.classifier(x)  # (B,S,num_classes)

        mask = attention_mask.bool()

        if labels is not None:
            # Truncate labels to match aligned sequence length
            if labels.shape[1] > emissions.shape[1]:
                labels = labels[:, : emissions.shape[1]]
            loss = -self.crf(emissions, labels, mask=mask)
            return loss

        else:
            preds = self.crf.decode(emissions, mask=mask)
            return preds
