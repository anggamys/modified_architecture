import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel


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
        bert: PreTrainedModel,
        dropout: float = 0.1,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()

        self.bert = bert

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

        # transition[i, j] = skor transisi dari tag i ke tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # skor memulai / mengakhiri dari/ke tag tertentu
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        # Langsung return mean NLL — jangan di-negate lagi di luar
        log_likelihood = self._log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _log_likelihood(self, emissions, tags, mask):
        gold_score = self._score_sentence(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return gold_score - partition

    def _score_sentence(self, emissions, tags, mask):
        B, S, _ = emissions.shape

        # mulai dari start transition ke tag pertama
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, S):
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit + trans) * mask[:, t].float()

        # tambahkan end transition dari tag terakhir yang valid
        last_tag_idx = mask.long().sum(1) - 1  # (B,)
        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)  # (B,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_partition(self, emissions, mask):
        B, S, C = emissions.shape

        # inisiasi dengan start transitions + emisi posisi pertama
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)

        for t in range(1, S):
            emit = emissions[:, t].unsqueeze(1)   # (B, 1, C)
            trans = self.transitions.unsqueeze(0)  # (1, C, C)

            # score[b, prev, next] = alpha[b, prev] + trans[prev, next] + emit[b, next]
            score = alpha.unsqueeze(2) + trans + emit  # (B, C, C)
            new_alpha = torch.logsumexp(score, dim=1)  # (B, C)

            # freeze alpha di posisi padding (jangan update kalau mask=False)
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            alpha = torch.where(mask_t, new_alpha, alpha)

        # tambahkan end transitions sebelum logsumexp final
        alpha += self.end_transitions.unsqueeze(0)  # (B, C)

        return torch.logsumexp(alpha, dim=1)  # (B,) — per-sample, bukan .sum()

    def decode(self, emissions, mask):
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        B, S, C = emissions.shape

        # inisiasi dengan start transitions
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        history = []

        for t in range(1, S):
            broadcast_score = score.unsqueeze(2)           # (B, C, 1)
            broadcast_trans = self.transitions.unsqueeze(0)  # (1, C, C)

            next_score = broadcast_score + broadcast_trans  # (B, C, C)
            best_score, best_path = next_score.max(dim=1)  # (B, C)

            new_score = best_score + emissions[:, t]

            # freeze di posisi padding
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            score = torch.where(mask_t, new_score, score)
            history.append(best_path)

        # tambahkan end transitions sebelum argmax
        score += self.end_transitions.unsqueeze(0)
        _, best_last_tag = score.max(dim=1)  # (B,)

        paths = [best_last_tag]

        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            paths.insert(0, best_last_tag)

        paths = torch.stack(paths, dim=1)  # (B, S)

        # zero-out padding positions
        paths = paths * mask.long()

        return paths


class HybridModel(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        bert: PreTrainedModel,
        num_classes: int,
        fusion_dim: int = 256,
    ) -> None:
        super().__init__()

        # encoder
        self.char_cnn = CharCNN(vocab_size=char_vocab_size)
        self.bert = Bert(bert)

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

    def _align_char_to_bert(
        self,
        char_out: Tensor,
        word_ids_batch: list[list[int | None]],
        S_bert: int,
    ) -> Tensor:
        B, S_word, char_dim = char_out.shape
        aligned = torch.zeros(B, S_bert, char_dim, device=char_out.device, dtype=char_out.dtype)

        for b, word_ids in enumerate(word_ids_batch):
            for t, word_id in enumerate(word_ids):
                if t >= S_bert:
                    break
        
                if word_id is not None and word_id < S_word:
                    aligned[b, t] = char_out[b, word_id]

        return aligned

    def forward(
        self,
        char_ids: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        word_ids: list[list[int | None]],
        labels: Tensor | None = None,
    ):
        bert_out = self.bert(input_ids, attention_mask)  # (B, S_bert, 768)
        char_out = self.char_cnn(char_ids)               # (B, S_word, 128)

        char_aligned = self._align_char_to_bert(
            char_out, word_ids, S_bert=bert_out.shape[1]
        )  # (B, S_bert, 128)

        x = torch.cat([bert_out, char_aligned], dim=-1)  # (B, S_bert, 896)
        x = F.relu(self.fusion(x))                       # (B, S_bert, fusion_dim)
        x = self.dropout(x)

        emissions = self.classifier(x)   # (B, S_bert, num_classes)
        mask = attention_mask.bool()     # (B, S_bert)

        if labels is not None:
            loss = self.crf(emissions, labels, mask)
            return loss

        else:
            preds = self.crf.decode(emissions, mask=mask)
            return preds
