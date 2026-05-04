import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from transformers import PreTrainedModel


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard, misclassified examples.
    
    Paper: Lin et al., "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha  # Balancing factor
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (B, num_classes) - raw model outputs
            targets: (B,) - target class indices
        
        Returns:
            Scalar loss value (mean across batch)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        num_filters: int = 96,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        output_dim: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim  # penting untuk fusion

        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.15)

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


class CharBiLSTM(nn.Module):
    """
    Char-level BiLSTM: membaca urutan huruf dari kiri ke kanan dan sebaliknya,
    lalu menggabungkan hidden state terakhir dari dua arah sebagai representasi kata.
    Lebih kuat dari Char-CNN untuk menangkap imbuhan jauh (di- ... -in), tapi lebih lambat.
    Digunakan pada skenario ablasi M5.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,  # Ditingkatkan dari 32
        hidden_dim: int = 128,  # Ditingkatkan dari 64
        output_dim: int = 128,
        dropout: float = 0.3,  # Disesuaikan dari 0.35
    ) -> None:
        super().__init__()

        self.output_dim = output_dim  # sama dengan CharCNN agar kompatibel di fusion

        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.15)

        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,  # Ditingkatkan ke Deep BiLSTM (dari 1)
            batch_first=True,
            bidirectional=True,
            dropout=0.3,  # Ditambahkan karena num_layers > 1
        )

        # Project hidden_dim*2 → output_dim agar dimensi sama dengan CharCNN
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: Tensor) -> Tensor:
        B, S, W = char_ids.shape

        x = char_ids.reshape(-1, W)  # (B*S, W)
        x = self.char_emb(x)  # (B*S, W, emb_dim)
        x = self.embed_dropout(x)

        # Hitung panjang nyata tiap kata (jumlah karakter non-padding)
        lengths = (char_ids.reshape(-1, W) != 0).sum(dim=1).clamp(min=1)  # (B*S,)

        # pack agar LSTM tidak memproses padding
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.bilstm(packed)  # h_n: (2, B*S, hidden_dim)

        # Gabungkan hidden state forward dan backward
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # (B*S, hidden_dim*2)
        h = self.proj(h)  # (B*S, output_dim)
        h = self.dropout(h)

        return h.reshape(B, S, -1)  # (B, S, output_dim)


class Bert(nn.Module):
    def __init__(
        self,
        bert: PreTrainedModel,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        x = outputs.last_hidden_state  # (B,S,768)
        x = self.dropout(x)

        return x


class CRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transition[i, j] = skor transisi dari tag i ke tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # skor memulai / mengakhiri dari/ke tag tertentu
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        # Langsung return mean NLL — jangan di-negate lagi di luar
        log_likelihood = self._log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _log_likelihood(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        gold_score = self._score_sentence(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return gold_score - partition

    def _score_sentence(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        B, S, _ = emissions.shape

        # mulai dari start transition ke tag pertama
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, S):
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit + trans) * mask[:, t].float()

        # Cari POSISI (bukan count) dari token valid terakhir di tiap sampel.
        # mask.sum()-1 salah karena mask non-contiguous (F,T,F,T,F) → sum=2 tapi
        # posisi terakhir adalah index 3, bukan 1.
        positions = torch.arange(S, device=mask.device).unsqueeze(0).expand(B, -1)
        last_tag_idx = (positions * mask.long()).max(dim=1).values  # (B,)
        last_tag_idx = last_tag_idx.clamp(min=0)  # guard: jika mask semua False

        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)  # (B,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        B, S, C = emissions.shape

        # inisiasi dengan start transitions + emisi posisi pertama
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)

        for t in range(1, S):
            emit = emissions[:, t].unsqueeze(1)  # (B, 1, C)
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

    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        B, S, C = emissions.shape

        seq_len = mask.long().sum(dim=1)  # (B,) — valid length per sample

        # inisiasi dengan start transitions
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        history = []

        for t in range(1, S):
            broadcast_score = score.unsqueeze(2)  # (B, C, 1)
            broadcast_trans = self.transitions.unsqueeze(0)  # (1, C, C)

            next_score = broadcast_score + broadcast_trans  # (B, C, C)
            best_score, best_path = next_score.max(dim=1)  # (B, C)

            new_score = best_score + emissions[:, t]

            # freeze score di posisi padding
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            score = torch.where(mask_t, new_score, score)
            history.append(best_path)

        # tambahkan end transitions sebelum argmax
        score += self.end_transitions.unsqueeze(0)
        _, best_last_tag = score.max(
            dim=1
        )  # (B,) — correct: score frozen at last valid step

        paths = [best_last_tag]

        for rev_step, hist in enumerate(reversed(history)):
            # cur_pos = posisi yang sedang diisi dalam paths (mundur dari S-1)
            cur_pos = S - 1 - rev_step

            prev_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)

            # Hanya update best_last_tag jika cur_pos masih di dalam rentang valid.
            # Kalau padding (cur_pos >= seq_len[b]), biarkan best_last_tag tetap —
            # agar backtracking tidak tercemar oleh history dari posisi padding.
            is_padding = cur_pos >= seq_len  # (B,) bool
            best_last_tag = torch.where(is_padding, best_last_tag, prev_tag)
            paths.insert(0, best_last_tag)

        paths = torch.stack(paths, dim=1)  # (B, S)

        # zero-out posisi padding di output akhir
        paths = paths * mask.long()

        return paths


class HybridModel(nn.Module):
    """
    Model hibrida dengan flag ablation untuk Ablation Study.
    Mendukung semua skenario berikut melalui argumen CLI:

      Skenario | word_model  | char_type | use_crf | Keterangan
      ---------|-------------|-----------|---------|------------------------------------
        M1     | bert        | none      | False   | Baseline: IndoBERT + Linear
        M2     | bert        | none      | True    | Ablasi: IndoBERT + CRF
        M3     | bert        | cnn       | True    | *) Lihat catatan M3 di bawah
        M4     | bert        | cnn       | True    | Proposed: IndoBERT + Char-CNN + CRF
        M5     | bert        | bilstm    | True    | Eksperimen: IndoBERT + Char-BiLSTM + CRF
        M6     | bert        | cnn       | True    | Sama M4, ganti --model_name IndoBERTweet

    *) M3 (Word-BiLSTM) membutuhkan pipeline data berbeda (word-level vocab),
       belum terintegrasi; jalankan dengan arsitektur terpisah.
    """

    def __init__(
        self,
        char_vocab_size: int,
        bert: PreTrainedModel,
        num_classes: int,
        class_weights: Tensor | None = None,
        fusion_dim: int = 256,
        char_type: str = "cnn",  # "none" | "cnn" | "bilstm"
        use_crf: bool = True,
        use_word_bilstm: bool = False,
    ) -> None:
        super().__init__()

        if char_type not in ("none", "cnn", "bilstm"):
            raise ValueError(
                f"char_type harus 'none', 'cnn', atau 'bilstm'. Dapat: {char_type!r}"
            )

        self.char_type = char_type
        self.use_crf = use_crf

        # encoder
        self.bert = Bert(bert)
        bert_dim = self.bert.bert.config.hidden_size

        if char_type == "cnn":
            self.char_extractor: nn.Module = CharCNN(vocab_size=char_vocab_size)
            char_dim = self.char_extractor.output_dim
            self.char_proj = nn.Linear(char_dim, bert_dim)
        elif char_type == "bilstm":
            self.char_extractor = CharBiLSTM(vocab_size=char_vocab_size)
            char_dim = self.char_extractor.output_dim
            self.char_proj = nn.Linear(char_dim, bert_dim)

        # fusion
        self.fusion = nn.Linear(bert_dim, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.15)

        self.use_word_bilstm = use_word_bilstm
        if use_word_bilstm:
            self.word_bilstm = nn.LSTM(
                input_size=fusion_dim,
                hidden_size=fusion_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # classifier head with higher dropout
        self.dropout_final = nn.Dropout(0.4)
        self.classifier = nn.Linear(fusion_dim, num_classes)

        if use_crf:
            self.crf = CRF(num_classes)

        # Loss functions
        # 1. Focal Loss for handling class imbalance (hard examples)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # 2. CE loss with class weighting for balanced learning
        # ignore_index=-100 otomatis mengecualikan non-first subword & special tokens.
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100,
        )

    def _pool_bert_to_word(
        self,
        bert_out: Tensor,
        word_ids_batch: list[list[int | None]],
        S_word: int,
    ) -> Tensor:
        B, S_bert, bert_dim = bert_out.shape
        pooled = bert_out.new_zeros(B, S_word, bert_dim)

        for b, word_ids in enumerate(word_ids_batch):
            wids = torch.tensor(
                [-1 if w is None else w for w in word_ids[:S_bert]],
                device=bert_out.device,
                dtype=torch.long,
            )

            valid = (wids >= 0) & (wids < S_word)
            valid_wids = wids[valid]

            if valid_wids.numel() > 0:
                # Tambahkan semua vektor BERT untuk id kata yang sama
                pooled[b].index_add_(0, valid_wids, bert_out[b, valid])

                # Hitung kemunculan (jumlah subword per kata) untuk average pooling
                counts = torch.bincount(valid_wids, minlength=S_word)
                counts = counts[:S_word].unsqueeze(1).clamp(min=1)

                pooled[b] = pooled[b] / counts

        return pooled

    def forward(
        self,
        char_ids: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        word_ids: list[list[int | None]],
        word_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ):
        bert_out = self.bert(input_ids, attention_mask)  # (B, S_bert, 768)
        B, S_word, char_dim = char_ids.shape[:3]

        # 1. Subword Average Pooling: (B, S_bert, 768) -> (B, S_word, 768)
        bert_pooled = self._pool_bert_to_word(bert_out, word_ids, S_word)

        # 2. Additive Fusion langsung pada level kata (S_word)
        if self.char_type != "none":
            char_out = self.char_extractor(char_ids)  # (B, S_word, char_dim)
            x = bert_pooled + self.char_proj(char_out)
        else:
            x = bert_pooled  # baseline M1/M2: BERT saja tanpa fitur karakter

        x = self.fusion_norm(F.relu(self.fusion(x)))  # (B, S_word, fusion_dim)
        x = self.dropout(x)

        # --- Word-BiLSTM ---
        if getattr(self, "use_word_bilstm", False):
            # word_mask belum ada jika dipanggil saat evaluasi standar tanpa batch?
            # Sebaiknya kita generate dari char_ids sekarang juga jika None
            if word_mask is None:
                word_mask = char_ids.sum(dim=-1) > 0

            lengths = word_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

            packed_out, _ = self.word_bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=S_word
            )

        # Apply final dropout before classifier
        x = self.dropout_final(x)
        emissions = self.classifier(x)  # (B, S_word, num_classes)

        # Jika word_mask tidak diberikan (saat inference di luar train/val), buat sendiri
        if word_mask is None:
            word_mask = char_ids.sum(dim=-1) > 0

        if labels is not None:
            if self.use_crf:
                # --- CRF Loss (pada level kata) ---
                safe_labels = labels.clamp(min=0, max=self.crf.num_tags - 1)
                crf_loss = self.crf(emissions, safe_labels, word_mask)

                # --- Focal Loss + CE Loss (auxiliary) ---
                active_loss = word_mask.view(-1)
                active_logits = emissions.view(-1, emissions.shape[-1])[active_loss]
                active_labels = labels.view(-1)[active_loss]
                
                focal_loss = self.focal_loss(active_logits, active_labels)
                ce_loss = self.ce_loss(active_logits, active_labels)

                # Hybrid loss dengan triple weighting
                # - CRF untuk structural constraints (60%)
                # - Focal untuk hard examples (25%)
                # - CE untuk balanced learning (15%)
                return 0.6 * crf_loss + 0.25 * focal_loss + 0.15 * ce_loss
            else:
                # Baseline: Focal + CE (IndoBERT + Linear)
                active_loss = word_mask.view(-1)
                active_logits = emissions.view(-1, emissions.shape[-1])[active_loss]
                active_labels = labels.view(-1)[active_loss]

                focal_loss = self.focal_loss(active_logits, active_labels)
                ce_loss = self.ce_loss(active_logits, active_labels)
                
                return 0.4 * focal_loss + 0.6 * ce_loss

        else:
            if self.use_crf:
                return self.crf.decode(emissions, mask=word_mask)  # (B, S_word)

            else:
                return emissions.argmax(dim=-1)  # (B, S_word)
