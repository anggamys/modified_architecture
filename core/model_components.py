"""
Model components dan loss functions - refactored untuk clarity.
Dipisah dari main model file untuk better organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss untuk handling class imbalance.
    Focuses training pada hard, misclassified examples.

    Paper: Lin et al., "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Args:
            alpha: Balancing factor
            gamma: Focusing parameter (higher = more focus pada hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (B, num_classes) - raw model outputs
            targets: (B,) - target class indices

        Returns:
            Scalar loss value (mean across batch)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CharEmbedding(nn.Module):
    """Character embedding layer dengan dropout"""

    def __init__(self, vocab_size: int, emb_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: Tensor) -> Tensor:
        """
        Args:
            char_ids: (B, S, W) - batch_size, seq_len, word_len

        Returns:
            (B, S, W, emb_dim)
        """
        x = self.embedding(char_ids)
        return self.dropout(x)


class CharCNN(nn.Module):
    """
    Character-level CNN encoder.
    Encodes character sequences sebagai fixed-size vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        num_filters: int = 96,
        kernel_sizes: tuple = (2, 3, 4, 5),
        output_dim: int = 128,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.char_emb = CharEmbedding(vocab_size, emb_dim, dropout=0.15)

        # Convolution layers
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, num_filters, kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

        # Layer normalization untuk stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(num_filters) for _ in kernel_sizes]
        )

        # Projection ke output dimension
        self.proj = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: Tensor) -> Tensor:
        """
        Args:
            char_ids: (B, S, W) - batch_size, seq_len, word_len

        Returns:
            (B, S, output_dim)
        """
        B, S, W = char_ids.shape

        # Flatten untuk processing
        x = char_ids.reshape(-1, W)  # (B*S, W)
        x = self.char_emb(x)  # (B*S, W, emb_dim)
        x = x.permute(0, 2, 1)  # (B*S, emb_dim, W)

        # Apply convolutions + max pooling
        conv_outputs = []
        for conv, ln in zip(self.convs, self.layer_norms):
            c = conv(x)  # (B*S, num_filters, W-k+1)
            c = F.relu(c)
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)  # (B*S, num_filters)
            p = ln(p)
            conv_outputs.append(p)

        # Concatenate outputs dari semua kernel sizes
        x = torch.cat(conv_outputs, dim=1)  # (B*S, num_filters * len(kernels))
        x = self.proj(x)  # (B*S, output_dim)
        x = self.dropout(x)

        # Reshape back ke sequence format
        return x.reshape(B, S, -1)  # (B, S, output_dim)


class CharBiLSTM(nn.Module):
    """
    Char-level BiLSTM encoder.
    Reads character sequences bi-directionally untuk richer representations.
    Better untuk imbuhan jauh (di-...-in), tapi lebih slow.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 128,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.char_emb = CharEmbedding(vocab_size, emb_dim, dropout=0.15)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

        # Project dari bidirectional output ke output_dim
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: Tensor) -> Tensor:
        """
        Args:
            char_ids: (B, S, W)

        Returns:
            (B, S, output_dim)
        """
        B, S, W = char_ids.shape

        # Reshape untuk LSTM processing
        x = char_ids.reshape(-1, W)  # (B*S, W)
        x = self.char_emb(x)  # (B*S, W, emb_dim)

        # Permute untuk LSTM (expects seq_len first)
        x = x.permute(1, 0, 2)  # (W, B*S, emb_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (W, B*S, 2*hidden_dim)

        # Take last hidden state (concatenated from both directions)
        last_hidden = lstm_out[-1]  # (B*S, 2*hidden_dim)

        # Project
        x = self.proj(last_hidden)  # (B*S, output_dim)
        x = self.dropout(x)

        # Reshape back
        return x.reshape(B, S, -1)  # (B, S, output_dim)


class CharEncoder(nn.Module):
    """
    Factory untuk character encoder (CNN atau BiLSTM).
    Memilih implementasi berdasarkan config.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_type: str = "cnn",
        emb_dim: int = 32,
        num_filters: int = 96,
        kernel_sizes: tuple = (2, 3, 4, 5),
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "cnn":
            self.encoder = CharCNN(
                vocab_size=vocab_size,
                emb_dim=emb_dim,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes,
                output_dim=output_dim,
                dropout=dropout,
            )
        elif encoder_type == "bilstm":
            self.encoder = CharBiLSTM(
                vocab_size=vocab_size,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.output_dim = output_dim

    def forward(self, char_ids: Tensor) -> Tensor:
        return self.encoder(char_ids)
