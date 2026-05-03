import torch
import numpy as np
import pandas as pd

from functools import partial
from typing import Dict, List, Optional, Any, Callable
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding

from preprocess import prepare_char_ids


class POSDataset(Dataset):
    """
    Dataset untuk POS tagging dengan support untuk Char-CNN dan BERT.

    Attributes:
        char_vocab: Mapping dari character ke index
        class_to_idx: Mapping dari POS class (string) ke index (int)
        tokenizer: BERT tokenizer dari HuggingFace
        max_word_len: Maximum character sequence per word
        max_seq_len: Maximum BERT token sequence length
        sentences: List of grouped sentences (per global_sentence_id)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        char_vocab: Dict[str, int],
        class_to_idx: Dict[str, int],
        tokenizer: PreTrainedTokenizerBase,
        max_word_len: int = 50,
        max_seq_len: int = 512,
    ) -> None:
        self.char_vocab: Dict[str, int] = char_vocab
        self.class_to_idx: Dict[str, int] = class_to_idx
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_word_len: int = max_word_len
        self.max_seq_len: int = max_seq_len

        # Kelompokkan baris per kalimat, pertahankan urutan dokumen asli
        self.sentences: List[pd.DataFrame] = [
            group.reset_index(drop=True)
            for _, group in df.groupby("global_sentence_id", sort=False)
        ]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sent_df: pd.DataFrame = self.sentences[idx]

        tokens: List[str] = sent_df["token"].astype(str).tolist()
        pos_tags: List[str] = sent_df["pos_tag"].tolist()

        # Label word-level: (S_word,)
        labels: List[int] = [self.class_to_idx.get(tag, 0) for tag in pos_tags]

        # CharCNN ids: (S_word, max_word_len) — word-level
        char_ids: np.ndarray = prepare_char_ids(
            tokens, self.char_vocab, self.max_word_len
        )

        # BERT tokenisasi dengan is_split_into_words=True agar dapat word_ids()
        # Tidak ada padding di sini — padding dilakukan di collate_fn
        encoding: BatchEncoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

        # word_ids harus diambil sebelum squeeze / ke device
        word_ids: List[Optional[int]] = encoding.word_ids()

        return {
            "char_ids": torch.from_numpy(char_ids),  # (S_word, W)
            "input_ids": torch.tensor(
                encoding["input_ids"], dtype=torch.long
            ),  # (S_bert,)
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),  # (S_bert,)
            "word_ids": word_ids,  # list[int | None]
            "labels": torch.tensor(labels, dtype=torch.long),  # (S_word,)
        }


def pos_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Custom collate function untuk padding sequences ke ukuran batch.

    Args:
        batch: List of sample dictionaries dari POSDataset
        pad_token_id: Token ID untuk padding

    Returns:
        Dictionary dengan batched tensors dan metadata
    """
    # Calculate max lengths in this batch
    S_word_max: int = max(item["char_ids"].shape[0] for item in batch)
    S_bert_max: int = max(item["input_ids"].shape[0] for item in batch)
    W: int = batch[0]["char_ids"].shape[1]
    B: int = len(batch)

    # Initialize padded tensors
    char_ids: Tensor = torch.zeros(B, S_word_max, W, dtype=torch.long)
    input_ids: Tensor = torch.full((B, S_bert_max), pad_token_id, dtype=torch.long)
    attention_mask: Tensor = torch.zeros(B, S_bert_max, dtype=torch.long)
    word_mask: Tensor = torch.zeros(B, S_word_max, dtype=torch.bool)
    labels: Tensor = torch.zeros(B, S_bert_max, dtype=torch.long)
    word_ids_batch: List[List[Optional[int]]] = []

    # Fill tensors for each sample in batch
    for i, item in enumerate(batch):
        s_word: int = item["char_ids"].shape[0]
        s_bert: int = item["input_ids"].shape[0]

        char_ids[i, :s_word] = item["char_ids"]
        input_ids[i, :s_bert] = item["input_ids"]
        attention_mask[i, :s_bert] = item["attention_mask"]
        word_mask[i, :s_word] = True

        # Map word-level labels to BERT token-level using word_ids
        word_ids: List[Optional[int]] = item["word_ids"]
        word_labels: Tensor = item["labels"]  # (S_word,)

        for t, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(word_labels):
                labels[i, t] = word_labels[word_id]

        # Padding positions di word_ids diberi None agar model tahu itu bukan kata
        padded_wids: List[Optional[int]] = item["word_ids"] + [None] * (
            S_bert_max - s_bert
        )
        word_ids_batch.append(padded_wids)

    return {
        "char_ids": char_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_mask": word_mask,
        "word_ids": word_ids_batch,
        "labels": labels,
    }


def make_collate_fn(pad_token_id: int) -> Callable:
    """
    Factory function untuk membuat collate_fn dengan pad_token_id yang spesifik.

    Args:
        pad_token_id: Token ID untuk padding

    Returns:
        Callable collate function
    """
    return partial(pos_collate_fn, pad_token_id=pad_token_id)
