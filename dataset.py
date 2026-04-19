import torch
import numpy as np
import pandas as pd

from functools import partial
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from preprocess import prepare_char_ids


class POSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        char_vocab: dict[str, int],
        class_to_idx: dict[str, int],
        tokenizer: PreTrainedTokenizerBase,
        max_word_len: int = 50,
        max_seq_len: int = 512,
    ) -> None:
        self.char_vocab = char_vocab
        self.class_to_idx = class_to_idx
        self.tokenizer = tokenizer
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len

        # Kelompokkan baris per kalimat, pertahankan urutan dokumen asli
        self.sentences: list[pd.DataFrame] = [
            group.reset_index(drop=True)
            for _, group in df.groupby("global_sentence_id", sort=False)
        ]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        sent_df = self.sentences[idx]

        tokens: list[str] = sent_df["token"].astype(str).tolist()
        pos_tags: list[str] = sent_df["pos_tag"].tolist()

        # Label word-level: (S_word,)
        labels: list[int] = [self.class_to_idx.get(tag, 0) for tag in pos_tags]

        # CharCNN ids: (S_word, max_word_len) — word-level
        char_ids: np.ndarray = prepare_char_ids(tokens, self.char_vocab, self.max_word_len)

        # BERT tokenisasi dengan is_split_into_words=True agar dapat word_ids()
        # Tidak ada padding di sini — padding dilakukan di collate_fn
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

        # word_ids harus diambil sebelum squeeze / ke device
        word_ids: list[int | None] = encoding.word_ids(batch_index=0)

        return {
            "char_ids": torch.from_numpy(char_ids),           # (S_word, W)
            "input_ids": encoding["input_ids"].squeeze(0),    # (S_bert,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (S_bert,)
            "word_ids": word_ids,                             # list[int | None]
            "labels": torch.tensor(labels, dtype=torch.long), # (S_word,)
        }


def pos_collate_fn(
    batch: list[dict],
    pad_token_id: int = 0,
) -> dict:
    S_word_max = max(item["char_ids"].shape[0] for item in batch)
    S_bert_max = max(item["input_ids"].shape[0] for item in batch)
    W = batch[0]["char_ids"].shape[1]
    B = len(batch)

    char_ids = torch.zeros(B, S_word_max, W, dtype=torch.long)
    input_ids = torch.full((B, S_bert_max), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(B, S_bert_max, dtype=torch.long)
    labels = torch.zeros(B, S_word_max, dtype=torch.long)
    word_ids_batch: list[list[int | None]] = []

    for i, item in enumerate(batch):
        s_word = item["char_ids"].shape[0]
        s_bert = item["input_ids"].shape[0]

        char_ids[i, :s_word] = item["char_ids"]
        input_ids[i, :s_bert] = item["input_ids"]
        attention_mask[i, :s_bert] = item["attention_mask"]
        labels[i, :s_word] = item["labels"]

        # Padding positions di word_ids diberi None agar model tahu itu bukan kata
        padded_wids: list[int | None] = item["word_ids"] + [None] * (S_bert_max - s_bert)
        word_ids_batch.append(padded_wids)

    return {
        "char_ids": char_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": word_ids_batch,
        "labels": labels,
    }


def make_collate_fn(pad_token_id: int):
    return partial(pos_collate_fn, pad_token_id=pad_token_id)
