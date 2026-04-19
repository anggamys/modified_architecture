import string
import unicodedata
import numpy as np
import pandas as pd

from typing import Tuple
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

from utils import log, log_level


def normalize_text(text: str) -> str:
    # Lowercase
    text = str(text).lower()

    # Unify quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Normalize unicode form (NFKC)
    text = unicodedata.normalize("NFKC", text)

    return text


def clean_text(text: str) -> str:
    # Remove control characters (category C)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")

    return text


def split_train_val_test(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    test_ratio: float = 0.125,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.6f}"
        )

    df = dataframe.copy()
    df["_doc_id"] = df["global_sentence_id"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )

    groups = df["_doc_id"]

    n_docs = groups.nunique()

    log(
        f"Total dokumen: {n_docs} | Total kalimat: {df['global_sentence_id'].nunique()}",
        level=log_level.INFO,
    )

    gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(X=df, y=df["pos_tag"], groups=groups))

    train_df = df.iloc[train_idx].drop(columns="_doc_id").reset_index(drop=True)
    df_temp = df.iloc[temp_idx]

    val_size_relative = val_ratio / (val_ratio + test_ratio)
    gss_val = GroupShuffleSplit(
        n_splits=1, train_size=val_size_relative, random_state=seed
    )
    val_idx, test_idx = next(
        gss_val.split(X=df_temp, y=df_temp["pos_tag"], groups=df_temp["_doc_id"])
    )

    val_df = df_temp.iloc[val_idx].drop(columns="_doc_id").reset_index(drop=True)
    test_df = df_temp.iloc[test_idx].drop(columns="_doc_id").reset_index(drop=True)

    train_docs = (
        train_df["global_sentence_id"]
        .apply(lambda x: "_".join(x.split("_")[:-1]))
        .nunique()
    )

    val_docs = (
        val_df["global_sentence_id"]
        .apply(lambda x: "_".join(x.split("_")[:-1]))
        .nunique()
    )

    test_docs = (
        test_df["global_sentence_id"]
        .apply(lambda x: "_".join(x.split("_")[:-1]))
        .nunique()
    )

    log(
        f"Split (dokumen) → train: {train_docs} | val: {val_docs} | test: {test_docs}",
        level=log_level.INFO,
    )

    log(
        f"Split (token) → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}",
        level=log_level.INFO,
    )

    return train_df, val_df, test_df


def class_distribution(dataframe: pd.DataFrame, column: str) -> None:
    distribution = (
        dataframe[column].value_counts(normalize=True).sort_values(ascending=False)
    )

    summary = " | ".join(
        f"{label}: {percentage:.4f}" for label, percentage in distribution.items()
    )

    log(f"Class distribution for '{column}': {summary}", level=log_level.INFO)


def build_char_vocab(
    dataframe: pd.DataFrame, min_freq: int = 5, include_emoji: bool = False
) -> dict:
    char_vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    # Kumpulkan semua karakter dan frekuensinya
    char_freq = Counter()

    for token in dataframe["token"].astype(str):
        # Aplikasikan pipeline: normalize → clean
        normalized = normalize_text(token)
        cleaned = clean_text(normalized)

        for char in cleaned:
            char_freq[char] += 1

    # Standard printable chars (lowercase)
    standard_chars = string.ascii_lowercase + string.digits + " .,!?'-"

    log(f"Total unique chars sebelum filter: {len(char_freq)}", level=log_level.INFO)

    # Tambah standard chars
    for char in standard_chars:
        if char not in char_vocab:
            char_vocab[char] = idx
            idx += 1

    # Tambah char lain yang frequent enough
    for char, freq in sorted(char_freq.items(), key=lambda x: x[1], reverse=True):
        if freq >= min_freq:
            if char not in char_vocab and char not in standard_chars:
                # Check if emoji (optional)
                if not include_emoji and _is_emoji(char):
                    continue
                char_vocab[char] = idx
                idx += 1

    log(
        f"Total unique chars setelah filter (min_freq={min_freq}): {len(char_vocab)}",
        level=log_level.INFO,
    )

    log(f"Vocab size: {len(char_vocab)}", level=log_level.INFO)

    # Log distribusi karakter
    _log_char_distribution(char_freq, min_freq)

    return char_vocab


def _is_emoji(char: str) -> bool:
    # Sederhana: cek jika char berada di rentang emoji Unicode
    return ord(char) > 0x1F300


def _log_char_distribution(char_freq: Counter, min_freq: int = 5) -> None:
    top_chars = char_freq.most_common(15)

    parts = []
    for char, freq in top_chars:
        whitespace_chars = {" ", "\n", "\t"}
        display_char = repr(char) if char in whitespace_chars else char
        parts.append(f"'{display_char}': {freq}x")

    summary = " | ".join(parts)

    log(f"Top 15 karakter: {summary}", level=log_level.INFO)

    rare_chars = sum(1 for _, freq in char_freq.items() if freq < min_freq)

    if rare_chars > 0:
        log(f"Karakter langka (< {min_freq}x): {rare_chars}", level=log_level.WARNING)


def check_vocab_coverage(dataframe: pd.DataFrame, char_vocab: dict) -> None:
    total_chars_in_data = 0
    covered_chars = 0
    missing_chars_freq = {}

    for token in dataframe["token"].astype(str):
        # Aplikasikan pipeline yang sama seperti saat build vocab
        normalized = normalize_text(token)
        cleaned = clean_text(normalized)

        for char in cleaned:
            total_chars_in_data += 1

            if char in char_vocab:
                covered_chars += 1
            else:
                if char not in missing_chars_freq:
                    missing_chars_freq[char] = 1
                else:
                    missing_chars_freq[char] += 1

    if total_chars_in_data == 0:
        log("Tidak ada karakter dalam data", level=log_level.WARNING)

        return

    coverage_percent = (covered_chars / total_chars_in_data) * 100

    log(f"Vocab Coverage: {coverage_percent:.4f}%", level=log_level.INFO)

    if len(missing_chars_freq) > 0:
        sorted_missing = sorted(
            missing_chars_freq.items(), key=lambda x: x[1], reverse=True
        )

        log(f"OOV characters: {len(missing_chars_freq)}", level=log_level.WARNING)

        oov_summary = " | ".join(
            f"{repr(char)}: {freq}x" for char, freq in sorted_missing
        )

        log(f"OOV detail: {oov_summary}", level=log_level.WARNING)


def prepare_char_ids(tokens, char_vocab, max_word_len=50):
    """
    Konversi token strings menjadi char IDs.
    Shape: (batch_size, max_word_len)
    """
    char_ids_list = []

    for token in tokens:
        # Normalize & clean sesuai pipeline preprocessing
        token_clean = clean_text(normalize_text(token))

        # Konversi ke char IDs
        char_ids = [char_vocab.get(c, char_vocab.get("<UNK>", 1)) for c in token_clean]

        # Padding/truncate ke max_word_len
        if len(char_ids) < max_word_len:
            char_ids = char_ids + [char_vocab.get("<PAD>", 0)] * (
                max_word_len - len(char_ids)
            )
        else:
            char_ids = char_ids[:max_word_len]

        char_ids_list.append(char_ids)

    return np.array(char_ids_list, dtype=np.int64)
