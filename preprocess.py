import string
import pandas as pd
from typing import Tuple
from sklearn.model_selection import GroupShuffleSplit

from utils import log, log_level


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
    log(f"Total dokumen: {n_docs} | Total kalimat: {df['global_sentence_id'].nunique()}", level=log_level.INFO)

    gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(X=df, y=df["pos_tag"], groups=groups))

    train_df = df.iloc[train_idx].drop(columns="_doc_id").reset_index(drop=True)
    df_temp = df.iloc[temp_idx]

    val_size_relative = val_ratio / (val_ratio + test_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, train_size=val_size_relative, random_state=seed)
    val_idx, test_idx = next(
        gss_val.split(X=df_temp, y=df_temp["pos_tag"], groups=df_temp["_doc_id"])
    )

    val_df = df_temp.iloc[val_idx].drop(columns="_doc_id").reset_index(drop=True)
    test_df = df_temp.iloc[test_idx].drop(columns="_doc_id").reset_index(drop=True)

    log(
        f"Split (dokumen) → train: {train_df['global_sentence_id'].apply(lambda x: '_'.join(x.split('_')[:-1])).nunique()} "
        f"| val: {val_df['global_sentence_id'].apply(lambda x: '_'.join(x.split('_')[:-1])).nunique()} "
        f"| test: {test_df['global_sentence_id'].apply(lambda x: '_'.join(x.split('_')[:-1])).nunique()}",
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


def build_char_vocab(dataframe: pd.DataFrame) -> dict:
    char_vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    for char in string.printable:
        if char not in ["\n", "\t", "\r", "\x0b", "\x0c"]:
            if char not in char_vocab:
                char_vocab[char] = idx
                idx += 1

    for token in dataframe["token"].astype(str):
        for char in token:
            if char not in char_vocab:
                char_vocab[char] = idx
                idx += 1

    return char_vocab


def check_vocab_coverage(dataframe: pd.DataFrame, char_vocab: dict) -> None:
    total_chars_in_data = 0
    covered_chars = 0
    missing_chars_freq = {}

    for token in dataframe["token"].astype(str):
        for char in token:
            total_chars_in_data += 1

            if char in char_vocab:
                covered_chars += 1
            else:
                if char not in missing_chars_freq:
                    missing_chars_freq[char] = 1
                else:
                    missing_chars_freq[char] += 1

    coverage_percent = (covered_chars / total_chars_in_data) * 100

    log(f"Coverage Vocab: {coverage_percent:.4f}%", level=log_level.INFO)

    if len(missing_chars_freq) > 0:
        sorted_missing = sorted(
            missing_chars_freq.items(), key=lambda x: x[1], reverse=True
        )

        log(f"OOV characters: {len(missing_chars_freq)}", level=log_level.WARNING)

        for char, freq in sorted_missing[:10]:  # Tampilkan hanya top 10
            log(f"- {repr(char)}: {freq}x", level=log_level.WARNING)
