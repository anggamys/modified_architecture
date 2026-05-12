"""
Data preprocessing implementations.
Refactored dari existing preprocess.py dengan better separation of concerns.
"""

import unicodedata
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

from core import DataPreprocessor, CharVocabulary, DataSplit


class DefaultDataPreprocessor(DataPreprocessor):
    """Standard text normalization dan cleaning"""

    def normalize(self, text: str) -> str:
        """Normalize text dengan NFKC unicode form"""
        text = str(text)

        # Unify quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")

        # Normalize unicode form
        text = unicodedata.normalize("NFKC", text)

        return text

    def clean(self, text: str) -> str:
        """Remove control characters"""
        return "".join(c for c in text if unicodedata.category(c)[0] != "C")

    def build_char_vocabulary(
        self,
        texts: List[str],
        min_freq: int = 5,
        include_emoji: bool = False,
    ) -> CharVocabulary:
        """
        Build character vocabulary dari list of texts.

        Args:
            texts: List of text strings
            min_freq: Minimum frequency threshold untuk include character
            include_emoji: Whether to include emoji characters

        Returns:
            CharVocabulary object
        """
        char_counter = Counter()
        total_chars = 0
        coverage_chars = 0

        for text in texts:
            text = self.normalize(text)
            for char in text:
                char_counter[char] += 1
                total_chars += 1
                if char_counter[char] >= min_freq:
                    coverage_chars = sum(
                        v for v in char_counter.values() if v >= min_freq
                    )

        # Build vocab (reserve 0 untuk padding)
        char_to_idx = {"<PAD>": 0}  # index 0 reserved
        for idx, (char, freq) in enumerate(char_counter.most_common(), start=1):
            if freq >= min_freq:
                char_to_idx[char] = idx

        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        vocab_size = len(char_to_idx)
        coverage = (coverage_chars / total_chars * 100) if total_chars > 0 else 0.0

        return CharVocabulary(
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            vocab_size=vocab_size,
            coverage=coverage,
        )

    def calculate_class_distribution(
        self,
        df: pd.DataFrame,
        class_column: str,
    ) -> Dict[str, float]:
        """Calculate class distribution sebagai ratio"""
        value_counts = df[class_column].value_counts()
        total = len(df)
        return {str(cls): count / total for cls, count in value_counts.items()}


class StratifiedDataSplitter:
    """Data splitter dengan stratifikasi"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def split(
        self,
        dataframe: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[DataSplit, DataSplit, DataSplit]:
        """
        Split data dengan stratifikasi berdasarkan:
        - Document (group splitting)
        - Mean Length Utterance (MLU) untuk complexity stratification
        """

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        df = dataframe.copy()
        df["_doc_id"] = df["global_sentence_id"].apply(
            lambda x: "_".join(x.split("_")[:-1])
        )

        # MLU stratification
        df["_token_len"] = df["token"].astype(str).str.len()
        mlu_per_doc = df.groupby("_doc_id")["_token_len"].mean()
        df["_mlu_bucket"] = (
            df["_doc_id"]
            .map(mlu_per_doc)
            .apply(lambda x: "short" if x < 5 else ("medium" if x < 8 else "long"))
        )

        df["_strat_key"] = df["pos_tag"] + "_" + df["_mlu_bucket"]
        groups = df["_doc_id"]

        # First split: train vs (val + test)
        gss_train = GroupShuffleSplit(
            n_splits=1,
            train_size=train_ratio,
            random_state=self.random_seed,
        )
        train_idx, temp_idx = next(
            gss_train.split(X=df, y=df["_strat_key"], groups=groups)
        )

        train_df = (
            df.iloc[train_idx]
            .drop(columns=["_doc_id", "_token_len", "_mlu_bucket", "_strat_key"])
            .reset_index(drop=True)
        )
        df_temp = df.iloc[temp_idx]

        # Second split: val vs test
        val_size_relative = val_ratio / (val_ratio + test_ratio)
        gss_val = GroupShuffleSplit(
            n_splits=1,
            train_size=val_size_relative,
            random_state=self.random_seed,
        )
        val_idx, test_idx = next(
            gss_val.split(X=df_temp, y=df_temp["_strat_key"], groups=df_temp["_doc_id"])
        )

        val_df = (
            df_temp.iloc[val_idx]
            .drop(columns=["_doc_id", "_token_len", "_mlu_bucket", "_strat_key"])
            .reset_index(drop=True)
        )
        test_df = (
            df_temp.iloc[test_idx]
            .drop(columns=["_doc_id", "_token_len", "_mlu_bucket", "_strat_key"])
            .reset_index(drop=True)
        )

        return (
            DataSplit(train_df, "train", len(train_df)),
            DataSplit(val_df, "val", len(val_df)),
            DataSplit(test_df, "test", len(test_df)),
        )


def prepare_char_ids(
    tokens: List[str],
    char_vocab: Dict[str, int],
    max_word_len: int = 50,
    pad_token_id: int = 0,
) -> np.ndarray:
    """
    Prepare character IDs untuk list of tokens.

    Returns:
        numpy array (num_tokens, max_word_len)
    """
    char_ids = []

    for token in tokens:
        token_str = str(token)
        ids = [char_vocab.get(c, pad_token_id) for c in token_str]

        # Pad or truncate
        if len(ids) < max_word_len:
            ids = ids + [pad_token_id] * (max_word_len - len(ids))
        else:
            ids = ids[:max_word_len]

        char_ids.append(ids)

    return np.array(char_ids, dtype=np.int64)
