"""
End-to-end data pipeline orchestration.
Coordinates all data processing steps dengan clean separation of concerns.
"""

import pandas as pd
from typing import Tuple

from core import (
    DataPipeline,
    DataPreprocessor,
    DataSplit,
    CharVocabulary,
    ClassMapping,
    ClassWeights,
)
from core.preprocess import DefaultDataPreprocessor, StratifiedDataSplitter
from core.class_weights import ClassWeightCalculator


class PosDataPipeline(DataPipeline):
    """POS tagging data pipeline"""

    def __init__(
        self,
        preprocessor: DataPreprocessor | None = None,
        splitter: StratifiedDataSplitter | None = None,
        weight_calculator: ClassWeightCalculator | None = None,
    ):
        self.preprocessor = preprocessor or DefaultDataPreprocessor()
        self.splitter = splitter or StratifiedDataSplitter(random_seed=42)
        self.weight_calculator = weight_calculator or ClassWeightCalculator()

    def load_data(self, path: str) -> pd.DataFrame:
        """Load raw data dari CSV"""
        return pd.read_csv(path)

    def prepare(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_char_freq: int = 5,
        include_emoji: bool = False,
    ) -> Tuple[DataSplit, DataSplit, DataSplit, CharVocabulary, ClassMapping]:
        """
        Execute full preprocessing pipeline.

        Returns:
            (train_split, val_split, test_split, char_vocab, class_mapping)
        """

        # 1. Build character vocabulary
        texts = df["token"].astype(str).tolist()
        char_vocab = self.preprocessor.build_char_vocabulary(
            texts,
            min_freq=min_char_freq,
        )

        # 2. Build class mapping
        unique_classes = sorted(df["pos_tag"].unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

        class_dist = self.preprocessor.calculate_class_distribution(df, "pos_tag")

        class_mapping = ClassMapping(
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            num_classes=len(class_to_idx),
            class_distribution=class_dist,
        )

        # 3. Split data dengan stratifikasi
        train_split, val_split, test_split = self.splitter.split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        return train_split, val_split, test_split, char_vocab, class_mapping

    def calculate_class_weights(
        self,
        train_split: DataSplit,
        class_mapping: ClassMapping,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
    ) -> ClassWeights:
        """Calculate class weights untuk imbalanced learning"""

        return self.weight_calculator.calculate(
            train_split,
            class_mapping,
            min_weight=min_weight,
            max_weight=max_weight,
        )
