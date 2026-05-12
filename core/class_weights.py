"""
Class weights calculation untuk imbalanced learning.
Terisolasi dari data processing logic untuk clarity.
"""

import pandas as pd
import torch
from typing import Dict

from core import ClassWeights, ClassMapping, DataSplit


class ClassWeightCalculator:
    """Calculate class weights untuk imbalanced dataset"""

    def __init__(self, method: str = "inverse_frequency"):
        """
        Args:
            method: "inverse_frequency" (default) atau "effective_number"
        """
        self.method = method

    def calculate(
        self,
        train_split: DataSplit,
        class_mapping: ClassMapping,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
    ) -> ClassWeights:
        """
        Calculate normalized class weights.

        Args:
            train_split: Training data
            class_mapping: Class mapping object
            min_weight: Minimum weight bound (untuk rarest classes)
            max_weight: Maximum weight bound (untuk most frequent classes)

        Returns:
            ClassWeights object
        """

        if self.method == "inverse_frequency":
            raw_weights = self._inverse_frequency(
                train_split.dataframe,
                class_mapping,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Normalize ke range [min_weight, max_weight]
        normalized = self._normalize_range(
            raw_weights,
            min_val=min_weight,
            max_val=max_weight,
        )

        # Create torch tensor
        weights_tensor = torch.tensor(
            [normalized[cls] for cls in class_mapping.idx_to_class.values()],
            dtype=torch.float32,
        )

        class_names = [
            class_mapping.idx_to_class[i]
            for i in range(len(class_mapping.idx_to_class))
        ]

        return ClassWeights(
            weights=weights_tensor,
            class_names=class_names,
            method=self.method,
        )

    @staticmethod
    def _inverse_frequency(
        df: pd.DataFrame,
        class_mapping: ClassMapping,
    ) -> Dict[str, float]:
        """Calculate inverse frequency weights"""

        class_counts = df["pos_tag"].value_counts()
        total_samples = len(df)

        weights = {}
        for class_name in class_mapping.class_to_idx.keys():
            count = class_counts.get(class_name, 1)
            frequency = count / total_samples
            # Inverse frequency: rare classes get higher weight
            weight = 1.0 / frequency if frequency > 0 else 1.0
            weights[class_name] = weight

        return weights

    @staticmethod
    def _normalize_range(
        weights: Dict[str, float],
        min_val: float = 0.5,
        max_val: float = 2.0,
    ) -> Dict[str, float]:
        """Normalize weights ke range [min_val, max_val]"""

        values = list(weights.values())
        min_weight = min(values)
        max_weight = max(values)

        if max_weight == min_weight:
            # All weights are equal, assign uniform
            return {k: (min_val + max_val) / 2 for k in weights.keys()}

        normalized = {}
        for cls, weight in weights.items():
            # Linear scaling: [min_weight, max_weight] → [min_val, max_val]
            scaled = min_val + (weight - min_weight) / (max_weight - min_weight) * (
                max_val - min_val
            )
            normalized[cls] = scaled

        return normalized
