"""
Core domain abstractions dan models.
Layer ini berisi business logic yang independent dari framework atau implementation details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import pandas as pd
import torch
from pathlib import Path


@dataclass
class DataSplit:
    """Immutable representation dari data split"""

    dataframe: pd.DataFrame
    name: str  # "train", "val", "test"
    size: int

    def __len__(self) -> int:
        return len(self.dataframe)


@dataclass
class CharVocabulary:
    """Character vocabulary dengan metadata"""

    char_to_idx: Dict[str, int]
    idx_to_char: Dict[int, str]
    vocab_size: int
    coverage: float  # persentase coverage di dataset

    @property
    def pad_token_id(self) -> int:
        return 0  # Reserved untuk padding

    def __len__(self) -> int:
        return self.vocab_size


@dataclass
class ClassMapping:
    """POS tag class mapping dengan metadata"""

    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    num_classes: int
    class_distribution: Dict[str, float]  # class → frequency ratio

    def __len__(self) -> int:
        return self.num_classes


@dataclass
class ClassWeights:
    """Calculated class weights untuk imbalanced learning"""

    weights: torch.Tensor  # (num_classes,)
    class_names: List[str]
    method: str  # "inverse_frequency", "effective_number", etc

    def __len__(self) -> int:
        return len(self.weights)

    def get_weight(self, class_idx: int) -> float:
        return self.weights[class_idx].item()


@dataclass
class ModelMetadata:
    """Metadata tentang trained model untuk loading & inference"""

    model_path: Path
    char_vocab: CharVocabulary
    class_mapping: ClassMapping
    training_config: Dict[str, Any]
    metrics: Dict[str, float]  # test_acc, test_loss, etc
    timestamp: str  # when model was saved


# ============================================================================
# Abstract interfaces untuk Dependency Injection
# ============================================================================


class DataPreprocessor(ABC):
    """Interface untuk preprocessing pipeline"""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text"""
        pass

    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean text dari artifacts"""
        pass

    @abstractmethod
    def build_char_vocabulary(
        self, texts: List[str], min_freq: int = 5
    ) -> CharVocabulary:
        """Build character vocabulary dari texts"""
        pass

    @abstractmethod
    def calculate_class_distribution(
        self, df: pd.DataFrame, class_column: str
    ) -> Dict[str, float]:
        """Calculate class distribution dari dataset"""
        pass


class DataSplitter(ABC):
    """Interface untuk stratified data splitting"""

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int = 42,
    ) -> Tuple[DataSplit, DataSplit, DataSplit]:
        """Split data dengan stratifikasi"""
        pass


class DataPipeline(ABC):
    """Interface untuk end-to-end data processing"""

    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame:
        """Load raw data"""
        pass

    @abstractmethod
    def prepare(
        self, df: pd.DataFrame
    ) -> Tuple[DataSplit, DataSplit, DataSplit, CharVocabulary, ClassMapping]:
        """Preprocess & split data"""
        pass

    @abstractmethod
    def calculate_class_weights(
        self, train_split: DataSplit, class_mapping: ClassMapping
    ) -> ClassWeights:
        """Calculate class weights untuk imbalanced learning"""
        pass


class ModelBuilder(ABC):
    """Factory interface untuk model creation"""

    @abstractmethod
    def build(
        self,
        vocab_size: int,
        num_classes: int,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """Build model dengan given configuration"""
        pass


class ModelTrainer(ABC):
    """Interface untuk training loop"""

    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Execute training loop, return results"""
        pass

    @abstractmethod
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Evaluate model, return metrics"""
        pass


class ModelRepository(ABC):
    """Interface untuk model persistence"""

    @abstractmethod
    def save(self, model: torch.nn.Module, metadata: ModelMetadata) -> Path:
        """Save model & metadata"""
        pass

    @abstractmethod
    def load(self, model_path: Path) -> Tuple[torch.nn.Module, ModelMetadata]:
        """Load model & metadata"""
        pass
