"""
Application layer - Orchestration of domain logic.
Coordinates multiple domain services untuk achieve use cases.
"""

from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import AppConfig
from core import (
    CharVocabulary,
    ClassMapping,
    DataSplit,
    ClassWeights,
)
from core.data_pipeline import PosDataPipeline
from dataset import POSDataset, make_collate_fn
from feature_extraction import HybridModel
from train import (
    train_model,
    evaluate_with_tokens,
    compute_classification_report,
)


class DataService:
    """Service untuk data loading dan preprocessing"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = PosDataPipeline()

    def prepare_data(
        self,
    ) -> Tuple[
        DataSplit, DataSplit, DataSplit, CharVocabulary, ClassMapping, ClassWeights
    ]:
        """Execute full data pipeline"""

        # Load raw data
        df = self.pipeline.load_data(self.config.data.data_path)

        # Preprocess & split
        train_split, val_split, test_split, char_vocab, class_mapping = (
            self.pipeline.prepare(
                df,
                train_ratio=self.config.data.train_ratio,
                val_ratio=self.config.data.val_ratio,
                test_ratio=self.config.data.test_ratio,
                min_char_freq=self.config.data.min_char_freq,
                include_emoji=self.config.data.include_emoji,
            )
        )

        # Calculate class weights
        class_weights = self.pipeline.calculate_class_weights(
            train_split,
            class_mapping,
            min_weight=self.config.model.class_weight_min,
            max_weight=self.config.model.class_weight_max,
        )

        return (
            train_split,
            val_split,
            test_split,
            char_vocab,
            class_mapping,
            class_weights,
        )


class DataLoaderService:
    """Service untuk PyTorch DataLoader creation"""

    def __init__(self, config: AppConfig):
        self.config = config

    def create_loaders(
        self,
        train_split: DataSplit,
        val_split: DataSplit,
        test_split: DataSplit,
        char_vocab: CharVocabulary,
        class_mapping: ClassMapping,
        class_weights: ClassWeights,
        tokenizer,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders dengan WeightedRandomSampler untuk training"""

        # Create datasets
        train_dataset = POSDataset(
            train_split.dataframe,
            char_vocab.char_to_idx,
            class_mapping.class_to_idx,
            tokenizer,
            max_word_len=self.config.data.max_word_len,
            max_seq_len=self.config.data.max_seq_len,
        )

        val_dataset = POSDataset(
            val_split.dataframe,
            char_vocab.char_to_idx,
            class_mapping.class_to_idx,
            tokenizer,
            max_word_len=self.config.data.max_word_len,
            max_seq_len=self.config.data.max_seq_len,
        )

        test_dataset = POSDataset(
            test_split.dataframe,
            char_vocab.char_to_idx,
            class_mapping.class_to_idx,
            tokenizer,
            max_word_len=self.config.data.max_word_len,
            max_seq_len=self.config.data.max_seq_len,
        )

        # Weighted sampler untuk train (to handle class imbalance)
        train_labels = train_split.dataframe["pos_tag"].map(class_mapping.class_to_idx)
        sample_weights = [class_weights.get_weight(label) for label in train_labels]

        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=weighted_sampler,
            collate_fn=make_collate_fn(tokenizer.pad_token_id),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer.pad_token_id),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer.pad_token_id),
        )

        return train_loader, val_loader, test_loader


class ModelService:
    """Service untuk model creation dan training"""

    def __init__(self, config: AppConfig):
        self.config = config

    def build_model(
        self,
        vocab_size: int,
        num_classes: int,
        device: str,
    ) -> HybridModel:
        """Build POS tagging model"""

        from transformers import AutoModel

        bert = AutoModel.from_pretrained(self.config.model.bert_model)

        model = HybridModel(
            char_vocab_size=vocab_size,
            bert=bert,
            num_classes=num_classes,
            use_crf=self.config.model.use_crf,
            use_word_bilstm=self.config.model.use_word_bilstm,
        )

        return model.to(device)

    def train(
        self,
        model: HybridModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: ClassWeights,
        device: str,
    ) -> Dict[str, Any]:
        """Train model"""

        train_model(
            checkpoint_path=str(self.config.output.model_dir),
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=self.config.training.epochs,
            patience=self.config.training.patience,
        )

        return {}

    def evaluate(
        self,
        model: HybridModel,
        test_loader: DataLoader,
        test_dataset: POSDataset,
        class_mapping: ClassMapping,
        device: str,
    ) -> Dict[str, Any]:
        """Evaluate model on test set"""

        # Get predictions
        result = evaluate_with_tokens(
            model, test_loader, dataset=test_dataset, device=device
        )
        predictions = result[1] if isinstance(result, tuple) else result

        # Extract tokens from dataset
        all_tokens = []
        all_labels = []

        for sent_df in test_dataset.sentences:
            tokens = sent_df["token"].astype(str).tolist()
            labels = [
                class_mapping.class_to_idx.get(tag, 0) for tag in sent_df["pos_tag"]
            ]
            all_tokens.extend(tokens)
            all_labels.extend(labels)

        # Compute metrics
        accuracy = sum(p == label for p, label in zip(predictions, all_labels)) / len(
            all_labels
        )

        # Classification report
        predictions_int = [int(p) if isinstance(p, str) else p for p in predictions]
        class_report = compute_classification_report(
            all_labels,
            predictions_int,
            class_mapping.idx_to_class,
        )

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "labels": all_labels,
            "tokens": all_tokens,
            "classification_report": class_report,
        }


class ApplicationOrchestrator:
    """Main application orchestrator - coordinates all services"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.config.setup()  # Create required directories

        self.data_service = DataService(config)
        self.model_service = ModelService(config)

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute complete training & evaluation pipeline"""

        # Step 1: Data preparation
        train_split, val_split, test_split, char_vocab, class_mapping, class_weights = (
            self.data_service.prepare_data()
        )

        # Step 2: Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.bert_model)

        # Step 3: Create data loaders
        loader_service = DataLoaderService(self.config)
        train_loader, val_loader, test_loader = loader_service.create_loaders(
            train_split,
            val_split,
            test_split,
            char_vocab,
            class_mapping,
            class_weights,
            tokenizer,
        )

        # Step 4: Build model
        device = self.config.training.device or "cpu"
        model = self.model_service.build_model(
            vocab_size=char_vocab.vocab_size,
            num_classes=class_mapping.num_classes,
            device=device,
        )

        # Step 5: Train
        training_results = self.model_service.train(
            model,
            train_loader,
            val_loader,
            class_weights,
            device,
        )

        # Step 6: Evaluate
        test_dataset = POSDataset(
            test_split.dataframe,
            char_vocab.char_to_idx,
            class_mapping.class_to_idx,
            tokenizer,
            max_word_len=self.config.data.max_word_len,
            max_seq_len=self.config.data.max_seq_len,
        )

        eval_results = self.model_service.evaluate(
            model,
            test_loader,
            test_dataset,
            class_mapping,
            device,
        )

        return {
            "training": training_results,
            "evaluation": eval_results,
            "metadata": {
                "char_vocab": char_vocab,
                "class_mapping": class_mapping,
                "class_weights": class_weights,
            },
        }
