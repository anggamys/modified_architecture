import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from feature_extraction import CharCNN
from preprocess import prepare_char_ids
from main import main


class POS_Dataset(Dataset):
    def __init__(self, dataframe, char_vocab, class_to_idx, max_word_len=50):
        self.tokens = dataframe["token"].astype(str).values
        self.labels = dataframe["pos_tag"].values
        self.char_vocab = char_vocab
        self.class_to_idx = class_to_idx
        self.max_word_len = max_word_len

        self.char_ids = prepare_char_ids(
            self.tokens, self.char_vocab, max_word_len=self.max_word_len
        )

        # Convert labels to indices
        self.label_indices = torch.tensor(
            [self.class_to_idx[label] for label in self.labels], dtype=torch.long
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        char_ids = torch.tensor(self.char_ids[idx], dtype=torch.long)
        label = self.label_indices[idx]

        return {"char_ids": char_ids, "label": label, "token": self.tokens[idx]}


class POS_Tagger_Simple(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        char_embedding_dim=32,
        char_filters=96,
        output_dim=128,
    ):
        super().__init__()

        self.char_cnn = CharCNN(
            vocab_size=vocab_size,
            emb_dim=char_embedding_dim,
            num_filters=char_filters,
            kernel_sizes=(2, 3, 4, 5),
            output_dim=output_dim,
            dropout=0.3,
        )

        # Classifier: dari CharCNN output -> num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, char_ids):
        # CharCNN expect (batch, seq_len, max_word_len)
        # tapi kita punya single tokens, jadi unsqueeze untuk sequence dim
        char_ids = char_ids.unsqueeze(1)  # (B, 1, W)

        features = self.char_cnn(char_ids)  # (B, 1, output_dim)
        features = features.squeeze(1)  # (B, output_dim)

        logits = self.classifier(features)  # (B, num_classes)

        return logits


class Trainer:
    def __init__(self, model, device, weight_tensor=None, lr=1e-3):
        self.model = model
        self.device = device

        # Loss function dengan weighted loss untuk imbalance
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(device)
            self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            char_ids = batch["char_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            logits = self.model(char_ids)
            loss = self.loss_fn(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader, idx_to_class):
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                char_ids = batch["char_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(char_ids)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        # Weighted average (weighted by class frequency)
        class_counts = pd.Series(all_labels).value_counts()
        weighted_f1 = (f1 * class_counts[np.arange(len(f1))]) / len(all_labels)  # type: ignore
        macro_f1 = f1.mean()  # type: ignore

        metrics = {
            "loss": avg_loss,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": {idx_to_class[i]: f for i, f in enumerate(f1)},  # type: ignore
            "confusion_matrix": confusion_matrix(all_labels, all_preds),
        }

        return metrics

    def fit(self, train_loader, val_loader, idx_to_class, epochs=20, patience=5):
        best_f1 = -1
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Train loss: {train_loss:.6f}")

            # Validate
            val_metrics = self.evaluate(val_loader, idx_to_class)
            print(f"Val loss: {val_metrics['loss']:.6f}")
            print(f"Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"Weighted F1: {val_metrics['weighted_f1']:.4f}")
            print("\nPer-class F1:")
            for cls, f1 in val_metrics["per_class_f1"].items():
                print(f"  {cls}: {f1:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics["weighted_f1"])

            # Early stopping
            if val_metrics["weighted_f1"] > best_f1:
                best_f1 = val_metrics["weighted_f1"]
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pt")
                print(f"✓ Best model saved (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping after {epoch + 1} epochs")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pt"))
        return best_f1


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


def main_training_example():
    # 1. Prepare data
    result = main(
        data_path="./data/sample-anotasi-merge-valid.csv",
        model_name="indobenchmark/indobert-base-p1",
    )

    train_df = result["train_df"]
    val_df = result["val_df"]
    test_df = result["test_df"]
    char_vocab = result["char_vocab"]
    weight_tensor = result["weight_tensor"]
    class_to_idx = result["class_to_idx"]
    idx_to_class = result["idx_to_class"]
    device = result["device"]

    # 2. Create datasets
    train_dataset = POS_Dataset(train_df, char_vocab, class_to_idx)
    val_dataset = POS_Dataset(val_df, char_vocab, class_to_idx)
    test_dataset = POS_Dataset(test_df, char_vocab, class_to_idx)

    # 3. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Create model
    model = POS_Tagger_Simple(
        vocab_size=len(char_vocab), num_classes=len(class_to_idx)
    ).to(device)

    # 5. Train
    trainer = Trainer(
        model=model,
        device=device,
        weight_tensor=weight_tensor,  # ← IMPORTANT: Weighted loss!
        lr=1e-3,
    )

    best_f1 = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        idx_to_class=idx_to_class,
        epochs=20,
        patience=5,
    )

    # 6. Test
    print(f"\n{'=' * 60}")
    print("TESTING")
    print(f"{'=' * 60}")
    test_metrics = trainer.evaluate(test_loader, idx_to_class)
    print(f"Test loss: {test_metrics['loss']:.6f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in test_metrics["per_class_f1"].items():
        print(f"  {cls}: {f1:.4f}")

    return trainer, best_f1


if __name__ == "__main__":
    trainer, best_f1 = main_training_example()
    print(f"\n✓ Training completed! Best F1: {best_f1:.4f}")
