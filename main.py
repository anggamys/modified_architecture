import pandas as pd
import torch
from transformers import AutoModel

from preprocess import (
    build_char_vocab,
    check_vocab_coverage,
    class_distribution,
    split_train_val_test,
    calculate_class_weights,
    create_torch_weight_tensor,
)

from utils import dataInfo, log, log_level, argParser, dowloadModel
from testing import test_feature_extraction
# from train import train_model
from feature_extraction import HybridModel


def main(data_path: str, model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}", level=log_level.INFO)

    sample_data = pd.read_csv(data_path)
    dataInfo(sample_data)

    class_distribution(sample_data, "pos_tag")

    # Buat mapping class → index untuk torch tensor
    unique_classes = sorted(sample_data["pos_tag"].unique())
    class_to_idx: dict[str, int] = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class: dict[int, str] = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    log(f"Class to index mapping: {class_to_idx}", level=log_level.INFO)

    char_vocab = build_char_vocab(sample_data, min_freq=5, include_emoji=False)

    check_vocab_coverage(sample_data, char_vocab)

    train_df, val_df, test_df = split_train_val_test(
        sample_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    log(
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
        level=log_level.INFO,
    )

    # Verify label distribution balanced di semua split
    log("Label distribution setelah split:", level=log_level.INFO)
    for split_name, split_df in [
        ("Train", train_df),
        ("Val", val_df),
        ("Test", test_df),
    ]:
        dist = split_df["pos_tag"].value_counts(normalize=True)
        dist_str = " | ".join(f"{cls}: {pct:.4f}" for cls, pct in dist.items())
        log(f"{split_name}: {dist_str}", level=log_level.INFO)

    # ── Class weights (dari train split, via preprocess.py) ──────────────
    # calculate_class_weights: inverse frequency, dinormalisasi ke [0.5, 2.0]
    # create_torch_weight_tensor: mapping class name → tensor index
    class_weights_dict = calculate_class_weights(train_df["pos_tag"])
    class_weights = create_torch_weight_tensor(class_weights_dict, class_to_idx).to(device)

    log(
        "Class weights (top 5 heaviest): "
        + " | ".join(
            f"{idx_to_class[i]}: {class_weights[i].item():.3f}"
            for i in class_weights.argsort(descending=True)[:5].tolist()
        ),
        level=log_level.INFO,
    )

    # ── Sanity check: feature extraction ──────────────────────────────────
    model_path: str = dowloadModel(model_name)

    test_feature_extraction(
        sample_data=train_df,
        char_vocab=char_vocab,
        model_name=model_name,
        batch_size=32,
        device=device,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    bert_model = AutoModel.from_pretrained(model_path)

    model = HybridModel(
        char_vocab_size=len(char_vocab),
        bert=bert_model,
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    log(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,} total | "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable",
        level=log_level.INFO,
    )

    # ── DataLoaders ────────────────────────────────────────────────────────
    # TODO: implementasi POSDataset & collate_fn untuk membungkus train_df / val_df
    # ke dalam DataLoader. Setelah itu panggil train_model().
    #
    # Contoh:
    #   train_loader = DataLoader(
    #       POSDataset(train_df, char_vocab, class_to_idx, tokenizer),
    #       batch_size=16, shuffle=True, collate_fn=pos_collate_fn
    #   )
    #   val_loader = DataLoader(...)
    #
    #   train_model(model, train_loader, val_loader, device, epochs=10)


if __name__ == "__main__":
    args = argParser(
        description="Run the main function with the specified data path.",
        args=[
            {
                "flag": "--data_path",
                "type": str,
                "default": "./data/sample-anotasi-merge-valid.csv",
                "help": "Path to the input CSV data file.",
                "required": False,
            },
            {
                "flag": "--model_name",
                "type": str,
                "default": "indobenchmark/indobert-base-p1",
                "help": "Name of the Hugging Face model to download.",
                "required": False,
            },
        ],
    ).parse_args()

    main(data_path=args.data_path, model_name=args.model_name)
