import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from preprocess import (
    build_char_vocab,
    check_vocab_coverage,
    class_distribution,
    split_train_val_test,
    calculate_class_weights,
    create_torch_weight_tensor,
)

from train import train_model, evaluate, compute_accuracy
from feature_extraction import HybridModel
from dataset import POSDataset, make_collate_fn
from utils import dataInfo, log, log_level, argParser, dowloadModel


def main(data_path: str, model_name: str, epochs: int = 10) -> None:
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

    # calculate_class_weights: inverse frequency, dinormalisasi ke [0.5, 2.0]
    # create_torch_weight_tensor: mapping class name → tensor index
    class_weights_dict = calculate_class_weights(train_df["pos_tag"])
    class_weights = create_torch_weight_tensor(class_weights_dict, class_to_idx).to(
        device
    )

    log(
        "Class weights (top 5 heaviest): "
        + " | ".join(
            f"{idx_to_class[i]}: {class_weights[i].item():.3f}"
            for i in class_weights.argsort(descending=True)[:5].tolist()
        ),
        level=log_level.INFO,
    )

    model_path: str = dowloadModel(model_name)

    # test_feature_extraction(
    #     sample_data=train_df,
    #     char_vocab=char_vocab,
    #     model_name=model_name,
    #     batch_size=32,
    #     device=device,
    # )

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

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)

    train_dataset = POSDataset(train_df, char_vocab, class_to_idx, tokenizer)
    val_dataset = POSDataset(val_df, char_vocab, class_to_idx, tokenizer)
    test_dataset = POSDataset(test_df, char_vocab, class_to_idx, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    log(
        f"DataLoader → train: {len(train_loader)} batch | "
        f"val: {len(val_loader)} batch | test: {len(test_loader)} batch",
        level=log_level.INFO,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        checkpoint_path="best_model.pt",
    )

    # ── Final evaluation pada test set (hanya sekali, setelah training selesai) ──
    # Load checkpoint terbaik sebelum evaluasi agar tidak pakai model epoch terakhir
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    log("Loaded best checkpoint untuk final evaluation.", level=log_level.INFO)

    test_loss, test_preds, test_labels = evaluate(model, test_loader, device)
    test_acc = compute_accuracy(test_preds, test_labels)

    log(f"[Test] Loss     : {test_loss:.4f}", level=log_level.INFO)
    log(f"[Test] Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)", level=log_level.INFO)


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
            {
                "flag": "--epochs",
                "type": int,
                "default": 10,
                "help": "Number of training epochs.",
                "required": False,
            },
        ],
    ).parse_args()

    main(
        data_path=args.data_path,
        model_name=args.model_name,
        epochs=args.epochs,
    )
