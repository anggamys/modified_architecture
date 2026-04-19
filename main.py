import pandas as pd
import torch
from preprocess import (
    build_char_vocab,
    check_vocab_coverage,
    class_distribution,
    split_train_val_test,
    # calculate_class_weights,
    # create_torch_weight_tensor,
)

from utils import dataInfo, log, log_level, argParser
from testing import test_feature_extraction


def main(data_path: str, model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}", level=log_level.INFO)

    sample_data = pd.read_csv(data_path)
    dataInfo(sample_data)

    class_distribution(sample_data, "pos_tag")
    # class_weights = calculate_class_weights(sample_data["pos_tag"])

    # Buat mapping class -> index untuk torch tensor
    unique_classes = sorted(sample_data["pos_tag"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    # idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

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

        log(f"  {split_name}: {dist_str}", level=log_level.INFO)

    # Test feature extraction
    test_feature_extraction(
        sample_data=train_df,
        char_vocab=char_vocab,
        model_name=model_name,
        batch_size=32,
        device=device,
    )

    # weight_tensor = create_torch_weight_tensor(class_weights, class_to_idx)

    # log(
    #     f"Weight tensor untuk nn.CrossEntropyLoss:\n{weight_tensor}",
    #     level=log_level.INFO,
    # )

    # log(
    #     "Usage: loss_fn = nn.CrossEntropyLoss(weight=weight_tensor.to(device))",
    #     level=log_level.INFO,
    # )


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
