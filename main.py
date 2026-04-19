import pandas as pd
import torch
from preprocess import (
    build_char_vocab,
    check_vocab_coverage,
    class_distribution,
    split_train_val_test,
)

from feature_extraction import CharCNN, Bert
from utils import dataInfo, log, log_level, argParser, dowloadModel


def main(data_path: str, model_name: str) -> None:
    sample_data = pd.read_csv(data_path)
    dataInfo(sample_data)

    class_distribution(sample_data, "pos_tag")

    char_vocab = build_char_vocab(sample_data, min_freq=5, include_emoji=False)

    check_vocab_coverage(sample_data, char_vocab)

    train_df, val_df, test_df = split_train_val_test(
        sample_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    log(
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
        level=log_level.INFO,
    )

    char_extraction = CharCNN(vocab_size=len(char_vocab))

    log("Feature extraction model initialized.", level=log_level.INFO)

    char_extraction.to("cuda" if torch.cuda.is_available() else "cpu")

    model_path = dowloadModel(model_name)

    bert_extraction = Bert(model_name=model_path)

    log("BERT model initialized.", level=log_level.INFO)

    bert_extraction.to("cuda" if torch.cuda.is_available() else "cpu")


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
