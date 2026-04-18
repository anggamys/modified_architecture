import pandas as pd
from preprocess import (
    build_char_vocab,
    check_vocab_coverage,
    class_distribution,
    split_train_val_test,
)
from utils import dataInfo, log, log_level


def main():
    sample_data = pd.read_csv("./data/sample-anotasi-merge-valid.csv")
    dataInfo(sample_data)

    class_distribution(sample_data, "pos_tag")

    char_vocab = build_char_vocab(sample_data)
 
    check_vocab_coverage(sample_data, char_vocab)

    train_df, val_df, test_df = split_train_val_test(
        sample_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    log(
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
        level=log_level.INFO,
    )

if __name__ == "__main__":
    main()
