import json
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer

from dataset import POSDataset, make_collate_fn, get_all_labels_from_dataset
from feature_extraction import HybridModel
from preprocess import (
    build_char_vocab,
    calculate_class_weights,
    check_vocab_coverage,
    class_distribution,
    create_torch_weight_tensor,
    split_train_val_test,
)
from train import (
    compute_accuracy,
    compute_classification_report,
    evaluate_with_tokens,
    save_test_results,
    train_model,
)
from utils import argParser, dataInfo, dowloadModel, log, log_level


def main(
    data_path: str,
    model_name: str,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 16,
    char_type: str = "cnn",
    use_crf: bool = True,
    use_word_bilstm: bool = False,
    config_name: str = "",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(domain="Main", msg=f"Using device: {device}", level=log_level.INFO)

    # Log konfigurasi ablation yang aktif
    char_label = {"none": "(none)", "cnn": "Char-CNN", "bilstm": "Char-BiLSTM"}.get(
        char_type, char_type
    )
    word_label = "Word-BiLSTM" if use_word_bilstm else ""
    crf_label = "CRF" if use_crf else "Linear"

    parts = ["IndoBERT", char_label, word_label, crf_label]
    ablation_name = " + ".join([p for p in parts if p and p != "(none)"])

    # Setup Output Directory
    output_dir_name = config_name if config_name else "Custom"
    output_dir = os.path.join("outputs", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    log(
        domain="Main",
        msg=f"Konfigurasi Ablation: [{ablation_name}]",
        level=log_level.INFO,
    )
    log(domain="Main", msg=f"Direktori Output: {output_dir}/", level=log_level.INFO)

    sample_data = pd.read_csv(data_path)
    dataInfo(sample_data)

    class_distribution(sample_data, "pos_tag")

    # Buat mapping class → index untuk torch tensor
    unique_classes = sorted(sample_data["pos_tag"].unique())
    class_to_idx: dict[str, int] = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class: dict[int, str] = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    log(
        domain="Main",
        msg=f"Class to index mapping: {class_to_idx}",
        level=log_level.INFO,
    )

    char_vocab = build_char_vocab(sample_data, min_freq=5, include_emoji=False)

    check_vocab_coverage(sample_data, char_vocab)

    train_df, val_df, test_df = split_train_val_test(
        sample_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    log(
        domain="Main",
        msg=f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
        level=log_level.INFO,
    )

    # Verify label distribution balanced di semua split
    log(domain="Main", msg="Label distribution setelah split:", level=log_level.INFO)
    for split_name, split_df in [
        ("Train", train_df),
        ("Val", val_df),
        ("Test", test_df),
    ]:
        dist = split_df["pos_tag"].value_counts(normalize=True)
        dist_str = " | ".join(f"{cls}: {pct:.4f}" for cls, pct in dist.items())
        log(domain="Main", msg=f"{split_name}: {dist_str}", level=log_level.INFO)

    # calculate_class_weights: inverse frequency, dinormalisasi ke [0.5, 2.0]
    # create_torch_weight_tensor: mapping class name → tensor index
    class_weights_dict = calculate_class_weights(train_df["pos_tag"])
    class_weights = create_torch_weight_tensor(class_weights_dict, class_to_idx).to(
        device
    )

    log(
        domain="Main",
        msg="Class weights (top 5 heaviest): "
        + " | ".join(
            f"{idx_to_class[i]}: {class_weights[i].item():.3f}"
            for i in class_weights.argsort(descending=True)[:5].tolist()
        ),
        level=log_level.INFO,
    )

    model_path: str = dowloadModel(model_name)

    bert_model = AutoModel.from_pretrained(model_path)

    model = HybridModel(
        char_vocab_size=len(char_vocab),
        bert=bert_model,
        num_classes=num_classes,
        class_weights=class_weights,
        char_type=char_type,
        use_crf=use_crf,
        use_word_bilstm=use_word_bilstm,
    ).to(device)

    log(
        domain="Main",
        msg=f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,} total | "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable",
        level=log_level.INFO,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)

    train_dataset = POSDataset(train_df, char_vocab, class_to_idx, tokenizer)
    val_dataset = POSDataset(val_df, char_vocab, class_to_idx, tokenizer)
    test_dataset = POSDataset(test_df, char_vocab, class_to_idx, tokenizer)

    # Stratified batch sampling: compute sample weights from training labels
    train_labels = get_all_labels_from_dataset(train_dataset)
    class_counts = {}
    for label in train_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate weight for each sample: weight = 1 / class_frequency
    sample_weights = []
    for label in train_labels:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)
    
    # Create weighted sampler for balanced batch sampling
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use stratified sampler instead of shuffle
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    log(
        domain="Main",
        msg=f"DataLoader → train: {len(train_loader)} batch | val: {len(val_loader)} batch | test: {len(test_loader)} batch",
        level=log_level.INFO,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        patience=patience,
        checkpoint_path=os.path.join(
            output_dir, f"best_model_{config_name.lower()}.pt"
        ),
    )

    # Final evaluation pada test set dengan tracking tokens untuk confusion matrix analysis
    # train_model sudah me-restore best checkpoint sebelum return
    test_loss, test_tokens, test_preds, test_labels, sent_indices, token_indices = (
        evaluate_with_tokens(model, test_loader, test_dataset, device)
    )
    test_acc = compute_accuracy(test_preds, test_labels)

    log(domain="Main", msg=f"Test Loss     : {test_loss:.4f}", level=log_level.INFO)
    log(
        domain="Main",
        msg=f"Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)",
        level=log_level.INFO,
    )

    report_path = os.path.join(
        output_dir, f"classification_report_{config_name.lower()}.json"
    )
    compute_classification_report(
        test_preds, test_labels, idx_to_class, output_path=report_path
    )

    # Simpan hasil test (token, true label, pred label) untuk confusion matrix & error analysis
    test_results_path = os.path.join(output_dir, f"test_results_{config_name.lower()}")
    save_test_results(
        tokens=test_tokens,
        preds=test_preds,
        labels=test_labels,
        idx_to_class=idx_to_class,
        output_path=test_results_path,
        format_type="both",  # Simpan sebagai CSV dan JSON
        sent_indices=sent_indices,
        token_indices=token_indices,
    )

    # Simpan vocab & class mapping untuk dipakai oleh inference.py
    char_vocab_path = os.path.join(output_dir, f"char_vocab_{config_name.lower()}.json")
    with open(char_vocab_path, "w", encoding="utf-8") as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)

    class_mappings_path = os.path.join(
        output_dir, f"class_mappings_{config_name.lower()}.json"
    )
    with open(class_mappings_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if use_crf and hasattr(model, "crf"):
        transition_matrix = model.crf.transitions.detach().cpu().numpy()
        tag_list = [idx_to_class.get(i, f"TAG_{i}") for i in range(num_classes)]
        df_transitions = pd.DataFrame(
            transition_matrix, index=tag_list, columns=tag_list
        )
        transitions_csv_path = os.path.join(
            output_dir, f"crf_transitions_{config_name.lower()}.csv"
        )
        df_transitions.to_csv(transitions_csv_path)
        log(
            domain="Main",
            msg=f"Matriks transisi CRF berhasil disimpan ke: {transitions_csv_path}",
            level=log_level.INFO,
        )

    log(
        domain="Main",
        msg=f"Semua file (Model, Vocab, Class Mapping) berhasil disimpan di folder: {output_dir}/",
        level=log_level.INFO,
    )


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
                "default": 20,
                "help": "Maximum number of training epochs.",
                "required": False,
            },
            {
                "flag": "--patience",
                "type": int,
                "default": 3,
                "help": "Early stopping patience (epochs without val improvement).",
                "required": False,
            },
            {
                "flag": "--batch_size",
                "type": int,
                "default": 16,
                "help": "DataLoader batch size.",
                "required": False,
            },
            {
                "flag": "--char_type",
                "type": str,
                "default": "cnn",
                "help": "Tipe char extractor: 'none' (M1/M2), 'cnn' (M4/M6, default), 'bilstm' (M5).",
                "required": False,
            },
            {
                "flag": "--use_crf",
                "type": bool,
                "default": True,
                "help": "Aktifkan CRF decoder (False = baseline Linear/Softmax murni).",
                "required": False,
            },
            {
                "flag": "--use_word_bilstm",
                "type": bool,
                "default": False,
                "help": "Aktifkan Word-Level BiLSTM setelah agregasi/pooling subword (M3).",
                "required": False,
            },
            {
                "flag": "--config",
                "type": str,
                "default": "",
                "help": "Pilih skenario dari config.yml (contoh: M1, M2, M3, dst). Mengesampingkan argumen lain.",
                "required": False,
            },
        ],
    ).parse_args()

    # Jika --config digunakan, baca dari config.yml dan timpa (override) args
    if args.config:
        try:
            with open("config.yml", "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if args.config in config_data:
                scenario = config_data[args.config]
                log(
                    domain="Main",
                    msg=f"Memuat konfigurasi YML untuk skenario: [{args.config}] - {scenario.get('description', '')}",
                    level=log_level.INFO,
                )

                if "model_name" in scenario:
                    args.model_name = scenario["model_name"]
                if "char_type" in scenario:
                    args.char_type = scenario["char_type"]
                if "use_crf" in scenario:
                    args.use_crf = scenario["use_crf"]
                if "use_word_bilstm" in scenario:
                    args.use_word_bilstm = scenario["use_word_bilstm"]
            else:
                log(
                    domain="Main",
                    msg=f"Peringatan: Skenario '{args.config}' tidak ditemukan di config.yml!",
                    level=log_level.WARNING,
                )
        except Exception as e:
            log(
                domain="Main",
                msg=f"Gagal membaca config.yml: {e}",
                level=log_level.ERROR,
            )

    main(
        data_path=args.data_path,
        model_name=args.model_name,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        char_type=args.char_type,
        use_crf=args.use_crf,
        use_word_bilstm=args.use_word_bilstm,
        config_name=args.config,
    )
