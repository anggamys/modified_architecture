import re
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from feature_extraction import HybridModel
from preprocess import prepare_char_ids, normalize_text, clean_text
from utils import log, log_level


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENT_SPLIT.split(paragraph.strip())
    return [s.strip() for s in parts if s.strip()]


def _is_valid_sentence(text: str, min_words: int = 2) -> bool:
    words = text.split()
    return len(words) >= min_words


def load_corpus_from_folder(
    folder_path: str,
    min_words: int = 2,
    limit: int = 0,
) -> list[tuple[str, str]]:
    folder = Path(folder_path)
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        log(f"Tidak ada file .txt di: {folder_path}", level=log_level.WARNING)
        return []

    log(f"Ditemukan {len(txt_files)} file .txt", level=log_level.INFO)

    results: list[tuple[str, str]] = []

    for txt_file in txt_files:
        file_id = txt_file.stem  # nama file tanpa ekstensi
        try:
            raw = txt_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log(f"Gagal baca {txt_file.name}: {e}", level=log_level.WARNING)
            continue

        # Normalisasi line endings
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            # Normalisasi teks (tanpa lowercase!)
            line = normalize_text(clean_text(line))

            # Pecah paragraf → kalimat
            for sent in _split_sentences(line):
                if _is_valid_sentence(sent, min_words):
                    results.append((file_id, sent))

        log(
            f"  {txt_file.name}: {len(results)} kalimat terekstrak",
            level=log_level.INFO,
        )

    if limit > 0:
        results = results[:limit]
        log(f"Dibatasi {limit} kalimat pertama", level=log_level.INFO)

    log(f"Total kalimat siap diinference: {len(results)}", level=log_level.INFO)
    return results


def load_model(
    model_path: str,
    vocab_path: str,
    mapping_path: str,
    model_name: str,
    device: str,
) -> tuple[HybridModel, dict[str, int], dict[int, str], AutoTokenizer]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_vocab: dict[str, int] = json.load(f)

    with open(mapping_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    class_to_idx: dict[str, int] = mappings["class_to_idx"]
    idx_to_class: dict[int, str] = {
        int(k): v for k, v in mappings["idx_to_class"].items()
    }

    num_classes = len(class_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)

    model = HybridModel(
        char_vocab_size=len(char_vocab),
        bert=bert_model,
        num_classes=num_classes,
        class_weights=None,  # tidak diperlukan saat inference
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    log(
        f"Model loaded: {model_path} | {num_classes} kelas | vocab={len(char_vocab)}",
        level=log_level.INFO,
    )

    return model, char_vocab, idx_to_class, tokenizer


def _predict_sentence(
    words: list[str],
    model: HybridModel,
    char_vocab: dict[str, int],
    idx_to_class: dict[int, str],
    tokenizer: AutoTokenizer,
    device: str,
    max_word_len: int = 50,
    max_seq_len: int = 512,
) -> list[str]:
    if not words:
        return []

    # BERT tokenization
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    word_ids: list[int | None] = encoding.word_ids(batch_index=0)

    # CharCNN ids
    char_ids_np: np.ndarray = prepare_char_ids(words, char_vocab, max_word_len)
    char_ids = torch.from_numpy(char_ids_np).unsqueeze(0).to(device)  # (1, S_word, W)

    with torch.no_grad():
        preds = model(
            char_ids=char_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=[word_ids],  # model expects list[list]
            labels=None,
        )

    # preds: Tensor (1, S_bert) dari Viterbi decode
    pred_seq = preds[0].tolist() if hasattr(preds, "__getitem__") else preds

    # Sinkronisasi subword → kata asli (ambil posisi first-subword)
    tags: list[str] = []
    prev_word: int | None = None
    for t, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word:
            tag_idx = pred_seq[t] if t < len(pred_seq) else 0
            tags.append(idx_to_class.get(tag_idx, "UNID"))
        prev_word = word_id

    # Pastikan panjang tags == panjang words (guard terhadap truncation)
    if len(tags) < len(words):
        tags += ["UNID"] * (len(words) - len(tags))
    return tags[: len(words)]


def run_inference(
    sentences: list[tuple[str, str]],
    model: HybridModel,
    char_vocab: dict[str, int],
    idx_to_class: dict[int, str],
    tokenizer: AutoTokenizer,
    device: str,
    max_word_len: int = 50,
) -> pd.DataFrame:
    records: list[dict] = []
    total = len(sentences)

    for idx, (file_id, sent) in enumerate(sentences):
        if (idx + 1) % 100 == 0 or idx == total - 1:
            log(f"Inference {idx + 1}/{total}...", level=log_level.INFO)

        words = sent.split()
        tags = _predict_sentence(
            words, model, char_vocab, idx_to_class, tokenizer, device, max_word_len
        )

        for token_idx, (word, tag) in enumerate(zip(words, tags)):
            records.append(
                {
                    "file_id": file_id,
                    "sentence_id": idx + 1,
                    "token_index": token_idx + 1,
                    "token": word,
                    "pos_tag_pred": tag,  # tebakan model
                    "pos_tag_koreksi": "",  # kolom kosong untuk koreksi manual
                }
            )

    df = pd.DataFrame(records)
    log(f"Selesai: {len(df)} token dari {total} kalimat.", level=log_level.INFO)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-labeling inference pipeline")
    parser.add_argument(
        "--corpora_dir",
        type=str,
        default="./raw_corpora",
        help="Folder berisi file .txt mentah",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./best_model.pt",
        help="Path ke checkpoint model (.pt)",
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./char_vocab.json",
        help="Path ke char_vocab.json (disimpan oleh main.py)",
    )

    parser.add_argument(
        "--mapping_path",
        type=str,
        default="./class_mappings.json",
        help="Path ke class_mappings.json (disimpan oleh main.py)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="indobenchmark/indobert-base-p1",
        help="HuggingFace model name",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data_pseudo_label.csv",
        help="Path output CSV hasil pseudo-labeling",
    )

    parser.add_argument(
        "--limit", type=int, default=0, help="Batas jumlah kalimat (0 = semua)"
    )

    parser.add_argument(
        "--min_words", type=int, default=2, help="Minimum jumlah kata per kalimat"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}", level=log_level.INFO)

    # 1. Baca corpora
    sentences = load_corpus_from_folder(
        args.corpora_dir,
        min_words=args.min_words,
        limit=args.limit,
    )
    if not sentences:
        log("Tidak ada kalimat untuk diproses. Keluar.", level=log_level.WARNING)
        return

    # 2. Load model
    model, char_vocab, idx_to_class, tokenizer = load_model(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        mapping_path=args.mapping_path,
        model_name=args.model_name,
        device=device,
    )

    # 3. Inference
    df = run_inference(sentences, model, char_vocab, idx_to_class, tokenizer, device)

    # 4. Simpan
    df.to_csv(
        args.output, index=False, encoding="utf-8-sig"
    )  # utf-8-sig agar Excel bisa baca
    log(f"Hasil disimpan ke: {args.output}", level=log_level.INFO)
    log(
        "Buka CSV, koreksi kolom 'pos_tag_koreksi', lalu gabungkan ke dataset training.",
        level=log_level.INFO,
    )


if __name__ == "__main__":
    main()
