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
        prev_count = len(results)  # snapshot sebelum file ini diproses

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

        # Tampilkan jumlah kalimat dari file ini saja (bukan akumulatif)
        file_count = len(results) - prev_count
        log(
            f"  {txt_file.name}: {file_count:,} kalimat terekstrak",
            level=log_level.INFO,
        )

    if limit > 0:
        results = results[:limit]

        log(f"Dibatasi {limit} kalimat pertama", level=log_level.INFO)

    log(f"Total kalimat siap diinference: {len(results):,}", level=log_level.INFO)

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

    # strict=False: ce_loss.weight adalah buffer CrossEntropyLoss yang hanya ada
    # saat training (class_weights != None). Saat inference tidak dibutuhkan.
    incompatible = model.load_state_dict(state, strict=False)

    if incompatible.unexpected_keys:
        log(
            f"State dict keys dilewati (inference-only, aman): {incompatible.unexpected_keys}",
            level=log_level.INFO,
        )

    if incompatible.missing_keys:
        log(
            f"State dict keys hilang (periksa arsitektur): {incompatible.missing_keys}",
            level=log_level.WARNING,
        )

    model.to(device)
    model.eval()

    log(
        f"Model loaded: {model_path} | {num_classes} kelas | vocab={len(char_vocab)}",
        level=log_level.INFO,
    )

    return model, char_vocab, idx_to_class, tokenizer


def _decode_preds(
    preds: "torch.Tensor",
    word_ids_batch: list[list[int | None]],
    words_batch: list[list[str]],
    idx_to_class: dict[int, str],
) -> list[list[str]]:
    # Decode predictions to tags
    results: list[list[str]] = []

    for b, (word_ids, words) in enumerate(zip(word_ids_batch, words_batch)):
        pred_seq = preds[b].tolist()
        tags: list[str] = []
        prev_word: int | None = None

        for t, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if word_id != prev_word:
                tag_idx = pred_seq[t] if t < len(pred_seq) else 0
                tags.append(idx_to_class.get(tag_idx, "UNID"))

            prev_word = word_id

        if len(tags) < len(words):
            tags += ["UNID"] * (len(words) - len(tags))

        results.append(tags[: len(words)])

    return results


def _predict_batch(
    words_batch: list[list[str]],
    model: HybridModel,
    char_vocab: dict[str, int],
    idx_to_class: dict[int, str],
    tokenizer: AutoTokenizer,
    device: str,
    max_word_len: int = 50,
    max_seq_len: int = 512,
    use_amp: bool = True,
) -> list[list[str]]:
    encoding = tokenizer(
        words_batch,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,           # pad ke kalimat terpanjang dalam batch
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids = encoding["input_ids"].to(device)          # (B, S_bert)
    attention_mask = encoding["attention_mask"].to(device) # (B, S_bert)
    word_ids_batch = [
        encoding.word_ids(batch_index=i) for i in range(len(words_batch))
    ]

    max_words = max(len(w) for w in words_batch)
    char_ids_list: list[np.ndarray] = []

    for words in words_batch:
        cids = prepare_char_ids(words, char_vocab, max_word_len)  # (S_word, W)

        # Pad ke max_words jika perlu
        if len(words) < max_words:
            pad = np.zeros((max_words - len(words), max_word_len), dtype=np.int64)
            cids = np.concatenate([cids, pad], axis=0)

        char_ids_list.append(cids)

    char_ids = torch.from_numpy(
        np.stack(char_ids_list, axis=0)  # (B, S_word, W)
    ).to(device)

    amp_ctx = (
        torch.amp.autocast(device_type="cuda")
        if use_amp and device == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), amp_ctx:
        preds = model(
            char_ids=char_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids_batch,
            labels=None,
        )  # (B, S_bert)

    return _decode_preds(preds, word_ids_batch, words_batch, idx_to_class)


def run_inference(
    sentences: list[tuple[str, str]],
    model: HybridModel,
    char_vocab: dict[str, int],
    idx_to_class: dict[int, str],
    tokenizer: AutoTokenizer,
    device: str,
    output_path: str,
    batch_size: int = 32,
    max_word_len: int = 50,
    log_every: int = 5_000,
    flush_every: int = 50_000,
) -> int:
    total = len(sentences)
    total_tokens = 0
    buffer: list[dict] = []
    header_written = False
    output = Path(output_path)
    use_amp = device == "cuda"

    # Iterasi per-batch, bukan per-kalimat
    for batch_start in range(0, total, batch_size):
        batch = sentences[batch_start : batch_start + batch_size]
        file_ids   = [fid  for fid, _    in batch]
        words_batch = [sent.split() for _, sent in batch]
        sent_ids   = list(range(batch_start + 1, batch_start + len(batch) + 1))

        tags_batch = _predict_batch(
            words_batch, model, char_vocab, idx_to_class,
            tokenizer, device, max_word_len, use_amp=use_amp,
        )

        for file_id, global_sent_id, words, tags in zip(
            file_ids, sent_ids, words_batch, tags_batch
        ):
            for token_idx, (word, tag) in enumerate(zip(words, tags)):
                buffer.append(
                    {
                        "file_id": file_id,
                        "sentence_id": global_sent_id,
                        "token_index": token_idx + 1,
                        "token": word,
                        "pos_tag_pred": tag,
                        "pos_tag_koreksi": "",
                    }
                )

        processed = batch_start + len(batch)
        if processed % log_every == 0 or processed >= total:
            log(
                f"Inference {processed:,}/{total:,} kalimat"
                f" | total_ditulis={total_tokens:,} token",
                level=log_level.INFO,
            )

        if len(buffer) >= flush_every or processed >= total:
            chunk_df = pd.DataFrame(buffer)
            chunk_df.to_csv(
                output,
                mode="w" if not header_written else "a",
                index=False,
                header=not header_written,
                encoding="utf-8-sig",
            )

            total_tokens += len(buffer)
            header_written = True
            buffer.clear()

    log(
        f"Selesai: {total_tokens:,} token dari {total:,} kalimat → {output}",
        level=log_level.INFO,
    )

    return total_tokens


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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Jumlah kalimat per batch GPU (naikkan jika VRAM masih longgar, turunkan jika OOM)",
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

    # 3. Inference + simpan inkremental ke CSV
    run_inference(
        sentences,
        model,
        char_vocab,
        idx_to_class,
        tokenizer,
        device,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    log(
        "Buka CSV, koreksi kolom 'pos_tag_koreksi', lalu gabungkan ke dataset training.",
        level=log_level.INFO,
    )


if __name__ == "__main__":
    main()
