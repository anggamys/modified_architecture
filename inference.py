import re
import json
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from utils import log, log_level
from feature_extraction import HybridModel
from preprocess import prepare_char_ids, normalize_text, clean_text


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Regex untuk memisahkan emoji dari kata yang menempel.
# Mencakup hampir semua blok Unicode emoji yang umum digunakan.
_POLA_EMOJI = re.compile(
    r"("
    r"[\U0001F600-\U0001F64F]"  # Emoticons wajah
    r"|[\U0001F300-\U0001F5FF]"  # Simbol rupa-rupa & Piktograf
    r"|[\U0001F680-\U0001F6FF]"  # Transportasi & Peta
    r"|[\U0001F900-\U0001F9FF]"  # Simbol & Piktograf Tambahan
    r"|[\U0001FA70-\U0001FAFF]"  # Simbol Baru (Medis, Alam, dll)
    r"|[\u2600-\u26FF]"  # Simbol Miscellaneous (matahari, awan, dll)
    r"|[\u2700-\u27BF]"  # Dingbats (tanda centang, gunting, dll)
    r")",
    flags=re.UNICODE,
)


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
        log(
            domain="Main",
            msg=f"Tidak ada file .txt di: {folder_path}",
            level=log_level.WARNING,
        )
        return []

    log(
        domain="Main", msg=f"Ditemukan {len(txt_files)} file .txt", level=log_level.INFO
    )

    results: list[tuple[str, str]] = []

    # Regex untuk deteksi file WhatsApp secara umum (termasuk pesan sistem)
    pola_timestamp = re.compile(
        r"^\d{2}/\d{2}/\d{2,4}[, ]+\d{1,2}[:.]\d{2}(?:\s*[a-zA-Z]{2})? - "
    )
    # Regex untuk menangkap pesan yang dikirim oleh pengguna (memiliki 'Nama: Pesan')
    pola_chat = re.compile(
        r"^\d{2}/\d{2}/\d{2,4}[, ]+\d{1,2}[:.]\d{2}(?:\s*[a-zA-Z]{2})? - (.*?): (.*)"
    )

    for txt_file in txt_files:
        file_id = txt_file.stem  # nama file tanpa ekstensi
        prev_count = len(results)

        try:
            raw = txt_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log(
                domain="Main",
                msg=f"Gagal baca {txt_file.name}: {e}",
                level=log_level.WARNING,
            )
            continue

        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        if not lines:
            continue

        # Auto-deteksi format WhatsApp dengan mengecek 10 baris pertama
        is_whatsapp = any(
            pola_timestamp.match(sample_line) for sample_line in lines[:10]
        )

        if is_whatsapp:
            for line in lines:
                cocok = pola_chat.match(line)
                if cocok:
                    pesan = cocok.group(2).strip()

                    # 1. Filter Pesan Sampah (Media & URL)
                    if pesan == "<Media omitted>":
                        continue
                    # Ganti URL dengan token khusus
                    pesan = re.sub(r"http[s]?://\S+", "[URL]", pesan)

                    # 2. Pisahkan emoji dengan spasi agar tidak menempel ke kata
                    #    Contoh: "maaf🙏ya" → "maaf 🙏 ya"
                    pesan = _POLA_EMOJI.sub(r" \1 ", pesan)

                    # 3. Tokenisasi Sederhana (pisahkan tanda baca umum)
                    pesan = re.sub(r"([?!,.\(\)])", r" \1 ", pesan)

                    # Pecah berdasarkan spasi dan buang spasi kosong
                    tokens = [t for t in pesan.split() if t]

                    # 3. Masukkan ke format untuk inference
                    if len(tokens) >= min_words:
                        sent = " ".join(tokens)
                        results.append((file_id, sent))
        else:
            for line in lines:
                # Normalisasi teks (tanpa lowercase!)
                line = normalize_text(clean_text(line))

                # Pecah paragraf → kalimat
                for sent in _split_sentences(line):
                    if _is_valid_sentence(sent, min_words):
                        results.append((file_id, sent))

        file_count = len(results) - prev_count
        tipe_file = "WhatsApp" if is_whatsapp else "Teks Standar"
        log(
            domain="Main",
            msg=f"  {txt_file.name} ({tipe_file}): {file_count:,} kalimat/pesan terekstrak",
            level=log_level.INFO,
        )

    if limit > 0:
        results = results[:limit]
        log(
            domain="Main", msg=f"Dibatasi {limit} kalimat pertama", level=log_level.INFO
        )

    total_tokens = sum(len(sent.split()) for _, sent in results)
    log(
        domain="Main",
        msg=f"Total kalimat siap diinference: {len(results):,} ({total_tokens:,} token)",
        level=log_level.INFO,
    )

    return results


def load_model(
    model_path: str,
    vocab_path: str,
    mapping_path: str,
    model_name: str,
    device: str,
    char_type: str = "cnn",
    use_crf: bool = True,
) -> tuple[HybridModel, dict[str, int], dict[int, str], PreTrainedTokenizerBase]:
    """
    Load trained model dan dependencies untuk inference.

    Input files (dari training output):
    1. model_path (e.g., outputs/M6/best_model_m6.pt)
       - Model state_dict (weights) dari train.py → main.py
       - Dimuat ke model architecture yang di-instantiate di sini

    2. vocab_path (e.g., outputs/M6/char_vocab_m6.json)
       - Character vocabulary dari main.py
       - Format: {char: integer_id, ...}
       - Digunakan di run_inference() untuk prepare_char_ids()

    3. mapping_path (e.g., outputs/M6/class_mappings_m6.json)
       - Class ↔ Index mapping dari main.py
       - Format: {"class_to_idx": {tag: idx}, "idx_to_class": {idx: tag}}
       - Digunakan di _decode_preds() untuk convert predictions → tag names

    Args:
        model_path: Path ke saved model weights (.pt file)
        vocab_path: Path ke char_vocab.json
        mapping_path: Path ke class_mappings.json
        model_name: HuggingFace BERT model identifier
        device: Device untuk inference (cuda/cpu)
        char_type: Character extractor type ('none', 'cnn', 'bilstm')
        use_crf: Whether CRF layer was used during training

    Returns:
        Tuple of (model, char_vocab, idx_to_class, tokenizer):
        - model: HybridModel instance dengan loaded weights
        - char_vocab: Dictionary {char → id}
        - idx_to_class: Dictionary {id → tag_name}
        - tokenizer: BERT tokenizer untuk subword tokenization
    """
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
        char_type=char_type,
        use_crf=use_crf,
    )

    state = torch.load(model_path, map_location=device)

    # strict=False: ce_loss.weight adalah buffer CrossEntropyLoss yang hanya ada
    # saat training (class_weights != None). Saat inference tidak dibutuhkan.
    incompatible = model.load_state_dict(state, strict=False)

    if incompatible.unexpected_keys:
        log(
            domain="Main",
            msg=f"State dict keys dilewati (inference-only, aman): {incompatible.unexpected_keys}",
            level=log_level.INFO,
        )

    if incompatible.missing_keys:
        log(
            domain="Main",
            msg=f"State dict keys hilang (periksa arsitektur): {incompatible.missing_keys}",
            level=log_level.WARNING,
        )

    model.to(device)
    model.eval()

    log(
        domain="Main",
        msg=f"Model loaded: {model_path} | {num_classes} kelas | vocab={len(char_vocab)}",
        level=log_level.INFO,
    )

    return model, char_vocab, idx_to_class, tokenizer


def _decode_preds(
    preds: "torch.Tensor",
    words_batch: list[list[str]],
    idx_to_class: dict[int, str],
) -> list[list[str]]:
    # Preds sudah berada pada level kata (S_word) berkat pooling
    results: list[list[str]] = []

    for b, words in enumerate(words_batch):
        pred_seq = preds[b].tolist()
        tags: list[str] = []

        for w in range(len(words)):
            tag_idx = pred_seq[w] if w < len(pred_seq) else 0
            tags.append(idx_to_class.get(tag_idx, "UNID"))

        results.append(tags)

    return results


def _predict_batch(
    words_batch: list[list[str]],
    model: HybridModel,
    char_vocab: dict[str, int],
    idx_to_class: dict[int, str],
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_word_len: int = 50,
    max_seq_len: int = 512,
    use_amp: bool = True,
) -> list[list[str]]:
    encoding = tokenizer(
        words_batch,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,  # pad ke kalimat terpanjang dalam batch
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids = torch.as_tensor(encoding["input_ids"], device=device)  # (B, S_bert)
    attention_mask = torch.as_tensor(
        encoding["attention_mask"], device=device
    )  # (B, S_bert)
    word_ids_batch = [encoding.word_ids(batch_index=i) for i in range(len(words_batch))]

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
        torch.autocast(device_type="cuda")
        if use_amp and device == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), amp_ctx:
        preds = model(
            char_ids=char_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids_batch,
            labels=None,
        )  # (B, S_word)

    return _decode_preds(preds, words_batch, idx_to_class)


def _part_path(output_dir: Path, stem: str, part: int, suffix: str) -> Path:
    return output_dir / f"{stem}_part{part:03d}{suffix}"


def _source_path(output_dir: Path, stem: str, file_id: str, suffix: str) -> Path:
    safe_id = re.sub(r"[^\w\-]", "_", file_id)
    return output_dir / f"{stem}_{safe_id}{suffix}"


def run_inference(
    sentences: list[tuple[str, str]],
    model: HybridModel,
    char_vocab: dict[str, int],
    idx_to_class: dict[int, str],
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    output_path: str,
    batch_size: int = 32,
    max_word_len: int = 50,
    log_every: int = 5_000,
    flush_every: int = 50_000,
    split_by: str = "file",  # "file" | "rows" | "none"
    rows_per_file: int = 500_000,  # aktif hanya jika split_by="rows"
) -> int:
    """
    Run inference pada unlabeled corpus dan simpan pseudo-labels ke CSV.

    PROCESS:
    1. Tokenisasi kalimat dengan BERT tokenizer
    2. Prepare char-level features menggunakan char_vocab
    3. Forward melalui model (char + BERT + optional CRF)
    4. Decode predictions → POS tag names menggunakan idx_to_class
    5. Simpan token-level predictions ke CSV dengan kolom:
       - file_id: Source file identifier
       - sentence_id: Global sentence number (1-indexed)
       - token_index: Token position dalam sentence (1-indexed)
       - token: Actual word token
       - pos_tag_pred: Predicted POS tag (dari idx_to_class)
       - pos_tag_koreksi: Empty column untuk manual correction

    OUTPUT FILES:
    - split_by="none": {output_path} (single file)
    - split_by="file": {output_dir}/{stem}_files/{stem}_{file_id}.csv
    - split_by="rows": {output_dir}/{stem}_parts/{stem}_part###.csv

    Args:
        sentences: List of (file_id, sentence_text) tuples dari load_corpus_from_folder()
        model: Loaded HybridModel for inference
        char_vocab: Character vocabulary dari load_model()
        idx_to_class: POS tag index mapping dari load_model()
        tokenizer: BERT tokenizer dari load_model()
        device: Device untuk inference (cuda/cpu)
        output_path: Output CSV path
        batch_size: Batch size untuk GPU inference
        max_word_len: Max character count per word for char-level features
        log_every: Log progress every N sentences
        flush_every: Flush buffer to CSV every N tokens
        split_by: Output splitting strategy
        rows_per_file: Rows per file jika split_by="rows"

    Returns:
        Total number of tokens written to output files
    """
    total = len(sentences)
    total_tokens = 0
    buffer: list[dict] = []
    output = Path(output_path)
    use_amp = device == "cuda"

    # Siapkan direktori output untuk mode split
    if split_by in ("file", "rows"):
        output_dir = output.parent / output.stem  # ./data_pseudo_label/
        output_dir.mkdir(parents=True, exist_ok=True)
        log(domain="Main", msg=f"Output folder: {output_dir}/", level=log_level.INFO)
    else:
        output_dir = output.parent  # satu file langsung di parent dir

    stem = output.stem
    suffix = output.suffix

    # --- state untuk split ---
    writers: dict[str, bool] = {}  # path_str → header_sudah_ditulis?
    current_part = 1
    sent_in_part = 0  # jumlah KALIMAT (bukan token) di part ini

    def _flush(buf: list[dict]) -> None:
        nonlocal current_part, sent_in_part

        if not buf:
            return

        chunk_df = pd.DataFrame(buf)

        if split_by == "none":
            dest = output
            mode = "w" if str(dest) not in writers else "a"
            chunk_df.to_csv(
                dest,
                mode=mode,
                index=False,
                header=(str(dest) not in writers),
                encoding="utf-8-sig",
            )
            writers[str(dest)] = True

        elif split_by == "file":
            for fid, grp in chunk_df.groupby("file_id", sort=False):
                dest = _source_path(output_dir, stem, str(fid), suffix)
                key = str(dest)
                mode = "w" if key not in writers else "a"
                grp.to_csv(
                    dest,
                    mode=mode,
                    index=False,
                    header=(key not in writers),
                    encoding="utf-8-sig",
                )
                writers[key] = True

        elif split_by == "rows":
            sent_ids_in_buf = chunk_df["sentence_id"].unique()
            for sid in sent_ids_in_buf:
                rows = chunk_df[chunk_df["sentence_id"] == sid]
                dest = _part_path(output_dir, stem, current_part, suffix)
                key = str(dest)
                mode = "w" if key not in writers else "a"
                rows.to_csv(
                    dest,
                    mode=mode,
                    index=False,
                    header=(key not in writers),
                    encoding="utf-8-sig",
                )
                writers[key] = True
                sent_in_part += 1

                if sent_in_part >= rows_per_file:
                    current_part += 1
                    sent_in_part = 0

    # --- loop utama ---
    for batch_start in range(0, total, batch_size):
        batch = sentences[batch_start : batch_start + batch_size]
        file_ids = [fid for fid, _ in batch]
        words_batch = [sent.split() for _, sent in batch]
        sent_ids = list(range(batch_start + 1, batch_start + len(batch) + 1))

        tags_batch = _predict_batch(
            words_batch,
            model,
            char_vocab,
            idx_to_class,
            tokenizer,
            device,
            max_word_len,
            use_amp=use_amp,
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
                domain="Main",
                msg=f"Inference {processed:,}/{total:,} kalimat | Total ditulis = {total_tokens:,} token",
                level=log_level.INFO,
            )

        if len(buffer) >= flush_every or processed >= total:
            _flush(buffer)
            total_tokens += len(buffer)
            buffer.clear()

    # --- ringkasan file output ---
    written_files = list(writers.keys())
    log(
        domain="Main",
        msg=f"Selesai: {total_tokens:,} token dari {total:,} kalimat → {len(written_files)} file CSV",
        level=log_level.INFO,
    )
    for fpath in written_files:
        log(domain="Main", msg=f"  {fpath}", level=log_level.INFO)

    return total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-labeling inference pipeline menggunakan model terlatih dari main.py",
        epilog=(
            "USAGE EXAMPLES:\n"
            "  1. Dengan output paths dari training (config_name='M6'):\n"
            "     python inference.py --config_name M6\n"
            "  2. Dengan custom paths:\n"
            "     python inference.py --model_path ./my_model.pt "
            "--vocab_path ./my_vocab.json --mapping_path ./my_mapping.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="",
        help=(
            "Nama scenario dari training (M1-M6, M6a, M6b). "
            "Jika diisi, akan otomatis load paths dari outputs/{config_name}/. "
            "Contoh: --config_name M6 akan load outputs/M6/best_model_m6.pt, dll"
        ),
    )

    parser.add_argument(
        "--corpora_dir",
        type=str,
        default="./raw_corpora",
        help="Folder berisi file .txt mentah untuk pseudo-labeling",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help=(
            "Path ke checkpoint model (.pt). "
            "Jika kosong, akan diambil dari outputs/{config_name}/best_model_{config_name_lower}.pt"
        ),
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default="",
        help=(
            "Path ke char_vocab.json (disimpan oleh main.py). "
            "Jika kosong, akan diambil dari outputs/{config_name}/char_vocab_{config_name_lower}.json"
        ),
    )

    parser.add_argument(
        "--mapping_path",
        type=str,
        default="",
        help=(
            "Path ke class_mappings.json (disimpan oleh main.py). "
            "Jika kosong, akan diambil dari outputs/{config_name}/class_mappings_{config_name_lower}.json"
        ),
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="HuggingFace model name (e.g., indobenchmark/indobert-base-p1, dafqi/IndoBertTweet)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data_pseudo_label.csv",
        help="Path output CSV hasil pseudo-labeling",
    )

    parser.add_argument(
        "--char_type",
        type=str,
        default="",
        help=(
            "Tipe char extractor: 'none', 'cnn', atau 'bilstm'. "
            "Jika kosong dan --config_name diberikan, akan diambil dari konfigurasi scenario."
        ),
    )

    parser.add_argument(
        "--use_crf",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=None,
        help=(
            "Aktifkan modul CRF. Jika tidak diberikan dan --config_name ada, "
            "akan diambil dari konfigurasi scenario. Default=True jika config_name tidak ada."
        ),
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

    parser.add_argument(
        "--split_by",
        type=str,
        default="file",
        choices=["file", "rows", "none"],
        help=(
            "Strategi pemecahan output CSV. "
            "'file' = 1 CSV per file sumber (default). "
            "'rows' = pecah per N kalimat (lihat --rows_per_file). "
            "'none' = satu file tunggal."
        ),
    )

    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=500_000,
        help="Batas jumlah kalimat per file output (aktif hanya jika --split_by=rows)",
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(domain="Main", msg=f"Device: {device}", level=log_level.INFO)

    # --- Resolve paths berdasarkan config_name atau explicit paths ---
    model_path = args.model_path
    vocab_path = args.vocab_path
    mapping_path = args.mapping_path
    char_type = args.char_type
    use_crf = args.use_crf
    model_name = args.model_name

    if args.config_name:
        # Load dari outputs/{config_name}/
        config_name_lower = args.config_name.lower()
        output_dir = Path("outputs") / args.config_name

        # Auto-resolve paths jika tidak di-explicit-kan
        if not model_path:
            model_path = str(output_dir / f"best_model_{config_name_lower}.pt")
        if not vocab_path:
            vocab_path = str(output_dir / f"char_vocab_{config_name_lower}.json")
        if not mapping_path:
            mapping_path = str(output_dir / f"class_mappings_{config_name_lower}.json")

        # Load config scenario untuk char_type & use_crf jika belum explicit
        try:
            with open("config.yml", "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if args.config_name in config_data:
                scenario = config_data[args.config_name]
                log(
                    domain="Main",
                    msg=f"Memuat konfigurasi scenario [{args.config_name}]: {scenario.get('description', '')}",
                    level=log_level.INFO,
                )

                if not char_type and "char_type" in scenario:
                    char_type = scenario["char_type"]
                if use_crf is None and "use_crf" in scenario:
                    use_crf = scenario["use_crf"]
                if not model_name and "model_name" in scenario:
                    model_name = scenario["model_name"]
            else:
                log(
                    domain="Main",
                    msg=f"Peringatan: Scenario '{args.config_name}' tidak ditemukan di config.yml!",
                    level=log_level.WARNING,
                )
        except Exception as e:
            log(
                domain="Main",
                msg=f"Gagal membaca config.yml: {e}",
                level=log_level.WARNING,
            )

    # --- Set defaults jika masih kosong ---
    if not char_type:
        char_type = "bilstm"
    if use_crf is None:
        use_crf = True
    if not model_name:
        model_name = "indobenchmark/indobert-base-p1"

    log(
        domain="Main",
        msg=f"Konfigurasi: model_name={model_name}, char_type={char_type}, use_crf={use_crf}",
        level=log_level.INFO,
    )

    # --- Validasi file paths ---
    for fpath, fname in [
        (model_path, "Model"),
        (vocab_path, "Vocab"),
        (mapping_path, "Class Mappings"),
    ]:
        if not Path(fpath).exists():
            log(
                domain="Main",
                msg=f"ERROR: {fname} file tidak ditemukan: {fpath}",
                level=log_level.ERROR,
            )
            return

    # 1. Baca corpora
    sentences = load_corpus_from_folder(
        args.corpora_dir,
        min_words=args.min_words,
        limit=args.limit,
    )

    if not sentences:
        log(
            domain="Main",
            msg="Tidak ada kalimat untuk diproses. Keluar.",
            level=log_level.WARNING,
        )
        return

    # 2. Load model
    model, char_vocab, idx_to_class, tokenizer = load_model(
        model_path=model_path,
        vocab_path=vocab_path,
        mapping_path=mapping_path,
        model_name=model_name,
        device=device,
        char_type=char_type,
        use_crf=use_crf,
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
        split_by=args.split_by,
        rows_per_file=args.rows_per_file,
    )

    log(
        domain="Main",
        msg="Pipeline selesai. Output: "
        f"{args.output} (atau folder {Path(args.output).parent / Path(args.output).stem}/ jika split)",
        level=log_level.INFO,
    )
    log(
        domain="Main",
        msg="Buka file pseudo labels, koreksi kolom 'pos_tag_koreksi', lalu gabungkan ke dataset training.",
        level=log_level.INFO,
    )


if __name__ == "__main__":
    main()
