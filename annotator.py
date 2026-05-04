import json
import os
import re
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Import arsitektur model dan fungsi preprocessing dari file Anda
from feature_extraction import HybridModel
from preprocess import clean_text, normalize_text, prepare_char_ids


class POSAnnotator:
    """
    Kelas sederhana untuk melakukan anotasi POS Tagging menggunakan
    Hybrid Model (IndoBERT + Char-CNN/BiLSTM + CRF).
    Sangat cocok untuk di-import ke dalam sistem hilir (Flask, FastAPI, dll).
    """

    def __init__(
        self,
        model_dir: str,
        huggingface_model: str,
        char_type: str = "bilstm",
        use_crf: bool = True,
        use_word_bilstm: bool = False,
        device: str = None,
    ):
        """
        Inisialisasi anotator.

        Args:
            model_dir (str): Folder tempat best_model.pt, char_vocab.json, dan class_mappings.json berada (misal: 'outputs/M6').
            huggingface_model (str): Nama model dasar di HuggingFace (misal: 'dafqi/IndoBertTweet').
            char_type (str): Tipe karakter ekstraktor ('none', 'cnn', 'bilstm').
            use_crf (bool): Apakah menggunakan CRF layer.
            use_word_bilstm (bool): Apakah menggunakan Word-BiLSTM layer.
            device (str): 'cuda' atau 'cpu'. Jika None, otomatis mendeteksi ketersediaan GPU.
        """
        # Auto-detect device, fallback to CPU if GPU has compatibility issues
        if device:
            self.device = device
        else:
            try:
                if torch.cuda.is_available():
                    # Try to use CUDA; if it fails, fallback to CPU
                    torch.cuda.current_device()
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except RuntimeError:
                # CUDA incompatibility error - use CPU
                self.device = "cpu"
                print("⚠️  GPU incompatibility detected. Using CPU instead.")
        self.char_type = char_type
        self.use_crf = use_crf
        self.use_word_bilstm = use_word_bilstm
        self.max_word_len = 50

        # Load Vocab & Mappings
        vocab_path = os.path.join(model_dir, "char_vocab_m6.json")
        mapping_path = os.path.join(model_dir, "class_mappings_m6.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.char_vocab = json.load(f)

        with open(mapping_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
            self.class_to_idx = mappings["class_to_idx"]
            self.idx_to_class = {int(k): v for k, v in mappings["idx_to_class"].items()}

        num_classes = len(self.class_to_idx)

        # Load Tokenizer & Model Dasar
        print(f"Loading tokenizer & base model from {huggingface_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        bert_model = AutoModel.from_pretrained(huggingface_model)

        # Inisialisasi Arsitektur
        self.model = HybridModel(
            char_vocab_size=len(self.char_vocab),
            bert=bert_model,
            num_classes=num_classes,
            class_weights=None,  # Tidak diperlukan saat inference
            char_type=self.char_type,
            use_crf=self.use_crf,
            use_word_bilstm=self.use_word_bilstm,
        )

        # Load Weights (Checkpoint)
        model_path = os.path.join(model_dir, "best_model_m6.pt")
        print(f"Loading checkpoint weights from {model_path}...")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Hapus loss weight jika ada
        if "ce_loss.weight" in state_dict:
            del state_dict["ce_loss.weight"]

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"Annotator siap dijalankan pada device: {self.device}")

    def _tokenize_text(self, text: str) -> List[str]:
        """Pemisahan tanda baca dasar untuk membentuk list kata."""
        text = re.sub(r"([?!,.\(\)])", r" \1 ", text)
        return [t for t in text.split() if t]

    def annotate(
        self, text: str, return_format: str = "dict"
    ) -> Union[List[Dict[str, str]], List[tuple]]:
        """
        Melakukan anotasi POS Tagging pada satu kalimat atau paragraf.

        Args:
            text (str): Teks kalimat atau paragraf yang akan dianotasi.
            return_format (str): 'dict' mengembalikan [{'token': 'saya', 'tag': 'PR-P1'}],
                                 'tuple' mengembalikan [('saya', 'PR-P1')]

        Returns:
            Hasil anotasi kata beserta labelnya.
        """
        # Normalisasi
        text = normalize_text(clean_text(text))
        words = self._tokenize_text(text)

        if not words:
            return []

        # Encode teks dengan tokenizer
        encoding = self.tokenizer(
            [words],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = [encoding.word_ids(batch_index=0)]

        # Siapkan karakter embedding
        cids = prepare_char_ids(words, self.char_vocab, self.max_word_len)
        char_ids = torch.from_numpy(np.expand_dims(cids, axis=0)).to(
            self.device
        )  # (1, S_word, W)

        # Inference
        with torch.no_grad():
            # Support FP16 AMP jika tersedia
            amp_ctx = (
                torch.amp.autocast(device_type="cuda")
                if self.device == "cuda"
                else torch.amp.autocast(device_type="cpu", enabled=False)
            )

            with amp_ctx:
                preds = self.model(
                    char_ids=char_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_ids=word_ids,
                    labels=None,
                )

        # Decode (dari indeks angka ke label teks)
        pred_seq = preds[0].tolist()
        tags = []
        for w in range(len(words)):
            tag_idx = pred_seq[w] if w < len(pred_seq) else 0
            tags.append(self.idx_to_class.get(tag_idx, "UNID"))

        # Format kembalian
        if return_format == "tuple":
            return list(zip(words, tags))
        else:
            return [{"token": w, "tag": t} for w, t in zip(words, tags)]


# --- CONTOH PENGGUNAAN (Hanya dieksekusi jika file dijalankan secara langsung) ---
if __name__ == "__main__":
    # Ini contoh asumsikan kita sedang menggunakan skenario M6 (IndoBERTweet + BiLSTM + CRF)
    print("=== Demo POS Annotator ===")
    try:
        annotator = POSAnnotator(
            model_dir="training_result/M6",
            huggingface_model="dafqi/IndoBertTweet",
            char_type="bilstm",
            use_crf=True,
            use_word_bilstm=False,
            device="cpu",  # Force CPU if GPU has compatibility issues
        )

        teks_sampel = "Bsk gw mau pergi belanja ke Ps. Minggu buat beli sayuran seger bangetttt njirr wkwk 😂"
        hasil = annotator.annotate(teks_sampel)

        print("\n[Input Text]:", teks_sampel)
        print("\n[Hasil Anotasi]:")
        for item in hasil:
            print(f"{item['token']:<15} : {item['tag']}")

    except RuntimeError as e:
        if "CUDA" in str(e) or "cuDNN" in str(e):
            print("⚠️  GPU/CUDA compatibility error detected. Retrying with CPU...")
            try:
                annotator = POSAnnotator(
                    model_dir="training_result/M6",
                    huggingface_model="dafqi/IndoBertTweet",
                    char_type="bilstm",
                    use_crf=True,
                    use_word_bilstm=False,
                    device="cpu",
                )
                teks_sampel = "Bsk gw mau pergi belanja ke Ps. Minggu buat beli sayuran seger bangetttt njirr wkwk 😂"
                hasil = annotator.annotate(teks_sampel)

                print("\n[Input Text]:", teks_sampel)
                print("\n[Hasil Anotasi]:")
                for item in hasil:
                    print(f"{item['token']:<15} : {item['tag']}")
            except Exception as cpu_error:
                print(f"❌ Error even with CPU: {cpu_error}")
        else:
            raise
    except FileNotFoundError:
        print(
            "Folder 'training_result/M6' belum memiliki model yang lengkap. Lakukan training terlebih dahulu."
        )
