import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from utils import log, log_level


def build_optimizer(model: nn.Module, freeze_bert_layers: int = 8) -> torch.optim.AdamW:
    # --- Layer Freezing: Bekukan N layer pertama IndoBERT ---
    # Layer bawah (0-7) sudah paham bahasa Indonesia standar dari pre-training.
    # Hanya layer atas (8-11) yang perlu belajar fitur spesifik POS tag kita.
    # Ini mencegah overfitting pada dataset kecil (<15.000 token).
    frozen_count = 0
    if freeze_bert_layers > 0:
        for name, param in model.named_parameters():
            if 'encoder.layer' in name:
                try:
                    layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                    if layer_num < freeze_bert_layers:
                        param.requires_grad = False
                        frozen_count += 1
                except (IndexError, ValueError):
                    pass

        log(
            f"Layer Freezing: {frozen_count} parameter dibekukan "
            f"(encoder.layer 0–{freeze_bert_layers - 1})",
            level=log_level.INFO,
        )

    bert_params: list[Tensor] = []
    crf_params: list[Tensor] = []
    other_params: list[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bert" in name:
            bert_params.append(param)
        elif "crf" in name:
            crf_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": 2e-5},
            {"params": other_params, "lr": 1e-3},
            {"params": crf_params, "lr": 5e-3},  # CRF belajar lebih agresif
        ],
        weight_decay=0.01,  # standard untuk BERT fine-tuning
    )

    log(
        f"Optimizer: {len(bert_params)} BERT params (lr=2e-5) | "
        f"{len(other_params)} other params (lr=1e-3) | "
        f"{len(crf_params)} CRF params (lr=5e-3)",
        level=log_level.INFO,
    )

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
) -> object:
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    log(
        f"Scheduler: {num_warmup_steps} warmup steps / {num_training_steps} total",
        level=log_level.INFO,
    )

    return scheduler


def compute_accuracy(preds: list[int], labels: list[int]) -> float:
    if len(labels) == 0:
        return 0.0
    correct = sum(p == l for p, l in zip(preds, labels))  # noqa: E741
    return correct / len(labels)


def compute_classification_report(
    preds: list[int],
    labels: list[int],
    idx_to_class: dict[int, str],
) -> str:
    # Ambil hanya kelas yang muncul di preds atau labels (hindari baris kosong)
    present = sorted(set(preds) | set(labels))
    target_names = [idx_to_class.get(i, str(i)) for i in present]

    report: str = classification_report(
        labels,
        preds,
        labels=present,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )

    log(f"Classification Report:\n{report}", level=log_level.INFO)
    return report


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        char_ids: Tensor = batch["char_ids"].to(device)
        input_ids: Tensor = batch["input_ids"].to(device)
        attention_mask: Tensor = batch["attention_mask"].to(device)
        labels: Tensor = batch["labels"].to(device)
        word_ids: list[list[int | None]] = batch["word_ids"]  # list, jangan .to()

        optimizer.zero_grad()

        loss: Tensor = model(
            char_ids=char_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            labels=labels,
        )

        loss.backward()

        # Gradient clipping: CRF bisa gradient explosion di awal training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            char_ids: Tensor = batch["char_ids"].to(device)
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)
            word_ids: list[list[int | None]] = batch["word_ids"]

            # Forward dengan labels → dapat loss
            loss: Tensor = model(
                char_ids=char_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                labels=labels,
            )

            # Forward tanpa labels → dapat predictions dari Viterbi
            preds: Tensor = model(
                char_ids=char_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                labels=None,
            )

            total_loss += loss.item()

            # Kumpulkan preds & labels hanya di posisi valid (first-subword)
            # Gunakan word_ids untuk filter: ambil posisi first-subword per kalimat
            for b, wids in enumerate(word_ids):
                prev_word: int | None = None
                for t, word_id in enumerate(wids):
                    if word_id is None:
                        continue
                    if word_id != prev_word:
                        # first-subword: ambil pred & label di posisi t
                        all_preds.append(preds[b, t].item())
                        all_labels.append(labels[b, word_id].item())
                    prev_word = word_id

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 10,
    patience: int = 3,
    checkpoint_path: str = "best_model.pt",
) -> None:
    optimizer = build_optimizer(model)

    total_steps = epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps)

    best_val_loss = float("inf")
    no_improve_count = 0  # counter untuk early stopping

    for epoch in range(1, epochs + 1):
        log(f"Epoch {epoch}/{epochs}", level=log_level.INFO)

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device)

        acc = compute_accuracy(val_preds, val_labels)

        log(f"Train Loss : {train_loss:.4f}", level=log_level.INFO)
        log(f"Val Loss   : {val_loss:.4f}", level=log_level.INFO)
        log(f"Val Acc    : {acc:.4f} ({acc * 100:.2f}%)", level=log_level.INFO)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            log(f"Checkpoint saved → {checkpoint_path}", level=log_level.INFO)
        else:
            no_improve_count += 1
            log(
                f"No improvement ({no_improve_count}/{patience})",
                level=log_level.INFO,
            )

            if no_improve_count >= patience:
                log(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val loss: {best_val_loss:.4f}",
                    level=log_level.INFO,
                )
                break

    # Restore best weights sebelum return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    log(f"Training selesai. Best val loss: {best_val_loss:.4f}", level=log_level.INFO)
