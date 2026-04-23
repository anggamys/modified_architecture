import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from utils import log, log_level


def build_optimizer(model: nn.Module, freeze_bert_layers: int = 8) -> torch.optim.AdamW:
    # --- Layer Freezing: Bekukan N layer pertama IndoBERT ---
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

    param_groups = []
    other_params = []
    crf_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "bert" in name:
            if "encoder.layer" in name:
                try:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    # LLRD: Base LR 5e-5 untuk layer 11, turun perlahan ke bawah
                    lr = 5e-5 - (11 - layer_num) * 0.5e-5
                    param_groups.append({"params": [param], "lr": lr})
                except (IndexError, ValueError):
                    param_groups.append({"params": [param], "lr": 2e-5})
            else:
                # Embeddings, pooler
                param_groups.append({"params": [param], "lr": 1e-5})
        elif "crf" in name:
            crf_params.append(param)
        else:
            other_params.append(param)

    # Tambahkan layer luar dengan LR lebih agresif
    param_groups.append({"params": other_params, "lr": 1e-3})
    param_groups.append({"params": crf_params, "lr": 5e-3})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    log(
        f"Optimizer: Layer-wise Learning Rate Decay aktif | "
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
        word_mask: Tensor = batch["word_mask"].to(device)
        labels: Tensor = batch["labels"].to(device)
        word_ids: list[list[int | None]] = batch["word_ids"]

        optimizer.zero_grad()

        loss: Tensor = model(
            char_ids=char_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            word_mask=word_mask,
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
            word_mask: Tensor = batch["word_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)
            word_ids: list[list[int | None]] = batch["word_ids"]

            # Forward dengan labels → dapat loss
            loss: Tensor = model(
                char_ids=char_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                word_mask=word_mask,
                labels=labels,
            )

            # Forward tanpa labels → dapat predictions dari Viterbi
            preds: Tensor = model(
                char_ids=char_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                word_mask=word_mask,
                labels=None,
            )

            total_loss += loss.item()

            # Kumpulkan preds & labels hanya di posisi kata yang valid
            B, S_word = labels.shape
            for b in range(B):
                for w in range(S_word):
                    if word_mask[b, w]:
                        all_preds.append(preds[b, w].item())
                        all_labels.append(labels[b, w].item())

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
