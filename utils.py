import os
import enum
import argparse

from datetime import datetime as dt
from zoneinfo import ZoneInfo as tz
from huggingface_hub import snapshot_download

log_level = enum.Enum("LogLevel", "DEBUG INFO WARNING ERROR CRITICAL")


def timestamp() -> str:
    return dt.now(tz("Asia/Jakarta")).strftime("%Y-%m-%d_%H-%M-%S")

# Inisialisasi folder logs dan tentukan nama file log untuk sesi ini
os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", f"run_{timestamp()}.log")


def dowloadModel(model_name: str) -> str:
    log(domain="DownloadModel", msg=f"Downloading model: {model_name}", level=log_level.INFO)

    snapshot_download(model_name, local_dir=os.path.join("models", model_name))

    log(domain="DownloadModel", msg=f"Model {model_name} downloaded successfully", level=log_level.INFO)

    return os.path.join("models", model_name)


def argParser(description: str, args: list) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    for arg in args:
        parser.add_argument(
            arg["flag"],
            type=arg["type"],
            help=arg["help"],
            required=arg.get("required", False),
            default=arg.get("default"),
        )

    return parser


def dataInfo(dataframe) -> None:
    log(domain="DataInfo", msg=f"Dataframe shape: {dataframe.shape}", level=log_level.INFO)
    log(domain="DataInfo", msg=f"Dataframe columns: {dataframe.columns.tolist()}", level=log_level.INFO)


def log(domain: str, msg: str, level: log_level = log_level.INFO) -> None:
    levels = {
        log_level.DEBUG: "DEBUG",
        log_level.INFO: "INFO",
        log_level.WARNING: "WARNING",
        log_level.ERROR: "ERROR",
        log_level.CRITICAL: "CRITICAL",
    }
    
    level_str = levels.get(level, "INFO")
    formatted_msg = f"[{timestamp()}] [{level_str}] [{domain}] {msg}"
    
    # Cetak ke konsol
    print(formatted_msg)
    
    # Simpan ke file log
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
    except Exception as e:
        print(f"[{timestamp()}] [ERROR] Gagal menulis log ke file: {e}")
