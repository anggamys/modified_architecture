import os
import enum
import argparse

from datetime import datetime as dt
from zoneinfo import ZoneInfo as tz
from huggingface_hub import snapshot_download

log_level = enum.Enum("LogLevel", "DEBUG INFO WARNING ERROR CRITICAL")


def timestamp() -> str:
    return dt.now(tz("Asia/Jakarta")).strftime("%Y-%m-%d_%H-%M-%S")


def dowloadModel(model_name: str) -> str:
    log(f"Downloading model: {model_name}", level=log_level.INFO)

    snapshot_download(model_name, local_dir=os.path.join("models", model_name))

    log(f"Model {model_name} downloaded successfully", level=log_level.INFO)

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
    log(f"Dataframe shape: {dataframe.shape}", level=log_level.INFO)
    log(f"Dataframe columns: {dataframe.columns.tolist()}", level=log_level.INFO)


def log(msg: str, level: log_level = log_level.INFO) -> None:
    if level == log_level.DEBUG:
        print(f"[{timestamp()}] [DEBUG] {msg}")

    elif level == log_level.INFO:
        print(f"[{timestamp()}] [INFO] {msg}")

    elif level == log_level.WARNING:
        print(f"[{timestamp()}] [WARNING] {msg}")

    elif level == log_level.ERROR:
        print(f"[{timestamp()}] [ERROR] {msg}")

    elif level == log_level.CRITICAL:
        print(f"[{timestamp()}] [CRITICAL] {msg}")
