from pathlib import Path
from typing import Literal

CLASS_TYPE = Literal["background", "drone", "helicopter"]
CLASSES: list[CLASS_TYPE] = ["background", "drone", "helicopter"]

RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)


def get_split_dir(split: Literal["train", "test"]) -> Path:
    return RAW_DATA_DIR / split


def get_clz_dir(split: Literal["train", "test"], clz: CLASS_TYPE) -> Path:
    return get_split_dir(split) / clz
