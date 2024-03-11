from enum import Enum


class Whisper_Model(str, Enum):
    # tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2
    tiny_en = "tiny.en"
    tiny = "tiny"
    base_en = "base.en"
    base = "base"
    small_en = "small.en"
    small = "small"
    medium_en = "medium.en"
    medium = "medium"
    large_v1 = "large-v1"
    large_v2 = "large-v2"


class Device(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
