from dataclasses import dataclass


VISION_MODEL = "openai/clip-vit-base-patch32"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "You receive an image and a question, and answer concisely and accurately."
)


@dataclass
class TrainConfig:
    data_path: str = "data/coco_subset/manifest.json"
    output_dir: str = "checkpoints/v1_baseline"

    batch_size: int = 8
    grad_accum_steps: int = 1
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_text_length: int = 512

    log_every: int = 20
    save_every: int = 500
    seed: int = 42

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Stage 2 → Stage 1 projector 이어받기용 (선택)
    init_projector: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
