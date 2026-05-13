"""Stage 1 학습 — projector만 학습하여 시각 특징을 LLM 임베딩 공간으로 정렬.

사용 예:
  python -m src.train \\
    --data-path data/coco_subset/manifest.json \\
    --output-dir checkpoints/stage1 \\
    --batch-size 8 --epochs 1 --lr 1e-3
"""
from __future__ import annotations

import argparse
import math
import os
import random

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import VQACollator, VQADataset
from .model import MiniLLaVA


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr_lambda(total_steps: int, warmup_steps: int):
    def fn(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def maybe_apply_lora(model: MiniLLaVA, cfg: TrainConfig):
    """Stage 2: 기존 projector는 그대로 학습 가능 + LLM에 LoRA 어댑터 추가."""
    if not cfg.use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    # PEFT 가 base LLM을 자동 freeze. projector는 외부라 trainable 유지.
    return model


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/stage1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-text-length", type=int, default=512)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-lora", action="store_true",
                   help="Stage 2: LoRA adapter on LLM + projector 동시 학습")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--init-projector", type=str, default=None,
                   help="기존 projector ckpt에서 시작 (Stage 1 → Stage 2 이어 학습)")
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[init] loading MiniLLaVA ...")
    model = MiniLLaVA(freeze_vision=True, freeze_llm=not cfg.use_lora)
    if cfg.init_projector and os.path.exists(cfg.init_projector):
        print(f"[init] loading existing projector → {cfg.init_projector}")
        model.load_projector(cfg.init_projector, map_location="cpu")
    model = maybe_apply_lora(model, cfg)
    model.to(device)
    print(f"[init] trainable params: {model.num_trainable():,}")

    print(f"[data] loading {cfg.data_path}")
    dataset = VQADataset(
        cfg.data_path, model.tokenizer, model.image_processor, cfg.max_text_length
    )
    collator = VQACollator(pad_token_id=model.tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    print(f"[data] {len(dataset)} samples, {len(loader)} batches/epoch")

    optimizer = AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    total_steps = (len(loader) // cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = LambdaLR(optimizer, cosine_lr_lambda(total_steps, warmup_steps))

    global_step = 0
    model.train()
    if hasattr(model, "vision"):
        model.vision.eval()

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running_loss / (cfg.log_every * cfg.grad_accum_steps)
                    pbar.set_postfix(
                        loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0

                if global_step % cfg.save_every == 0:
                    ckpt = os.path.join(
                        cfg.output_dir, f"projector_step{global_step}.pt"
                    )
                    model.save_projector(ckpt)

    final_path = os.path.join(cfg.output_dir, "projector.pt")
    model.save_projector(final_path)
    print(f"[done] saved → {final_path}")

    if cfg.use_lora:
        lora_dir = os.path.join(cfg.output_dir, "lora_adapter")
        model.llm.save_pretrained(lora_dir)
        print(f"[done] saved LoRA → {lora_dir}")


if __name__ == "__main__":
    main()
