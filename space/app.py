"""Mini-LLaVA Gradio Demo for Hugging Face Spaces (v2 demo deployment).

이 파일은 HF Spaces 빌드 환경에서 자동 실행됩니다.
첫 실행 시 가중치를 HF Hub (AD-Styles/mini-llava-stage2) 에서 다운로드합니다 (~1 GB).
"""
from __future__ import annotations

import os

import gradio as gr
from huggingface_hub import snapshot_download
from PIL import Image

from src.config import GenerationConfig
from src.infer import VLMInference

WEIGHT_REPO = "AD-Styles/mini-llava-stage2"
WEIGHT_LOCAL = "checkpoints/v2_stage2_lora"

DEMO_BANNER_MD = """
# 🖼️ Mini-LLaVA — v2 Demo (Stage 2 LoRA)

> ⚠️ **Demo version (v2)** — 학습된 그대로의 모델입니다. 의도적으로 한계를 함께 공개합니다.
>
> | 잘하는 것 | 약한 것 (학습 데이터/구조 한계) |
> |----------|------------------------------|
> | 영문 단답 VQA (Dog / White / Yes / Hat 등) | 한국어 응답 (catastrophic forgetting) |
> | 영문 짧은 묘사 | 만화·애니 캐릭터 (OOD: 학습 분포 외부) |
>
> 🔮 v3 (한국어 데이터 + ViT-L/14 + OOD detection) 개발 중.
> 📖 자세한 분석: **[GitHub README](https://github.com/AD-Styles/vlm-from-scratch#-results)**
"""

FOOTER_MD = """
---
🔗 [GitHub Code](https://github.com/AD-Styles/vlm-from-scratch) · 🤗 [Model Weights](https://huggingface.co/AD-Styles/mini-llava-stage2)
· 김도윤 (AD-Styles) · 2026

> 💡 **CPU 환경**에서 동작하므로 응답에 5-15초 소요됩니다. 첫 응답은 모델 로드로 더 느립니다.
"""

EXAMPLES = [
    ["What is in this image?"],
    ["What color is the main subject?"],
    ["Describe this image in one sentence."],
    ["Is anything unusual in this image?"],
    ["이 이미지에 무엇이 보이나요?"],  # 한국어 limitation 시연용
]


def ensure_weights() -> tuple[str, str]:
    """HF Hub에서 가중치 다운로드 (첫 실행 1회). 이후는 캐시 재사용."""
    if not os.path.exists(WEIGHT_LOCAL):
        print(f"[init] Downloading weights from {WEIGHT_REPO} → {WEIGHT_LOCAL}")
        snapshot_download(
            repo_id=WEIGHT_REPO,
            local_dir=WEIGHT_LOCAL,
        )
        print(f"[init] Download complete.")
    else:
        print(f"[init] Weights already present at {WEIGHT_LOCAL}")

    projector = os.path.join(WEIGHT_LOCAL, "projector.pt")
    lora = os.path.join(WEIGHT_LOCAL, "lora_adapter")
    return projector, lora


# ─── Eager init at module load (Spaces best practice) ──────────────
print("[init] Preparing weights...")
PROJECTOR_PATH, LORA_PATH = ensure_weights()
print("[init] Building inference engine (CPU)...")
ENGINE = VLMInference(
    checkpoint_path=PROJECTOR_PATH,
    lora_adapter_path=LORA_PATH,
    device="cpu",
)
print("[init] Ready.")


def predict(
    image: Image.Image | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    if image is None:
        return "⚠️ 이미지를 먼저 업로드해 주세요.", ""
    if not question or not question.strip():
        return "⚠️ 질문을 입력해 주세요.", ""

    cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
    )
    result = ENGINE(image, question.strip(), gen_cfg=cfg)
    meta = (
        f"⏱️ {result['elapsed']:.2f}s · max_new={cfg.max_new_tokens} "
        f"· T={cfg.temperature} · top_p={cfg.top_p}"
    )
    return result["answer"], meta


with gr.Blocks(title="Mini-LLaVA Demo (v2)") as demo:
    gr.Markdown(DEMO_BANNER_MD)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="🖼️ 이미지 업로드", height=380)
            question_in = gr.Textbox(
                label="❓ 질문",
                placeholder="예: What is in this image?",
                lines=2,
            )
            gr.Examples(examples=EXAMPLES, inputs=[question_in], label="💡 예시 질문")

            with gr.Accordion("⚙️ 생성 옵션 (고급)", open=False):
                max_new_tokens = gr.Slider(
                    16, 512, value=128, step=16, label="max_new_tokens"
                )
                temperature = gr.Slider(
                    0.1, 1.5, value=0.7, step=0.05, label="temperature"
                )
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")

            submit_btn = gr.Button("🚀 응답 생성", variant="primary")

        with gr.Column(scale=1):
            answer_out = gr.Textbox(label="🤖 모델 응답", lines=12, interactive=False)
            meta_out = gr.Markdown("")

    submit_btn.click(
        fn=predict,
        inputs=[image_in, question_in, max_new_tokens, temperature, top_p],
        outputs=[answer_out, meta_out],
    )
    question_in.submit(
        fn=predict,
        inputs=[image_in, question_in, max_new_tokens, temperature, top_p],
        outputs=[answer_out, meta_out],
    )

    gr.Markdown(FOOTER_MD)


if __name__ == "__main__":
    demo.launch()
