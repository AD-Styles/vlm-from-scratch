---
license: mit
language:
  - en
  - ko
library_name: peft
tags:
  - vision-language-model
  - multimodal
  - lora
  - llava
  - mini-llava
  - vlm
base_model:
  - Qwen/Qwen2.5-0.5B-Instruct
  - openai/clip-vit-base-patch32
pipeline_tag: image-to-text
---

# Mini-LLaVA — Stage 2 LoRA Weights

LLaVA-1.5 의 핵심 아키텍처를 처음부터 직접 구현한 멀티모달 LLM 의 학습된 가중치.
HuggingFace 의 `LlavaForConditionalGeneration` 같은 고수준 추상화 미사용, 융합 로직 직접 구현.

🚀 **Live Demo:** https://huggingface.co/spaces/AD-Styles/mini-llava-demo (설치 없이 즉시 체험)
📂 **코드 레포:** https://github.com/AD-Styles/vlm-from-scratch
📝 **상세 분석 (v1→v2 회고록 포함):** [GitHub README](https://github.com/AD-Styles/vlm-from-scratch#readme)
📊 **Test A/B/C 결과 표:** [GitHub README #-results](https://github.com/AD-Styles/vlm-from-scratch#-results)

## 🧩 구성

| 파일 | 크기 | 역할 |
|------|------|------|
| `projector.pt` | 5.7 MB | CLIP-ViT-B/32 → Qwen2.5 임베딩 공간 매핑 (2-layer MLP) |
| `lora_adapter/adapter_config.json` | 1 KB | PEFT LoRA 설정 |
| `lora_adapter/adapter_model.safetensors` | ~1 GB | LoRA r=16 가중치 + embed_tokens / lm_head (학습 시 변경분 포함) |

> **왜 1GB?** PEFT 가 `<image>` 특수 토큰 추가로 인한 embedding resize 를 감지해 `embed_tokens` / `lm_head` 를 함께 저장. 이를 단순 분리 시도했으나 (코드 레포 [scripts/extract_lora.py](https://github.com/AD-Styles/vlm-from-scratch/blob/main/scripts/extract_lora.py)), 5문항 중 3문항이 완전히 다른 응답으로 변하면서 **embed_tokens 가 학습된 상태의 일부임을 실험으로 확인**. 따라서 원본 1GB 그대로 배포.

## 🚀 사용법

```bash
# 1. 코드 + 환경 준비
git clone https://github.com/AD-Styles/vlm-from-scratch
cd vlm-from-scratch
pip install --upgrade "torch>=2.6.0" "torchvision>=0.21.0" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. 이 레포에서 가중치 다운로드 (~1 GB)
huggingface-cli download AD-Styles/mini-llava-stage2 --local-dir checkpoints/v2_stage2_lora

# 3. Gradio 데모 실행
python app.py \
  --checkpoint checkpoints/v2_stage2_lora/projector.pt \
  --lora-adapter checkpoints/v2_stage2_lora/lora_adapter
```

브라우저에서 `http://localhost:7860` 접속 → 이미지 업로드 → 자연어 질문.

## 📊 학습 설정

| 항목 | 값 |
|------|-----|
| Base 모델 | `openai/clip-vit-base-patch32` (frozen) + `Qwen/Qwen2.5-0.5B-Instruct` (frozen + LoRA) |
| 데이터 | `HuggingFaceM4/the_cauldron` 3-config 믹스 (localized_narratives + aokvqa + vqav2), 9,000 샘플 |
| LoRA | rank=16, alpha=32, target=q/k/v/o projection |
| Epochs | 2 |
| Batch size | 2 (grad_accum=4 → effective 8) |
| Learning rate | 2e-4 (cosine) |
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| 학습 시간 | 47 분 |
| 학습 가능 파라미터 | 3,655,424 (전체의 0.66%) |
| Init | v1 baseline projector 이어 학습 (Stage 1 → Stage 2) |

## ✅ 검증 결과 (영문 VQA, 강아지 사진)

| 질문 | 응답 | 정확도 |
|------|------|--------|
| What is in this image? | "Dog." | ✅ |
| What color is the dog? | "White." | ✅ |
| Is the dog wearing anything on its head? | "Yes." | ✅ |
| What is on the dog's head? | "Hat." | ✅ |
| Describe this image in one sentence. | "...cat on the floor." | ⚠️ (헬로키티 모자 영향) |

**Test A: 4/5 정확**, instruction format 자동 매칭 성공.

## ⚠️ 한계 (정직한 명시)

- **한국어:** LoRA의 catastrophic forgetting — 학습 데이터 100% 영어. "이 강아지는 머리에 무엇을 쓰고 있나요?" → "개." (영어 단답 편향이 한국어 표현으로 잘못 변환)
- **OOD (만화/애니메이션):** CLIP-ViT-B/32 표현 한계. 피카츄 → "Giraffe" (학습 분포 내 가장 가까운 동물로 매핑)
- **Hallucination:** "모른다" 답변 못 함 (VLM 공통 문제)

자세한 분석은 [GitHub README의 Test B/C 섹션](https://github.com/AD-Styles/vlm-from-scratch#-results) 참조.

## 🔮 향후 개선 (v3 로드맵)

1. 한국어 instruction 데이터 30%+ 추가 → catastrophic forgetting 해소
2. CLIP-ViT-L/14 (576 patches) 업그레이드 → OOD 견고성 ↑
3. OOD detection module → "모른다" 답변 학습
4. vLLM / Triton Inference Server 통합 → 프로덕션 서빙

## 📚 References

- LLaVA-1.5 (Liu et al., 2023) — [arxiv:2310.03744](https://arxiv.org/abs/2310.03744)
- CLIP (Radford et al., 2021) — [arxiv:2103.00020](https://arxiv.org/abs/2103.00020)
- LoRA (Hu et al., 2022) — [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
- the_cauldron (Laurençon et al., 2024) — [arxiv:2405.02246](https://arxiv.org/abs/2405.02246)

## License

MIT — 자유롭게 사용 / 수정 / 배포 가능.

---

🤖 김도윤 (AD-Styles) · 2026
