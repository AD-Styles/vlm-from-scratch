# Mini-LLaVA from Scratch

> CLIP-ViT 와 Qwen2.5 를 직접 조립해 만든 멀티모달 LLM. **결과의 완벽함보다 "한계 분석 → 다음 단계 도출" 의 반복 사이클** 을 기록한 포트폴리오입니다.

| | |
|---|---|
| **Backbone** | CLIP-ViT-B/32 + Qwen2.5-0.5B-Instruct |
| **학습 환경** | RTX 4060 Laptop · 8GB VRAM (단일 노트북) |
| **학습 가능 파라미터** | v1: 1.49M · v2: 3.66M (전체의 0.6%) |
| **학습 시간** | v1: 6분 43초 · v2: 47분 |
| **결과 요약** | 영문 VQA **4/5** ✅ · 한국어 ⚠️ catastrophic forgetting · OOD ⚠️ 환각 ([상세](#-결과-results)) |
| **레퍼런스** | LLaVA-1.5 (Liu et al., 2023) — 동일한 2-Stage 레시피의 mini 버전 (9K 샘플) |
| **사전 학습 가중치** | 🤗 [AD-Styles/mini-llava-stage2](https://huggingface.co/AD-Styles/mini-llava-stage2) (HuggingFace Hub) |
| **🚀 Live Demo** | [Hugging Face Spaces](https://huggingface.co/spaces/AD-Styles/mini-llava-demo) — 브라우저에서 즉시 체험 (설치 0) |


---

## 🧠 아키텍처 (Architecture)

LLaVA-1.5의 핵심 통찰: **거대 모델 두 개를 학습시키는 것이 아니라, 두 모달리티 간의 "통역사"(projector) + 작은 LoRA 만 학습.**

> CLIP의 patch grid: 입력 224×224 / patch 32px = 7 → **7×7 = 49 patch**.

```
   Image (224×224)              Text + <image> placeholder
        │                                  │
        ▼                                  ▼
   CLIP-ViT-B/32 (frozen)            Tokenizer + Embeds
        │ [49, 768]                        │ [L=text_len, 896]
        ▼                                  │
   ★ MLP Projector (학습)                  │
        │ [49, 896]                        │
        └────────┬─────────────────────────┘
                 ▼
   <image> 1개 → patch 49개로 splice  ←★ 직접 구현한 핵심 로직
                 │
                 ▼
   Qwen2.5-0.5B (frozen + ★ LoRA on q/k/v/o)
                 │
                 ▼
       "Dog. The dog is wearing a hat."
```

★ 표시가 직접 구현 (`src/model.py`). HuggingFace `LlavaForConditionalGeneration` 같은 고수준 추상화 미사용.

> **Stage 1 (v1)** = projector 만 학습 · **Stage 2 (v2)** = projector + LoRA 동시 학습 ([회고록 §Step 2 참조](#-회고록-retrospective))

---

## 📊 결과 (Results)

### v1 — Stage 1 Baseline (Projector Alignment)

**학습 설정:** Flickr30k 5K caption · projector 만 학습 · 1 epoch · lr 1e-3 · 6분 43초 · **최종 loss 2.4403**

**대표 응답 (강아지 사진):**

> Q: "What is in this image?" → A: "A black and white dog in a red frisbee stands on the beach."

**진단:** "dog" 키워드만 정확. 나머지(frisbee, beach, black) 모두 환각. 모델이 **Flickr30k 캡션 패턴(`A [person] in [clothes] is [verb]`)을 모방할 뿐, 질문에 응답하는 능력 부재.** LLaVA-1.5 논문 ([Liu et al., 2023, §4.1](https://arxiv.org/abs/2310.03744)) 의 "Stage 1 alignment limitations" 와 일치.

> 💭 왜 캡션 패턴만 모방하나? Stage 1 은 image embedding 을 LLM 공간에 "정렬" 만 하고 instruction-following 학습은 안 하기 때문. → Stage 2 가 필요한 이유 ([회고록 §Step 2 참조](#-회고록-retrospective))

---

### v2 — Stage 2 Instruction Tuning + LoRA

9,000 instruction 샘플(`localized_narratives` + `aokvqa` + `vqav2` 균형 믹스) 로 projector + LoRA 동시 학습.

#### Test A — 영문 VQA (강아지 사진, v1과 동일 입력)

<p align="center">
  <img src="assets/source_dog.jpg" width="220" alt="강아지 입력 이미지"><br>
  <em>입력 이미지 (Test A · B 공통)</em>
</p>

> 측정 환경: RTX 4060 Laptop GPU · `do_sample=True, T=0.7, top_p=0.9`

| 질문 | v2 응답 | 시간 |
|------|---------|------|
| What is in this image? | **Dog.** | 2.43s |
| What color is the dog? | **White.** ✅ | 0.47s |
| Is the dog wearing anything on its head? | **Yes.** ✅ | 0.46s |
| What is on the dog's head? | **Hat.** ✅ | 0.51s |
| Describe this image in one sentence. | "In this image I can see a cat on the floor." ⚠️ | 1.58s |

**🎯 핵심 발견 — Instruction Tuning 의 결정적 증거:**

v2는 **질문 형식에 따라 응답 포맷을 자동으로 바꿉니다** (단어 / 색상 / Yes-No / 객체 / 문장). 동일 입력의 v1 Q1 응답 — *"A black and white dog in a red frisbee stands on the beach."* — 같은 Flickr30k 캡션 패턴 모방과 명백한 대비. 시각적 정확도 **v1 0/1 (Q1 만 측정) → v2 4/5 (80%)**.

> 💭 v2 가 잘하는 이유: 학습 시 **assistant 응답 토큰만 loss** (instruction-only label masking) → 모델이 "질문에 답하는" 패턴 학습. v1 은 전체 caption 의 next-token loss → "Flickr30k 캡션 모방" 만 학습.

> Test A 의 마지막 행 (Q5 "Describe...") 에서의 "cat" 혼동: 헬로키티(고양이 캐릭터) 모자 패턴이 main object 인식에 영향. CLIP-ViT-B/32 의 49 patch (7×7) 해상도로는 강아지 얼굴 + 모자 위 고양이 얼굴이 모호해짐. **→ v3 의 CLIP-ViT-L/14 (576 patch) 업그레이드 motivation 의 직접적 증거** ([회고록 §Step 6 참조](#-회고록-retrospective)).

---

#### Test B — 한국어 (Catastrophic Forgetting 시연)

> 측정 환경 · 입력 이미지: Test A 와 동일 (강아지 사진).

| 질문 | v2 응답 | 평가 |
|------|---------|------|
| 이 이미지에 무엇이 보이나요? | "화이트의 소파, 물건." | ❌ 강아지를 "흰 소파"로 |
| 이 강아지는 무슨 색이에요? | "보통." | ❌ 의미 없음 |
| 이 강아지는 머리에 무엇을 쓰고 있나요? | **"개."** | 🔍 **분석의 결정적 증거** |
| 이 이미지를 한 문장으로 설명해 주세요. | "In this picture I can see a cat..." | ⚠️ 영어로 fallback |

**🔍 핵심 발견 — LoRA의 Catastrophic Forgetting 정량 입증:**

B3 응답 **"개."** 가 모델 내부를 그대로 보여줍니다:
- ✅ 시각 인식 작동 (dog)
- ✅ 한국어 키워드 인식 ("강아지")
- ❌ 영어 단답 편향이 한국어 표현으로 변환되며 **정답("모자")이 아닌 객체 카테고리("개")** 출력

> B1 ("화이트의 소파") / B2 ("보통") 도 같은 패턴: 시각 정보는 부분 인식하나 한국어 표현 매핑 깨짐. B4 는 모델이 한국어를 포기하고 영어로 fallback — catastrophic forgetting 의 역방향 시연.

원인: 학습 데이터 100% 영어 → LoRA가 영어 시각-언어 매핑만 강화 → base Qwen2.5의 한국어 능력이 부분 손상. **PEFT 사용 시 다국어 균형 데이터의 중요성**을 보여주는 정직한 결과. **→ v3 의 한국어 instruction 데이터 30%+ 추가 motivation** ([회고록 §Step 6 참조](#-회고록-retrospective)).

---

#### Test C — 피카츄 (OOD: 만화 캐릭터)

<p align="center">
  <img src="assets/source_pikachu.png" width="280" alt="피카츄 입력 이미지"><br>
  <em>입력 이미지 (OOD: 학습 분포 외부의 만화 캐릭터)</em>
</p>

> 측정 환경: Test A 와 동일 (RTX 4060 GPU · sampling).

| 질문 | v2 응답 | 정답 | 모델 내부 추론 (추정) |
|------|---------|------|---------------------|
| What is in this image? | "Giraffe." | Pikachu | 노랑+검정 패턴 → 학습 분포 중 가장 가까운 동물 |
| What color is the main character? | "White." | Yellow | Main subject 인식 실패 → 가장 두드러진 영역(흰 모자) |
| What is the character wearing on its head? | "Tie." | Hat | 공간 localization 실패 + 몸의 검은 띠를 넥타이로 |
| Describe this image in one sentence. | "In this image we can see a human figure..." | 만화 캐릭터 | 이족보행 + 팔 들기 자세 → 인간 형상으로 추상화 |

**🔍 핵심 발견 — "랜덤 환각이 아닌 체계적 오류":**

응답이 모두 틀렸지만, **각 응답에서 모델이 무엇을 보고 있는지** 가 드러납니다. v1 의 무관한 환각("man on motorcycle")과 달리, v2는 시각 특징(색상/패턴/자세)을 부분 인식하고 학습 분포 내 가장 가까운 클래스로 매핑합니다.

> 💭 왜 "체계적 오류"가 진보인가? v1 환각은 random — 디버깅 불가. v2 환각은 시각 특징 → 가까운 클래스 매핑 — 패턴 발견 가능 → **OOD detection module 설계 근거 제공**.

이는 VLM 분야의 **두 가지 본질적 문제**를 보여줍니다:
1. **CLIP-ViT-B/32 의 OOD 표현력 한계** — 만화/애니메이션 학습 데이터 부재
2. **VLM Hallucination Problem** — 모델이 "모른다"고 답하지 않고 "가장 가까운 답"을 만들어냄 (GPT-4V 까지 포함한 모든 VLM의 공통 문제)

**→ v3 의 두 가지 motivation 직결:** (1) **CLIP-ViT-L/14 (576 patch)** + OOD 데이터 augmentation, (2) **OOD detection module** ([회고록 §Step 6 참조](#-회고록-retrospective)).

---

## 💡 회고록 (Retrospective)

> 이 프로젝트는 **단발성 결과물이 아니라 6단계 의사결정 사이클** 의 기록입니다. 각 단계에서 어떤 한계를 발견하고, 어떤 옵션을 검토했고, 왜 그 선택을 했는지 정리합니다.

### Step 1 — 첫 시도: Stage 1 Alignment (v1)
**가설:** LLaVA-1.5 §3 의 핵심대로, projector 만 학습해도 시각-언어 정렬이 가능할 것이다.

**결과:** 학습 자체는 성공 (loss 2.44, 6분 43초), 그러나 응답이 **Flickr30k 캡션 패턴 모방에 그침**. "dog" 키워드만 잡고 나머지는 환각.

**얻은 것:** 멀티모달 융합 아키텍처 (`<image>` splice 로직, `inputs_embeds` 기반 generate, instruction-only label masking) 를 직접 구현하면서 LLaVA 의 내부 동작을 정확히 이해. **→ Stage 2 (Step 2 결정) 가 실제로 작동한 직접적 근거**: [Test A "왜 v2 잘하는 이유"](#-결과-results) 에서 instruction-only label masking 의 효과 입증.

### Step 2 — v1 한계 진단 + 다음 단계 결정
v1 결과를 분석하고 **3가지 옵션** 을 검토:

| 옵션 | 내용 | 결정 |
|------|------|------|
| A) 한계 인정 후 마무리 | 현재 수준으로 README 작성, "Stage 1 한계는 LLaVA 논문대로" 명시 | ❌ 단발 |
| B) 같은 데이터 더 학습 | epoch ↑, 데이터 ↑ — 단순 양적 증가 | ❌ 방법론적 진보 없음 |
| C) Stage 2 LoRA 추가 | LLaVA 정통 레시피 (instruction tuning) | ✅ **선택** |

**C 선택 이유:**
- **포트폴리오 시그널:** "데이터 더 부어봤네" 보다 "LLaVA 학습 레시피를 정확히 이해하고 재현했네" 가 채용 담당자에게 훨씬 매력적
- **이전 작업과의 연결성:** [unsloth-qlora-finetuning](https://github.com/AD-Styles/unsloth-qlora-finetuning) 의 LoRA 경험을 자연스럽게 확장
- **NCA-GENL 자격증 준비** 와 시너지

### Step 3 — v2 학습 중 발견한 데이터 함정
첫 시도로 VQAv2 단독을 사용했더니 답변의 **90.6% 가 10글자 미만** (Yes/No 위주). 이대로 학습하면 모델이 "Yes." / "No." 만 반복하게 됨.

**해결:** 3개 config 균형 믹스로 다양성 확보.

| Config | 비중 | 역할 |
|--------|------|------|
| `localized_narratives` | 33% | 긴 묘사 캡션 (캡셔닝 능력) |
| `aokvqa` | 33% | 추론 답변 (이해 능력) |
| `vqav2` | 33% | 짧은 사실 질문 (yes/no 자동 필터) |

→ 평균 답변 길이 **~5글자 → 77.8글자 (15배 향상)**. 이 데이터 진단 단계가 v2 성공의 결정적 분기점.

### Step 4 — v2 결과의 명과 암
- ✅ **명:** 영문 VQA 4/5 정확 (Test A) — instruction tuning 이 작동
- ⚠️ **암 1:** 한국어 catastrophic forgetting (Test B) — 학습 데이터 100% 영어의 부작용
- ⚠️ **암 2:** OOD 입력에 환각 여전 (Test C) — 그러나 "체계적 오류" 로 진화

세 가지 모두 **사전에 예측 가능했던 한계**. 모르고 당한 게 아니라 **트레이드오프를 인지하고 진행한 결정의 결과**.

### Step 5 — 배포 용이성 시도 (실패에서 배운 것)

학습 완료 후, 1GB adapter 를 GitHub 100MB 제한에 맞추기 위해 **순수 LoRA 추출** 을 시도했습니다.

**가설 (실패):** PEFT 가 저장한 `embed_tokens` / `lm_head` (총 ~1GB) 는 학습되지 않은 단순 보존용. 제거 후에도 inference 시 `resize_token_embeddings` 가 동적으로 재생성하므로 무영향. → 8.68 MB 슬림 adapter 로 충분할 것.

**실험:** [scripts/extract_lora.py](scripts/extract_lora.py) 로 LoRA 키 192개만 추출 (99.2% 축소).

**결과:** Test A 5문항 중 3문항이 명확히 다른 응답.

| 질문 | 원본 1GB | 슬림 8.68MB |
|------|---------|------------|
| What is in this image? | "Dog." ✅ | "Cat." ❌ |
| What color is the dog? | "White." ✅ | "Brown and black." ❌ |
| What is on the dog's head? | "Hat." ✅ | "Mittens." ❌ |

> Q3 (Yes/No) 와 Q5 (Describe) 는 슬림에서도 비슷한 응답 — Yes/No 답변의 단순성 + Describe 의 환각 양쪽 모두 그대로 유지. 즉 정확도 손실은 **fact-based 질문 (Q1/Q2/Q4) 에 집중**됨.

**가설 반증.** `embed_tokens` 는 단순 보존이 아니라 학습된 상태의 일부였습니다. 추정 원인:
- Qwen2.5 의 `tie_word_embeddings=True` → LoRA gradient 가 `lm_head` 를 거쳐 `embed_tokens` 까지 미세 영향
- 또는 PEFT 가 resize 감지 시 silent unfreeze
- 정확한 메커니즘 분석은 v3 의 과제

**얻은 교훈:** PEFT 의 `save_embedding_layers="auto"` 는 단순 저장 옵션이 아니다. 함부로 제거 시 모델 품질 손실.

**결정:** Hugging Face Hub 으로 1GB 그대로 배포 → 위의 [Pre-trained 가중치 링크](https://huggingface.co/AD-Styles/mini-llava-stage2) 참조.

### Step 6 — 다음으로 무엇을 할 것인가 (v3 로드맵)

1. **한국어 instruction 데이터 30%+ 추가** — KoLLaVA / KoVQA / DeepL 번역 → catastrophic forgetting 해소
2. **CLIP-ViT-L/14 (576 patches) 업그레이드** — 49 → 576 patch (16배 해상도) → 세부/OOD 인식 ↑
3. **OOD detection module** — CLIP similarity threshold 또는 entropy-based confidence → "모른다" 학습
4. **`tie_word_embeddings=False` 로 재학습** — Step 5의 가설 검증 + 슬림 adapter 재시도
5. **vLLM / Triton Inference Server 통합** — [nlp-triton-deployment](https://github.com/AD-Styles/nlp-triton-deployment) 와 연계, 프로덕션 서빙

이 5단계가 **v3 의 출발점**이 됩니다.

---

## ⚠️ 한계 (Limitations) — 정직한 한계 명시

| 한계 | 진단 | 해결 방향 |
|------|------|----------|
| **🔴 모델 한계** | | |
| 한국어 응답 약함 | LoRA의 catastrophic forgetting | 한국어 데이터 추가 학습 (Step 6-1) |
| OOD (만화) 환각 | CLIP-ViT-B/32 표현 한계 | ViT-L/14 업그레이드 + OOD detection (Step 6-2, 6-3) |
| Hallucination ("모른다" 못 함) | VLM 공통 문제 | Confidence calibration (Step 6-3) |
| LoRA adapter 1GB | embedding resize → PEFT가 embed_tokens 자동 저장 (학습 상태 포함) | HF Hub 로 배포 (Step 5 — 단순 분리는 품질 손실 확인됨) |
| **🟡 구현 한계** | | |
| 단일 이미지만 지원 | 구현 단순화 | Multi-image / video 확장 |
| 학습 resume 불가 | optimizer state 미저장 | accelerate 통합 |

---

## 🔗 참고 자료 (References)

- Liu et al., **"Visual Instruction Tuning"** (LLaVA-1, NeurIPS 2023) — [arxiv:2304.08485](https://arxiv.org/abs/2304.08485)
- Liu et al., **"Improved Baselines with Visual Instruction Tuning"** (LLaVA-1.5, 2023) — [arxiv:2310.03744](https://arxiv.org/abs/2310.03744)
- Radford et al., **"CLIP"** (ICML 2021) — [arxiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Qwen Team, **"Qwen2.5 Technical Report"** (Alibaba, 2024) — [arxiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Hu et al., **"LoRA: Low-Rank Adaptation"** (ICLR 2022) — [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Laurençon et al., **"What matters when building VLMs?"** (the_cauldron, 2024) — [arxiv:2405.02246](https://arxiv.org/abs/2405.02246)
