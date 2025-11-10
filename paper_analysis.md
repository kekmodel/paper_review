# 논문 비교 분석: GLM-4.5, MiniMax-M1, Kimi K2

## 📋 개요

이 문서는 세 가지 최첨단 추론 모델에 대한 심층 비교 분석입니다:

1. **GLM-4.5** (Zhipu AI & Tsinghua University)
2. **MiniMax-M1** (MiniMax)
3. **Kimi K2** (Moonshot AI)

---

## 1️⃣ 각 논문 상세 분석

### GLM-4.5: Agentic, Reasoning, and Coding Foundation Models

**기본 스펙:**
- **파라미터**: 355B total, 32B activated (MoE)
- **컨텍스트 길이**: 128K tokens
- **공개 상태**: Open-source

**핵심 기여:**

1. **아키텍처 혁신**
   - Loss-free balance routing과 sigmoid gates를 사용한 MoE
   - 더 깊은 레이어 설계로 추론 능력 향상
   - 96개의 attention heads (경쟁 모델보다 많음)
   - QK-Norm으로 attention 안정성 확보
   - Multi-Token Prediction을 통한 speculative decoding 지원

2. **3단계 훈련 접근법**
   - **Pre-training**: 23T tokens (web, code, math, science)
   - **Mid-training**: Repository-level code, synthetic reasoning, long-context/agent training
   - **Post-training**: Expert model iteration with self-distillation

3. **Post-training 기법**
   - **Function Call Template**: XML-like 태그 형식으로 코드 이스케이핑 문제 해결
   - **Reasoning RL**: Difficulty-based curriculum learning
   - **Agentic RL**: Step-wise rule-based + end-to-end multi-turn approaches
   - **Custom "Slime" Framework**: Sync/async RL 모드 지원

**성능 하이라이트:**
- AIME 24: 91.0%
- TAU-Bench: 70.1%
- SWE-bench Verified: 64.2%
- BrowseComp (web browsing): 26.4% (Claude Opus 4 능가)
- CC-Bench (tool-calling): 90.6%

**특징:**
- Hybrid reasoning: "thinking mode" vs "direct response mode"
- DeepSeek-R1의 절반 파라미터, Kimi K2의 1/3 파라미터로 효율적
- 강력한 다국어 능력 (전문 번역 모델 능가)

---

### MiniMax-M1: Scaling Test-Time Compute Efficiently

**기본 스펙:**
- **파라미터**: 456B total, 45.9B activated (MoE)
- **컨텍스트 길이**: 1M tokens (DeepSeek R1의 8배)
- **공개 상태**: Open-weight

**핵심 기여:**

1. **하이브리드 아키텍처**
   - Linear attention (transnormer blocks) + 주기적 softmax attention
   - Lightning attention으로 극도로 긴 컨텍스트 지원
   - 100K 토큰 생성 시 DeepSeek R1 대비 25% FLOPs만 사용

2. **CISPO RL 알고리즘**
   - PPO/GRPO와 달리 토큰 레벨 업데이트가 아닌 importance sampling weights를 클립
   - 희귀한 추론 토큰 손실 방지 → chain-of-thought 패턴 보존

3. **훈련 파이프라인**
   - 7.5T tokens continual pretraining (STEM/reasoning 강조)
   - Chain-of-thought SFT
   - Large-scale RL with diverse environments

4. **데이터 전략**
   - 수학 (50K samples) - rule-based verification
   - 논리 추론 (53K) - SynLogic framework
   - 경쟁 프로그래밍 (30K) - online judge
   - 소프트웨어 엔지니어링 - sandboxed execution
   - 일반 도메인 - generative reward models

5. **기술적 해결책**
   - LM output head를 FP32로 올려 train/inference precision mismatch 해결
   - 토큰 확률 상관성 0.9x → 0.99x 개선

**성능 하이라이트:**
- AIME 2024: 86.0%
- SWE-bench: 56.0%
- TAU-bench: Gemini 2.5 Pro 능가
- Long-context: OpenAI o3, Claude 4 Opus 능가

**특징:**
- 3주 훈련 (512 H800 GPUs, ~$534K)
- 순수 수학/코딩에서는 DeepSeek-R1보다 약하나 실세계 extended reasoning에서 우수

---

### Kimi K2: Open Agentic Intelligence

**기본 스펙:**
- **파라미터**: 1.04T total, 32B activated (MoE)
- **컨텍스트 길이**: 128K tokens (4K → YaRN으로 확장)
- **공개 상태**: Open-source (base + instruct)

**핵심 기여:**

1. **MuonClip Optimizer**
   - Novel weight-clipping mechanism (QK-Clip)
   - Attention logits를 명시적으로 제약하여 훈련 안정성 확보
   - Query/key projection weights를 rescale하여 폭발적 성장 방지
   - 15.5T tokens을 loss spike 없이 pre-train 성공

2. **Pre-training 혁신**
   - **Token Efficiency**:
     - Knowledge domain rephrasing (style-diverse prompting + fidelity verification)
     - Mathematics를 learning-note style로 변환
     - Chunk-wise autoregressive generation으로 문서 일관성 유지

   - **Ultra-sparse MoE**:
     - 384 total experts, sparsity=48 (DeepSeek-V3의 256 experts 대비)
     - Attention heads 128→64로 줄여 long-context inference 효율성 증가
     - Scaling law 분석: 고정 compute budget에서 sparsity 증가가 유리

3. **Agentic Data Synthesis Pipeline**
   - Domain evolution: 20,000+ synthetic tools + 3,000+ real MCP tools
   - Agent diversification: 다양한 system prompts & tool combinations
   - Rubric-based task generation with explicit success criteria
   - Multi-turn trajectory with user simulation
   - Hybrid simulated + real execution sandboxes

4. **Advanced RL Framework**
   - **Verifiable Rewards Gym**:
     - Math/STEM (difficulty-balanced)
     - Complex instruction following (hybrid rule verification)
     - Faithfulness (sentence-level judges)
     - Coding/SWE (real sandbox testing)
     - Safety (adversarial prompt evolution)

   - **Self-Critique Mechanism**:
     - 모델이 자체 출력을 rubric-based pairwise comparison으로 평가
     - 외부 보상 신호 없이 주관적 선호도 정렬

5. **Infrastructure**
   - 16-way Pipeline Parallelism + Expert Parallelism + ZeRO-1
   - Interleaved 1F1B scheduling으로 통신 오버랩
   - Activation reduction (selective recomputation, FP8 compression, CPU offloading)
   - Colocated RL architecture (sync training+inference engines)

**성능 하이라이트:**
- **Agentic (SOTA)**:
  - Tau2-Bench: 66.1 Pass@1
  - ACEBench: 76.5
  - SWE-Bench Verified (agentic): 65.8%
- **Coding**:
  - LiveCodeBench v6: 53.7 Pass@1
  - OJBench: 27.1 Pass@1
- **Math**:
  - AIME 2025: 49.5 (Avg@64)
  - GPQA-Diamond: 75.1 (Avg@8)
- **General**:
  - MMLU: 89.5
  - IFEval: 89.8
  - LMSYS Arena: 5th overall, top open-source

**특징:**
- 가장 큰 모델 (1.04T total params)
- Ultra-sparse MoE로 activated params는 32B로 유지
- Agentic 능력에 특화된 설계

---

## 2️⃣ 공통점 (Commonalities)

### 아키텍처 & 스케일

1. **Mixture-of-Experts (MoE) 아키텍처**
   - 세 모델 모두 MoE 사용으로 파라미터 효율성 극대화
   - Activated parameters: 32B~45.9B로 유사한 범위
   - Total parameters: 355B~1.04T로 대규모

2. **긴 컨텍스트 지원**
   - GLM-4.5: 128K tokens
   - MiniMax-M1: 1M tokens (최장)
   - Kimi K2: 128K tokens
   - 모두 extended reasoning과 agentic tasks 지원

### 훈련 방법론

3. **3단계 훈련 파이프라인**
   - Pre-training → Mid/Continual training → Post-training
   - 모두 대규모 토큰 (7.5T~23T tokens) 사용

4. **Reinforcement Learning 중심 Post-training**
   - GLM-4.5: Reasoning RL + Agentic RL with curriculum learning
   - MiniMax-M1: CISPO (novel clipping strategy)
   - Kimi K2: Verifiable Rewards Gym + Self-Critique
   - 모두 다양한 도메인 (math, coding, reasoning, tool-use)에 대한 RL 적용

5. **합성 데이터 (Synthetic Data) 활용**
   - GLM-4.5: Synthetic reasoning data in mid-training
   - MiniMax-M1: SynLogic framework for logical reasoning (53K samples)
   - Kimi K2: Agentic data synthesis pipeline (20,000+ tools)

### 목표 능력

6. **ARC 트라이펙타**: Agentic + Reasoning + Coding
   - 세 논문 모두 이 세 영역을 핵심 능력으로 강조
   - Tool-calling, web browsing, software engineering 벤치마크 중시

7. **오픈소스 철학**
   - GLM-4.5: Open-source (Hugging Face)
   - MiniMax-M1: Open-weight
   - Kimi K2: Open-source (base + instruct)
   - 모두 연구 커뮤니티 기여를 목표

### 기술적 솔루션

8. **Attention 안정성 개선**
   - GLM-4.5: QK-Norm
   - Kimi K2: QK-Clip (MuonClip의 일부)
   - Attention logits 폭발 방지가 공통 과제

9. **Tool-use & Function Calling**
   - 모두 structured function calling 지원
   - GLM-4.5: XML-like template
   - Kimi K2: 3,000+ real MCP tools 통합
   - Real-world agentic applications 중시

10. **Hybrid Reasoning Strategies**
    - GLM-4.5: Thinking mode vs direct response
    - MiniMax-M1: Test-time compute scaling
    - Kimi K2: Self-critique mechanism
    - 작업에 따라 추론 깊이 조정

---

## 3️⃣ 차이점 (Differences)

### 아키텍처 디자인

| 측면 | GLM-4.5 | MiniMax-M1 | Kimi K2 |
|------|---------|------------|---------|
| **Total Params** | 355B | 456B | 1.04T |
| **Activated** | 32B | 45.9B | 32B |
| **Experts** | ? | ? | 384 (sparsity=48) |
| **Attention Heads** | 96 (많음) | ? | 64 (효율성) |
| **Attention Type** | Standard + QK-Norm | Hybrid (Linear + Softmax) | Standard + QK-Clip |
| **Context Length** | 128K | 1M (최장) | 128K |

**핵심 차별점:**

1. **MiniMax-M1의 Hybrid Attention**
   - Linear attention (transnormer) + periodic softmax
   - 100K 토큰 생성 시 DeepSeek R1 대비 25% FLOPs
   - 1M context length로 독보적

2. **Kimi K2의 Ultra-Sparse MoE**
   - 384 experts, sparsity=48 (가장 희소)
   - Scaling law: 고정 compute에서 sparsity↑ = 성능↑
   - Attention heads를 줄여 inference 효율성 우선

3. **GLM-4.5의 Deeper & More Heads**
   - 더 많은 레이어 (구체적 수치 미제공)
   - 96 attention heads로 복잡한 패턴 캡처
   - 파라미터 대비 가장 효율적 (355B)

### Optimizer & 훈련 안정성

| 모델 | Optimizer | 핵심 혁신 |
|------|-----------|----------|
| GLM-4.5 | ? | Mid-training 단계 추가 (repo-level code, long-context) |
| MiniMax-M1 | Standard | **FP32 LM head** (precision mismatch 해결) |
| Kimi K2 | **MuonClip** | **QK-Clip weight clipping** (15.5T tokens loss spike 없음) |

**핵심 차별점:**

- **Kimi K2**: Muon optimizer의 스케일링 문제를 QK-Clip으로 해결 → 15.5T tokens 안정적 훈련
- **MiniMax-M1**: Train/inference kernel precision mismatch를 FP32 head로 해결 → 토큰 확률 상관성 0.9→0.99
- **GLM-4.5**: Mid-training 단계로 domain-specific capabilities 강화

### Reinforcement Learning 접근

| 모델 | RL 알고리즘 | 특징 |
|------|------------|------|
| GLM-4.5 | Custom "Slime" Framework | Difficulty-based curriculum, step-wise + end-to-end agentic RL |
| MiniMax-M1 | **CISPO** | **Importance sampling weight clipping** (토큰 업데이트 아님) |
| Kimi K2 | Verifiable Rewards Gym | **Self-critique** (external reward 없이 자체 평가) |

**핵심 차별점:**

1. **CISPO (MiniMax-M1)**
   - PPO/GRPO는 토큰 레벨 업데이트를 클립 → 희귀 추론 토큰 손실
   - CISPO는 importance sampling weights 클립 → CoT 패턴 보존
   - Extended reasoning에 최적화

2. **Self-Critique (Kimi K2)**
   - 모델이 자신의 출력을 rubric-based로 평가
   - 주관적 선호도 정렬에 강점
   - Human feedback 의존도 감소

3. **Curriculum Learning (GLM-4.5)**
   - Difficulty-based progression
   - Step-wise rule-based + end-to-end 결합
   - Iterative self-distillation

### 데이터 전략

| 모델 | Pre-training Tokens | 데이터 특화 전략 |
|------|---------------------|-----------------|
| GLM-4.5 | 23T | Quality-based sampling, mid-training with repo-level code |
| MiniMax-M1 | 7.5T | STEM/reasoning 강조, SynLogic 논리 추론 |
| Kimi K2 | 15.5T | **Knowledge rephrasing**, **math→learning-note**, chunk-wise autoregressive |

**핵심 차별점:**

1. **Kimi K2의 Token Efficiency 혁신**
   - Knowledge domain rephrasing: style-diverse prompting + fidelity verification
   - Mathematics를 learning-note style로 변환
   - Chunk-wise autoregressive로 문서 일관성 유지
   - → 적은 토큰으로 높은 품질

2. **MiniMax-M1의 도메인 특화**
   - 50K math with rule-based verification
   - 53K logical reasoning (SynLogic)
   - 30K competitive programming
   - Sandboxed execution으로 실행 가능성 검증

3. **GLM-4.5의 Mid-training**
   - Repository-level code training
   - Long-context extension to 128K
   - Agent-specific capabilities
   - Post-training 전 전문화 단계

### Agentic Data & Tools

| 모델 | Tool 수 | Agentic Data 생성 |
|------|---------|-------------------|
| GLM-4.5 | ? | Function call template (XML-like) |
| MiniMax-M1 | ? | Sandboxed execution environments |
| Kimi K2 | **20,000+ synthetic + 3,000+ MCP** | **Domain evolution, agent diversification, rubric-based** |

**핵심 차별점:**

- **Kimi K2**: 가장 체계적인 agentic data pipeline
  - Domain evolution으로 20,000+ tools 생성
  - Real MCP tools 3,000+ 통합
  - User simulation + multi-turn trajectories
  - Hybrid simulated/real execution

- **GLM-4.5**: XML-like function call template으로 코드 이스케이핑 문제 해결

- **MiniMax-M1**: Sandboxed execution으로 SWE 작업 검증

### 성능 프로필

#### AIME (수학 올림피아드)

- GLM-4.5: **91.0%** (AIME 24) ← 최고
- MiniMax-M1: 86.0% (AIME 24)
- Kimi K2: 49.5 Avg@64 (AIME 25) ← 더 어려운 버전

#### SWE-bench (소프트웨어 엔지니어링)

- GLM-4.5: 64.2% (Verified)
- Kimi K2: **65.8%** (Verified, agentic) ← 최고
- MiniMax-M1: 56.0%

#### TAU-bench (Tool-use)

- GLM-4.5: 70.1%
- Kimi K2: **66.1%** (Tau2-Bench)
- MiniMax-M1: Gemini 2.5 Pro 능가 (구체적 수치 미제공)

#### Long-context

- MiniMax-M1: **OpenAI o3, Claude 4 Opus 능가** ← 최고 (1M context)
- GLM-4.5: 128K context
- Kimi K2: 128K context

**성능 트레이드오프:**

1. **GLM-4.5**: Pure math에서 최강, 파라미터 대비 가장 효율적
2. **MiniMax-M1**: Long-context & extended reasoning에서 최강, FLOPs 효율성
3. **Kimi K2**: Agentic tasks & SWE에서 최강, 가장 큰 모델

### Infrastructure & 훈련 비용

| 모델 | 훈련 인프라 | 비용/시간 |
|------|-------------|----------|
| GLM-4.5 | Custom "Slime" framework | ? |
| MiniMax-M1 | 512 H800 GPUs | **3주, ~$534K** |
| Kimi K2 | 16-way PP + EP + ZeRO-1 | ? |

**핵심 차별점:**

- **MiniMax-M1**: 훈련 비용 투명성 (3주, $534K로 SOTA 근접 가능 입증)
- **Kimi K2**: 가장 정교한 distributed training setup
  - Interleaved 1F1B scheduling
  - Activation reduction (FP8, CPU offload)
  - Colocated RL architecture
- **GLM-4.5**: Custom RL framework (Slime) with sync/async modes

### Function Calling 설계

| 모델 | Function Call 형식 | 특징 |
|------|-------------------|------|
| GLM-4.5 | **XML-like tags** | 코드 세그먼트에서 character escaping 불필요 |
| MiniMax-M1 | ? | ? |
| Kimi K2 | Standard | 3,000+ real MCP tools 통합 |

**핵심 차별점:**

- **GLM-4.5**: XML-like template으로 엔지니어링 문제 해결 (코드 이스케이핑)
- **Kimi K2**: 실제 MCP (Model Context Protocol) tools 대규모 통합

---

## 4️⃣ 배울 점 (Key Learnings)

### 아키텍처 설계

**1. MoE Sparsity vs Activated Params 트레이드오프**

- **Kimi K2의 교훈**:
  - Ultra-sparse MoE (384 experts, sparsity=48)로 1.04T total params를 32B activated로 유지
  - Scaling law: 고정 compute budget에서 sparsity 증가가 더 나은 성능
  - **배울 점**: 더 많은 experts + 더 높은 sparsity = 파라미터 효율성

- **GLM-4.5의 교훈**:
  - 355B total, 32B activated로 Kimi K2의 1/3 크기로 경쟁적 성능
  - Deeper layers + more attention heads (96)로 복잡도 증가
  - **배울 점**: 모델 깊이와 attention head 수도 중요한 설계 변수

**2. Attention Mechanism 혁신**

- **MiniMax-M1의 교훈**:
  - Hybrid linear + softmax attention으로 1M context 달성
  - 100K 토큰 생성 시 25% FLOPs만 사용
  - **배울 점**: Linear attention은 long-context에서 computational efficiency의 게임 체인저

- **Kimi K2의 교훈**:
  - Attention heads 128→64로 줄여 long-context inference 효율성 증가
  - QK-Clip으로 attention logits 폭발 방지
  - **배울 점**: Attention heads는 많을수록 좋은 것이 아님 (inference 효율성 고려)

**3. Context Length 전략**

- **MiniMax-M1**: 1M tokens (최장) → long-context reasoning에서 SOTA
- **GLM-4.5 & Kimi K2**: 128K tokens → 대부분의 작업에 충분
- **배울 점**: 1M context는 특정 use case (complex agentic tasks, extended reasoning)에서만 필요

### Optimizer & 훈련 안정성

**4. Optimizer 혁신의 중요성**

- **Kimi K2의 MuonClip**:
  - Muon optimizer의 스케일링 문제를 QK-Clip으로 해결
  - 15.5T tokens을 loss spike 없이 훈련
  - **배울 점**: Novel optimizer는 대규모 훈련의 안정성과 효율성을 크게 향상

**5. Precision Engineering**

- **MiniMax-M1의 FP32 LM Head**:
  - Train/inference kernel precision mismatch 해결
  - 토큰 확률 상관성 0.9→0.99 개선
  - **배울 점**: 작은 engineering detail (FP32 head)이 실제 성능에 큰 영향

### Reinforcement Learning 설계

**6. RL 알고리즘 선택의 중요성**

- **MiniMax-M1의 CISPO**:
  - PPO/GRPO의 토큰 레벨 클리핑은 희귀 추론 토큰을 손실
  - Importance sampling weight 클리핑으로 CoT 패턴 보존
  - **배울 점**: RL 알고리즘은 reasoning task의 특성에 맞춰 설계해야 함

**7. Self-Critique & Verifiable Rewards**

- **Kimi K2의 Self-Critique**:
  - External reward 없이 자체 평가로 주관적 선호도 정렬
  - Rubric-based pairwise comparison
  - **배울 점**: Self-critique는 human feedback scaling 문제의 해법

- **Kimi K2의 Verifiable Rewards Gym**:
  - Math/STEM: Rule-based verification
  - Coding: Sandbox testing
  - Safety: Adversarial prompt evolution
  - **배울 점**: 도메인별로 적절한 verification method 필요

**8. Curriculum Learning**

- **GLM-4.5의 Difficulty-based Curriculum**:
  - Easy → hard progression
  - Two-stage reasoning RL
  - **배울 점**: Curriculum learning은 complex reasoning 학습에 효과적

### 데이터 전략

**9. Token Efficiency vs Token Count**

- **Kimi K2의 혁신**:
  - 15.5T tokens (MiniMax 7.5T, GLM 23T 사이)
  - Knowledge rephrasing, math→learning-note 변환으로 품질 향상
  - **배울 점**: 더 많은 토큰보다 더 나은 토큰 품질 (rephrasing, transformation)

**10. Synthetic Data의 전략적 활용**

- **GLM-4.5**: Mid-training에 synthetic reasoning data
- **MiniMax-M1**: SynLogic framework로 53K logical reasoning samples
- **Kimi K2**: 20,000+ synthetic tools + agentic trajectories
- **배울 점**: Synthetic data는 단순 augmentation이 아닌 strategic capability building

**11. Repository-level vs File-level Code Training**

- **GLM-4.5의 Mid-training**:
  - Repository-level code training
  - **배울 점**: Repo-level context는 real-world coding task에 필수

### Agentic Capabilities

**12. Tool Ecosystem 구축**

- **Kimi K2의 접근**:
  - 20,000+ synthetic tools (domain evolution)
  - 3,000+ real MCP tools 통합
  - **배울 점**: 대규모 tool ecosystem은 agentic capabilities의 foundation

**13. Agentic Data Synthesis Pipeline**

- **Kimi K2의 체계적 접근**:
  - Domain evolution → Agent diversification → Rubric-based task → Multi-turn trajectory
  - User simulation + hybrid execution (simulated + real)
  - **배울 점**: Agentic data는 단순 데이터셋이 아닌 복잡한 pipeline 필요

**14. Function Call Template 설계**

- **GLM-4.5의 XML-like Template**:
  - Character escaping 문제 해결
  - Code segments에서 특히 유용
  - **배울 점**: Function call format은 실용성 (escaping, parsing)도 중요

### Infrastructure & Engineering

**15. Distributed Training 최적화**

- **Kimi K2의 Infrastructure**:
  - 16-way PP + EP + ZeRO-1
  - Interleaved 1F1B scheduling으로 communication overlap
  - Activation reduction (selective recomputation, FP8, CPU offload)
  - **배울 점**: Trillion-scale 모델은 극도로 정교한 parallelism strategy 필요

**16. RL Infrastructure**

- **Kimi K2의 Colocated Architecture**:
  - Synchronized training + inference engines
  - Distributed checkpoint engine
  - Partial rollout for long-horizon tasks
  - **배울 점**: RL infrastructure는 supervised training과 완전히 다른 설계 필요

**17. 훈련 비용 투명성**

- **MiniMax-M1**:
  - 3주, 512 H800 GPUs, ~$534K
  - **배울 점**: SOTA 근접 모델을 $534K로 훈련 가능 (접근성 증가)

### 성능 최적화

**18. Hybrid Reasoning Modes**

- **GLM-4.5**: Thinking mode vs Direct response
- **MiniMax-M1**: Test-time compute scaling
- **배울 점**: 작업에 따라 추론 깊이를 조정하는 것이 효율적

**19. Benchmark Selection의 중요성**

- **세 모델 모두**:
  - AIME, SWE-bench, TAU-bench, MMLU 공통 평가
  - 각자 custom benchmark도 개발 (GLM: CC-Bench, Kimi: ACEBench)
  - **배울 점**: 표준 벤치마크 + domain-specific 벤치마크 조합 필요

**20. Trade-off 인식**

- **MiniMax-M1**: Pure math/coding에서는 약하나 long-context에서 최강
- **GLM-4.5**: 파라미터 효율성 최고, 하지만 Kimi K2보다 agentic에서 약함
- **Kimi K2**: Agentic 최강, 하지만 가장 큰 모델
- **배울 점**: 모든 영역에서 최고는 불가능 → 목표에 맞는 trade-off 선택

### Safety & Alignment

**21. Safety Evaluation**

- **Kimi K2**:
  - Human-curated risk categories
  - Adversarial prompt evolution (attack-target-judge)
  - **배울 점**: Safety는 post-training의 핵심 구성 요소

### Open-Source 철학

**22. 오픈소스의 가치**

- **세 모델 모두 오픈소스/오픈 웨이트**:
  - GLM-4.5: Hugging Face + API
  - MiniMax-M1: Open-weight
  - Kimi K2: Base + Instruct 공개
  - **배울 점**: Proprietary 모델과 경쟁하려면 오픈소스 커뮤니티 협력 필수

---

## 5️⃣ 종합 인사이트

### 트렌드 분석

1. **MoE는 새로운 표준**
   - 세 모델 모두 MoE로 파라미터 효율성 극대화
   - Ultra-sparse MoE (Kimi K2)가 차세대 방향

2. **RL is King in Post-training**
   - SFT만으로는 부족, RL이 reasoning/agentic capabilities의 핵심
   - 각 모델이 독자적 RL 알고리즘 개발 (CISPO, Self-Critique, Curriculum)

3. **Attention 혁신 지속**
   - QK-Norm, QK-Clip, Hybrid Linear+Softmax
   - Long-context와 computational efficiency의 균형

4. **Agentic은 차세대 frontier**
   - Tool-use, function calling, multi-turn interaction이 핵심
   - Real-world integration (MCP tools, sandboxed execution)

5. **데이터 품질 > 데이터 양**
   - Synthetic data, rephrasing, transformation이 중요
   - Domain-specific curation (repo-level code, learning-note math)

### 향후 연구 방향

1. **더 효율적인 Attention 메커니즘**
   - MiniMax-M1의 linear attention이 promising
   - Multi-million token context가 차세대 목표?

2. **Novel RL 알고리즘**
   - CISPO, Self-Critique처럼 reasoning-specific RL 알고리즘
   - Verifiable rewards와 self-evaluation의 결합

3. **Token Efficiency 극대화**
   - Kimi K2의 rephrasing/transformation 접근
   - Less tokens, better quality

4. **Agentic Ecosystem**
   - 대규모 tool integration
   - Real-world execution environments
   - Multi-agent collaboration?

5. **Infrastructure 최적화**
   - Trillion-scale 모델의 효율적 훈련
   - RL infrastructure 전문화

---

## 6️⃣ 결론

이 세 논문은 현재 LLM 연구의 최전선을 보여줍니다:

- **GLM-4.5**: 파라미터 효율성과 균형잡힌 ARC 능력
- **MiniMax-M1**: Long-context efficiency와 novel RL (CISPO)
- **Kimi K2**: Ultra-sparse MoE와 agentic specialization

각 모델은 서로 다른 trade-off를 선택했지만, 공통적으로:
- MoE 아키텍처
- 대규모 RL post-training
- Synthetic data 전략
- Agentic capabilities 중시
- 오픈소스 철학

을 공유합니다. 이는 차세대 foundation model의 청사진이라 할 수 있습니다.

**핵심 메시지**:
- Bigger is not always better → Smarter architecture (MoE, sparse, hybrid attention)
- More data is not enough → Better data (synthetic, rephrased, domain-specific)
- SFT is not sufficient → Advanced RL (CISPO, self-critique, curriculum)
- General capability is not the goal → Agentic specialization

---

**분석 완료일**: 2025-11-10
**논문 출처**:
1. GLM-4.5: https://ar5iv.labs.arxiv.org/html/2508.06471
2. MiniMax-M1: https://arxiv.org/html/2506.13585
3. Kimi K2: https://arxiv.org/html/2507.20534
