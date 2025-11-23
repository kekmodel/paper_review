# PTX Loss 완전 설명

## 📋 PTX의 정의

**PTX = Pretraining Mix**

OpenAI의 InstructGPT 논문 (Ouyang et al., 2022)에서 처음 제안된 기법입니다.

---

## 🎯 핵심 개념

### 문제: "Alignment Tax"

RL 훈련 (특히 RLHF) 중에 발생하는 문제:
```
Pre-training → Fine-tuning → RL
                            ↓
                    General capability 손실!
```

**예시:**
- GPT-3: Excellent general knowledge, QA, reasoning
- After RLHF: Better at following instructions
- But: Worse at general NLP benchmarks (summarization, QA, etc.)

**이것이 "Alignment Tax"** - 정렬을 위해 일반 능력을 희생

---

## 💡 PTX Loss의 솔루션

### 아이디어

RL 훈련 중에:
```
RL updates (instruction following)
+
Pretraining updates (general capability)
```

**동시에 두 objective를 최적화!**

---

## 📐 수학적 정의

### InstructGPT의 Combined Objective

```
objective(ϕ) = E_(x,y)~D_RL [r_θ(x,y) - β·log(π_ϕ^RL(y|x) / π^SFT(y|x))]
               ↑ PPO reward term

             + γ·E_x~D_pretrain [log(π_ϕ^RL(x))]
               ↑ PTX term (pretraining mix)
```

**구성 요소:**

1. **First term (PPO):**
   - `r_θ(x,y)`: Reward model score
   - `β·log(π/π^SFT)`: KL penalty (SFT model과의 거리)
   - Purpose: Follow instructions well

2. **Second term (PTX):**
   - `log(π_ϕ^RL(x))`: Log-likelihood on pretraining data
   - `γ`: Mixing coefficient (hyperparameter)
   - Purpose: Maintain general capabilities

---

## 🔧 구현 세부사항

### 데이터

**RL Data (D_RL):**
- User prompts + model responses + rewards
- 예: Instruction following tasks
- Purpose: Alignment

**Pretraining Data (D_pretrain):**
- Original pretraining corpus
- 예: Web text, books, Wikipedia, etc.
- Purpose: General capability

### Training Loop

```python
for batch in training:
    # 1. RL samples
    rl_prompts, rl_responses = policy.generate(rl_tasks)
    rl_rewards = reward_model(rl_prompts, rl_responses)

    # PPO objective
    ppo_loss = compute_ppo_loss(
        policy,
        rl_prompts,
        rl_responses,
        rl_rewards,
        sft_model  # KL reference
    )

    # 2. Pretraining samples
    pretrain_text = sample_from_pretrain_data()

    # PTX objective (language modeling)
    ptx_loss = -log_likelihood(policy, pretrain_text)

    # 3. Combined
    total_loss = ppo_loss + γ * ptx_loss

    optimizer.step(total_loss)
```

### Hyperparameter: γ (Gamma)

**InstructGPT 논문:**
- Standard PPO: γ = 0 (PTX 없음)
- PPO-ptx: γ > 0 (PTX 포함)
- Specific value not disclosed (likely 0.1 - 0.5 based on common practice)

**Trade-off:**
- γ too small: Alignment tax 여전히 발생
- γ too large: RL 효과 감소, instruction following 저하
- Sweet spot: Balance between alignment & capability

---

## 📊 효과 (InstructGPT 결과)

### Standard PPO vs PPO-ptx

**Public NLP Benchmarks (without PTX):**
```
After RL: Performance drop 5-10%
→ Alignment tax
```

**With PTX (γ > 0):**
```
After RL: Performance maintained or slight drop only
→ Alignment tax greatly reduced!
```

**Instruction Following:**
```
PPO: 95% preference
PPO-ptx: 95% preference (동일!)
```

**결론:** PTX로 general capability 보존하면서도 instruction following은 유지!

---

## 🔄 Kimi K2에서의 사용

### Kimi K2 논문 언급

> "To prevent the potential forgetting of valuable, high-quality data during joint RL training, we curate a dataset comprising hand-selected, **high-quality samples** and integrate it into the RL objective through an auxiliary PTX loss."

### Kimi K2의 변형

**InstructGPT PTX:**
```
Pretraining data (original corpus)
→ Very general, broad coverage
```

**Kimi K2 PTX:**
```
Curated high-quality samples
→ Hand-selected specific capabilities
→ More targeted than original PTX
```

**차이점:**

| Aspect | InstructGPT PTX | Kimi K2 PTX |
|--------|----------------|-------------|
| **Data** | Original pretraining corpus | Curated high-quality samples |
| **Selection** | Random sampling | Hand-selected |
| **Coverage** | Very broad | Targeted capabilities |
| **Size** | Very large (100B+ tokens) | Smaller (likely 10M-100M tokens) |
| **Purpose** | General capability | Specific valuable skills |

### 왜 Curated Data?

**Kimi K2의 추론 (논문에 명시 안됨, 추측):**

1. **Efficiency:**
   - Full pretraining corpus는 너무 큼
   - Curated data가 더 효율적

2. **Quality:**
   - Pretraining data에는 low-quality content 포함
   - High-quality만 선택하면 더 효과적

3. **Targeted Preservation:**
   - 특정 중요한 능력 (reasoning, knowledge, etc.)에 집중
   - Broad but shallow보다 narrow but deep

---

## 🆚 PTX vs 다른 Forgetting Prevention 방법

### 1. PTX Loss vs Self-Distillation

| Aspect | PTX Loss | Self-Distillation |
|--------|----------|-------------------|
| **Teacher** | Pretraining data (static) | Previous model (dynamic) |
| **Loss** | Cross-entropy (hard targets) | KL divergence (soft targets) |
| **Storage** | Data storage (moderate) | Model weights (large) |
| **Compute** | 1x inference | 2x inference (teacher + student) |
| **Simplicity** | Simple | Complex |

**PTX가 더 간단하고 효율적!**

### 2. PTX Loss vs Experience Replay

| Aspect | PTX Loss | Experience Replay |
|--------|----------|-------------------|
| **Data** | Pretraining data | Previous RL trajectories |
| **Coverage** | Broad general capability | Specific RL tasks |
| **Storage** | Large (static corpus) | Growing buffer |
| **Purpose** | Maintain pre-RL capability | Maintain previous RL learning |

**PTX는 RL 이전 capability 보존, ER은 RL 중 learning 보존**

### 3. PTX Loss vs Multi-task Learning

| Aspect | PTX Loss | Multi-task Learning |
|--------|----------|---------------------|
| **Tasks** | Language modeling (single) | Multiple explicit tasks |
| **Data** | Unlabeled text | Task-specific labeled |
| **Objective** | Log-likelihood | Task-specific losses |
| **Flexibility** | Fixed (LM) | Flexible (various tasks) |

**PTX는 multi-task의 special case (LM as auxiliary task)**

---

## 🎓 실용적 가이드

### 언제 PTX Loss를 사용하는가?

#### ✅ 추천 상황:

1. **RLHF 또는 RL fine-tuning 시**
   - Alignment tax 방지
   - General capability 유지

2. **Domain-specific fine-tuning 시**
   - 특정 domain에 특화하면서
   - General knowledge 보존

3. **리소스가 충분할 때**
   - Pretraining data 저장 공간
   - 추가 compute (γ > 0 시 비용 증가)

#### ❌ 불필요한 상황:

1. **Small-scale fine-tuning**
   - Forgetting 위험이 낮음
   - PTX 오버헤드가 아까움

2. **Complete domain shift**
   - 예: English → Japanese (multilingual training 없이)
   - Pretraining data가 relevant하지 않음

3. **리소스 제약**
   - Pretraining corpus 접근 불가
   - Compute budget 부족

---

### 구현 체크리스트

**1. 데이터 준비:**
```python
# Option A: Full pretraining corpus (InstructGPT style)
pretrain_data = load_pretraining_corpus()

# Option B: Curated samples (Kimi K2 style)
pretrain_data = curate_high_quality_samples([
    "reasoning_examples",
    "general_qa",
    "common_knowledge",
    "basic_math",
    # ... targeted capabilities
])
```

**2. Loss 구현:**
```python
def compute_ptx_loss(model, pretrain_batch):
    """
    Simple language modeling loss
    """
    input_ids = pretrain_batch['input_ids']
    labels = pretrain_batch['labels']  # Next token prediction

    logits = model(input_ids)
    loss = cross_entropy(logits, labels)

    return loss

def training_step(model, rl_batch, pretrain_batch, gamma=0.1):
    # RL loss (PPO, GRPO, etc.)
    rl_loss = compute_rl_loss(model, rl_batch)

    # PTX loss
    ptx_loss = compute_ptx_loss(model, pretrain_batch)

    # Combined
    total_loss = rl_loss + gamma * ptx_loss

    return total_loss
```

**3. Hyperparameter 튜닝:**
```python
# Start with small gamma
gamma_candidates = [0.0, 0.05, 0.1, 0.2, 0.5]

for gamma in gamma_candidates:
    model = train_with_ptx(gamma=gamma)

    # Evaluate
    rl_performance = eval_rl_tasks(model)
    general_performance = eval_general_benchmarks(model)

    # Find balance
    if rl_performance > threshold and general_performance > threshold:
        optimal_gamma = gamma
        break
```

**4. 데이터 배칭:**
```python
# Mixed batching strategy
for step in training_steps:
    if step % ptx_frequency == 0:
        # Include PTX batch
        rl_batch = sample_rl_data(batch_size=32)
        ptx_batch = sample_pretrain_data(batch_size=32)
        loss = rl_loss(rl_batch) + gamma * ptx_loss(ptx_batch)
    else:
        # RL only
        rl_batch = sample_rl_data(batch_size=32)
        loss = rl_loss(rl_batch)
```

---

## 📚 관련 개념

### 1. Elastic Weight Consolidation (EWC)

**차이:**
- EWC: Important weights를 보호 (weight-level)
- PTX: Data를 계속 훈련 (data-level)

**PTX가 더 간단하고 효과적 (LLM에서)**

### 2. Knowledge Distillation

**차이:**
- KD: Teacher model의 knowledge transfer
- PTX: Data의 knowledge maintain

**PTX가 더 storage-efficient (no teacher model)**

### 3. Continual Learning

**관계:**
- PTX는 continual learning의 한 기법
- Specifically for maintaining pretraining capability during fine-tuning

---

## 🎯 핵심 요약

### PTX Loss란?

**정의:**
> RL 훈련 중에 pretraining data의 log-likelihood를 auxiliary objective로 추가하여, general capability를 보존하는 기법

**수식:**
```
Total Loss = RL Loss + γ × PTX Loss
           = RL Loss + γ × (-log P(pretrain_data))
```

**목적:**
- Alignment tax 방지
- General capability 유지
- Catastrophic forgetting 완화

**장점:**
- ✅ 단순한 구현
- ✅ Single model (no teacher)
- ✅ 효과적 (InstructGPT 검증)
- ✅ Flexible (γ 조정 가능)

**단점:**
- ❌ Pretraining data 필요 (storage)
- ❌ 추가 compute (γ > 0 시)
- ❌ Hyperparameter tuning (γ)

---

## 🔗 출처

**Original Paper:**
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). "Training language models to follow instructions with human feedback." *arXiv preprint arXiv:2203.02155*.
- InstructGPT / ChatGPT의 기반 논문

**사용 사례:**
- OpenAI InstructGPT (2022)
- Kimi K2 (2024) - Curated variant
- 다수의 RLHF 논문들

---

**작성일**: 2025-11-10
**근거**: InstructGPT 논문 (2022), Kimi K2 논문 (2024)
