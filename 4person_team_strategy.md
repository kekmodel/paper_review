# 4명 팀을 위한 월드클래스 모델 구축 전략

## 🎯 Executive Summary

**우리의 현실:**
- 팀원: 4명
- 리소스: 제한적
- 경쟁자: OpenAI, Anthropic (수백억 달러)
- 목표: 월드클래스 수준의 특화 모델

**우리의 전략:**
```
❌ 하지 않을 것: Pre-training from scratch (불가능)
✅ 할 것: Post-training specialization (가능하고 효과적)

핵심: 범용성이 아닌 특화된 영역에서 SOTA
```

**이 문서의 목적:**
지난 수개월간 분석한 GLM-4.5, MiniMax-M1, Kimi K2의 성공 요인을 4명 팀의 현실에 맞춰 **실행 가능한 전략**으로 변환

---

## 1️⃣ 4명 팀의 현실과 강점

### 1.1 우리가 가진 것

**✅ 강점:**
```
1. 빠른 의사결정
   - 회의: 30분
   - 방향 전환: 1일
   - 대기업: 주-개월 소요

2. 높은 집중도
   - 한 프로젝트에 100% 집중
   - 대기업: 여러 프로젝트 분산

3. 실험 속도
   - 아이디어 → 검증: 1-2일
   - 대기업: 승인 절차 복잡

4. 전문성 깊이
   - 4명 각자 전문 영역
   - T-shaped skills 활용
```

**❌ 없는 것:**
```
1. 대규모 GPU 클러스터
   - 우리: 8-32 GPUs 현실적
   - OpenAI: 10,000+ GPUs

2. 무한 예산
   - 우리: $10K-$100K
   - 대기업: $10M-$100M

3. 대규모 데이터팀
   - 우리: 4명 all-in
   - 대기업: 50-200명

4. 긴 runway
   - 우리: 3-6개월 내 성과 필요
   - 대기업: 수년 투자 가능
```

---

### 1.2 우리의 승리 조건

**잘못된 목표 (절대 불가능):**
```
❌ GPT-4 수준의 범용 모델
❌ 모든 벤치마크에서 SOTA
❌ 다국어 완벽 지원
❌ 1M context length
```

**올바른 목표 (달성 가능):**
```
✅ 특정 도메인에서 GPT-4 능가
   예: 한국 의료, 한국 법률, 특정 산업

✅ 핵심 벤치마크 2-3개에서 SOTA
   예: 한국어 reasoning, 특정 코딩 task

✅ 명확한 경제적 가치
   예: 월 $10K+ 매출 가능한 niche

✅ 6개월 내 POC, 12개월 내 production
```

**성공의 정의:**
```
ROI >= 5x
사용자가 돈 내고 쓰는 모델
특정 영역에서 확실한 차별화
```

---

## 2️⃣ 전략적 방향: Post-Training 중심

### 2.1 왜 Post-Training인가?

**숫자로 보는 현실:**

| 접근법 | 비용 | 시간 | 성공 확률 | 4명 팀 적합성 |
|--------|------|------|-----------|---------------|
| **Pre-training** | $100K-$10M | 6-18개월 | 20% | ❌ 불가능 |
| **Post-training** | $10K-$100K | 1-3개월 | 70% | ✅ 최적 |
| **RAG** | $1K-$10K | 2-4주 | 50% | ⚠️ 한계 명확 |
| **Prompt Eng** | $0-$1K | 1주 | 30% | ⚠️ 차별화 어려움 |

**논문들이 증명한 것:**

**GLM-4.5의 현실:**
```
Pre-training: 23T tokens, 수개월, $10M+
Post-training: Multi-stage RL
→ AIME 91% 달성의 핵심은 post-training

우리의 교훈:
Base model (Llama-2-7B 등) + 우리의 post-training
= 특정 영역에서 경쟁력 확보 가능
```

**Kimi K2의 증거:**
```
6+ RL stages로 multiple SOTA
각 stage: 2-3주, $10K-$50K

우리의 적용:
3-4 stages로 축소
각 stage에서 명확한 목표
= 총 $30K-$100K, 2-3개월
```

---

### 2.2 Base Model 선택 전략

**4명 팀을 위한 decision tree:**

```
Q1: 한국어가 핵심인가?
├─ Yes:
│  ├─ 7B: Polyglot-Ko-7B (최적)
│  ├─ 13B: Polyglot-Ko-12.8B
│  └─ 더 크게: Llama-3-70B + 한국어 post-training
│
└─ No (영어/코드 중심):
   ├─ 7B: Llama-3-8B (최신, 강력)
   ├─ 13B: Llama-3-13B
   └─ 70B: Llama-3-70B (budget 충분 시)

Q2: 특화 영역이 있는가?
├─ 의료: BioMistral-7B → Fine-tune
├─ 코드: CodeLlama-7B → Fine-tune
├─ 수학: Llemma-7B → Fine-tune
└─ 범용: Llama-3-8B → Custom post-training

추천: Llama-3-8B (또는 Polyglot-Ko-7B)
이유:
- 최신 (2024)
- Strong baseline
- 커뮤니티 지원 우수
- 8 GPUs로 fine-tuning 가능
```

**핵심 원칙:**
> 우리는 base model을 만들지 않는다. 최고의 base model을 선택하고, post-training으로 차별화한다.

---

## 3️⃣ Catastrophic Forgetting 방지 (핵심!)

### 3.1 논문들의 실제 방법 (self-distillation 아님!)

**중요한 발견:**
지난 분석에서 밝혀낸 것 - 논문들은 "self-distillation"이라고 vague하게 언급했지만, 실제로는:

**GLM-4.5의 실제 방법:**
```python
# Iterative Data Replacement (Bootstrapping)

# Stage 1: RL training
model_v1 = rl_train(base_model, math_tasks, epochs=3)

# Stage 2: Generate better data
improved_data = model_v1.generate(original_prompts)
# RL로 학습된 모델이 더 나은 reasoning chain 생성

# Stage 3: SFT on improved data
model_v2 = sft(model_v1, improved_data, epochs=2)

# Stage 4: RL again on new baseline
model_v3 = rl_train(model_v2, math_tasks, epochs=3)

# Repeat cycle: RL → Data Gen → SFT → RL
```

**핵심:** Self-distillation이 아니라 **data quality improvement cycle**

---

**Kimi K2의 실제 방법:**
```python
# PTX Loss (Pretraining Mix)

def train_rl_with_ptx(model, rl_data, ptx_data, gamma=0.1):
    """
    PTX = auxiliary task learning
    """
    for step in range(steps):
        # RL task (새로운 능력)
        rl_loss = compute_rl_loss(model, rl_data)

        # PTX task (기존 능력 유지)
        # Curated high-quality samples
        ptx_loss = compute_lm_loss(model, ptx_data)

        # Combined
        total_loss = rl_loss + gamma * ptx_loss

        optimizer.step(total_loss)

# PTX data: 10K-50K hand-selected samples
# - General QA
# - Basic reasoning
# - Common knowledge
# - Facts
```

**핵심:** PTX Loss는 **multi-task learning**의 일종

---

**MiniMax-M1의 실제 방법:**
```python
# Multi-task Curriculum Learning

def curriculum_training(model, tasks):
    # Stage 1: Math + Logic only
    stage1_data = mix_tasks({
        "math": 0.6,
        "logic": 0.4
    })
    model = train(model, stage1_data, epochs=3)

    # Stage 2: Add coding, KEEP math+logic
    stage2_data = mix_tasks({
        "math": 0.4,      # Still training!
        "logic": 0.3,     # Still training!
        "coding": 0.3     # NEW
    })
    model = train(model, stage2_data, epochs=3)

    # Stage 3: Add general, KEEP all previous
    stage3_data = mix_tasks({
        "math": 0.25,     # Still training!
        "logic": 0.20,    # Still training!
        "coding": 0.25,   # Still training!
        "general": 0.30   # NEW
    })
    model = train(model, stage3_data, epochs=3)

    return model
```

**핵심:** **Never stop practicing old tasks** = Continued practice

---

### 3.2 4명 팀을 위한 실전 구현

**Method 1: Multi-task Training (가장 중요!)**

```python
"""
4명 팀 구현: Multi-task RL
"""

# Step 1: Task definition
tasks = {
    "target_skill": {
        "weight": 0.60,  # 60% 새로운 능력
        "data": load_target_data(),
        "description": "우리가 강화하고 싶은 능력"
    },
    "general_qa": {
        "weight": 0.20,  # 20% 일반 능력 유지
        "data": load_general_qa(),
        "description": "기본 QA 능력"
    },
    "reasoning": {
        "weight": 0.10,
        "data": load_reasoning_data(),
        "description": "기본 reasoning"
    },
    "safety": {
        "weight": 0.10,
        "data": load_safety_data(),
        "description": "Safety alignment"
    }
}

# Step 2: Training loop
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # Sample task proportionally
        task_name = random.choices(
            list(tasks.keys()),
            weights=[t["weight"] for t in tasks.values()]
        )[0]

        batch = sample(tasks[task_name]["data"], batch_size=16)

        # Train on sampled task
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

        # Log per-task metrics
        if step % 100 == 0:
            evaluate_each_task(model, tasks)
            # Check for regression!

# Cost: $10K-$30K (2-3주)
# 효과: Catastrophic forgetting 80-90% 감소
```

**왜 이것이 작동하는가?**
```
뇌의 학습과 동일:
- 피아노 배우면서도 걷기는 계속 연습
- 새 언어 배우면서도 모국어는 계속 사용

모델도 동일:
- 새 task 학습 중에도 old tasks 계속 연습
- "Use it or lose it" → Keep using it!
```

---

**Method 2: PTX Loss (보조 방법)**

```python
"""
PTX Loss 간단 구현 (Kimi K2 style)
"""

# Step 1: PTX data curation (중요!)
ptx_dataset = []

# A. From base model의 strong outputs
base_outputs = base_model.generate(diverse_prompts, temperature=0.7)
high_quality = filter_by_quality(base_outputs, threshold=0.8)
ptx_dataset.extend(high_quality[:5000])

# B. Public high-quality datasets
ptx_dataset.extend(load_dataset("HuggingFaceH4/no_robots", split="train[:5000]"))
ptx_dataset.extend(load_dataset("OpenAssistant/oasst1", split="train[:5000]"))

# C. Hand-curated (가장 중요!)
# 팀원들이 직접 선별한 high-quality examples
ptx_dataset.extend(load_curated_examples("curated/high_quality.jsonl"))

# Total: 10K-20K samples

# Step 2: Training with PTX
def train_with_ptx(model, rl_data, ptx_data, gamma=0.1):
    for step in range(steps):
        # Main task (RL)
        rl_batch = sample(rl_data, 16)
        rl_loss = compute_grpo_loss(model, rl_batch)

        # PTX task (preservation)
        ptx_batch = sample(ptx_data, 16)
        ptx_loss = compute_sft_loss(model, ptx_batch)

        # Combine
        total_loss = rl_loss + gamma * ptx_loss

        total_loss.backward()
        optimizer.step()

# Hyperparameter: gamma
# Start: 0.1
# If forgetting detected: increase to 0.2
# If target task not improving: decrease to 0.05

# Cost: +10% training time
# 효과: Additional 10-15% forgetting reduction
```

---

**Method 3: Curriculum Learning**

```python
"""
Difficulty-based Curriculum (GLM-4.5 style)
"""

def curriculum_rl(model, dataset):
    # Step 1: Score difficulty
    # (base model의 pass rate로 측정)
    difficulties = {}
    for sample in dataset:
        pass_rate = base_model.eval(sample)
        if pass_rate > 0.8:
            difficulties[sample.id] = "easy"
        elif pass_rate > 0.3:
            difficulties[sample.id] = "medium"
        else:
            difficulties[sample.id] = "hard"

    # Step 2: Stage 1 (Easy warmup)
    easy_data = [s for s in dataset if difficulties[s.id] == "easy"]
    model = rl_train(model, easy_data, epochs=2)
    print("Stage 1 complete: Easy problems mastered")

    # Step 3: Stage 2 (Add medium, keep easy)
    medium_data = [s for s in dataset if difficulties[s.id] == "medium"]
    mixed_data = easy_data + medium_data  # BOTH!
    model = rl_train(model, mixed_data, epochs=3)
    print("Stage 2 complete: Medium problems learned, easy maintained")

    # Step 4: Stage 3 (Add hard, keep all)
    hard_data = [s for s in dataset if difficulties[s.id] == "hard"]
    final_data = easy_data + medium_data + hard_data  # ALL!
    model = rl_train(model, final_data, epochs=5)
    print("Stage 3 complete: All difficulties mastered")

    return model

# Cost: Same as regular training (just reordered)
# 효과: 5-10% better final performance
#       More stable training (fewer spikes)
```

---

### 3.3 Forgetting Detection & Mitigation

**실시간 모니터링:**

```python
"""
Forgetting 조기 감지 시스템
"""

class ForgettingMonitor:
    def __init__(self, baseline_model):
        # Baseline performance 측정
        self.baseline_scores = {
            "mmlu": evaluate(baseline_model, "mmlu"),
            "humaneval": evaluate(baseline_model, "humaneval"),
            "gsm8k": evaluate(baseline_model, "gsm8k"),
            "safety": evaluate_safety(baseline_model),
        }

        self.thresholds = {
            "warning": -0.03,   # 3% drop
            "critical": -0.05,  # 5% drop
        }

    def check(self, current_model, step):
        """매 100 steps마다 체크"""
        if step % 100 != 0:
            return

        current_scores = {
            "mmlu": evaluate(current_model, "mmlu"),
            "humaneval": evaluate(current_model, "humaneval"),
            "gsm8k": evaluate(current_model, "gsm8k"),
            "safety": evaluate_safety(current_model),
        }

        alerts = []
        for metric, baseline in self.baseline_scores.items():
            current = current_scores[metric]
            delta = current - baseline

            if delta < self.thresholds["critical"]:
                alerts.append({
                    "severity": "CRITICAL",
                    "metric": metric,
                    "baseline": baseline,
                    "current": current,
                    "delta": delta,
                    "action": "STOP TRAINING & INVESTIGATE"
                })
            elif delta < self.thresholds["warning"]:
                alerts.append({
                    "severity": "WARNING",
                    "metric": metric,
                    "baseline": baseline,
                    "current": current,
                    "delta": delta,
                    "action": "Increase multi-task mixing or PTX weight"
                })

        if alerts:
            send_slack_alert(alerts)
            log_alerts(alerts)

            # Auto-mitigation
            if any(a["severity"] == "CRITICAL" for a in alerts):
                # Rollback to previous checkpoint
                current_model.load_state_dict(
                    torch.load("checkpoint_step_{}.pt".format(step - 100))
                )

                # Adjust hyperparameters
                increase_old_task_weight()
                increase_ptx_gamma()

        return alerts

# Usage in training loop
monitor = ForgettingMonitor(baseline_model)

for step in training_steps:
    # Train
    loss.backward()
    optimizer.step()

    # Check forgetting
    alerts = monitor.check(model, step)

    if alerts:
        handle_alerts(alerts)
```

**Cost:** 평가 시간 +5% (but crucial!)
**Value:** Forgetting 조기 발견 → 수백만원 절약

---

## 4️⃣ 데이터 전략: Quality > Quantity

### 4.1 데이터 철학

**논문들의 교훈:**
```
GLM-4.5: 23T tokens (많음)
Kimi K2: 15.5T tokens + knowledge rephrasing (적지만 고품질)
MiniMax-M1: 7.5T tokens + careful curation (가장 적지만 효율적)

→ 더 많은 tokens < 더 나은 tokens
```

**SmolLM3 Playbook의 핵심:**
> "Using what seems like highest quality data doesn't always yield stronger models. Training on pure arXiv underperforms diverse general text."

**우리의 전략:**
```
❌ 1억 개의 noisy samples
✅ 10만 개의 carefully curated samples

❌ 모든 웹 크롤링
✅ Strategic domain mixture

❌ Synthetic data로만
✅ Real + Synthetic balanced
```

---

### 4.2 4명 팀의 데이터 소싱

**Phase 1: Public Datasets (Week 1)**

```python
"""
High-quality Public Sources
"""

# General Instruction Following
datasets = [
    "Open-Orca/OpenOrca",  # 1M samples, GPT-4 generated
    "HuggingFaceH4/no_robots",  # 10K, human-written
    "timdettmers/openassistant-guanaco",  # 10K conversations
]

# Domain-Specific
domain_datasets = {
    "coding": [
        "bigcode/starcoderdata",
        "m-a-p/CodeFeedback-Filtered-Instruction",
    ],
    "math": [
        "lighteval/MATH",
        "openai/gsm8k",
        "competition_math",
    ],
    "korean": [
        "nlpai-lab/kullm-v2",
        "beomi/KoAlpaca-v1.1a",
    ],
}

# Download & Filter
def collect_base_data():
    data = []

    # Load public datasets
    for ds_name in datasets:
        ds = load_dataset(ds_name)
        # Filter by quality score
        filtered = [sample for sample in ds if sample.get("quality_score", 0) > 3.5]
        data.extend(filtered)

    # Deduplicate (중요!)
    data = deduplicate(data, threshold=0.85)

    # Sample diverse subset
    final_data = stratified_sample(data, n=50000)

    return final_data

# Time: 1-2 days
# Cost: $0 (public data)
# Output: 50K high-quality samples
```

---

**Phase 2: Synthetic Data Generation (Week 2-3)**

**Method A: GPT-4 Generation (가장 효과적)**

```python
"""
Domain-specific Synthetic Data with GPT-4
"""

def generate_synthetic_dataset(domain, target_count=10000):
    """
    Cost: 10K samples x $0.01 = $100
    Time: 2-3 days
    """

    # Templates for domain
    templates = load_templates(domain)

    synthetic_data = []

    for i in range(target_count):
        # Random template
        template = random.choice(templates)

        # Generate with GPT-4
        prompt = f"""
        Generate a high-quality {domain} example.

        Template: {template}

        Requirements:
        1. Realistic scenario
        2. Clear, correct answer
        3. Appropriate difficulty
        4. {domain}-specific knowledge

        Format:
        Question: [user query]
        Reasoning: [step-by-step thinking]
        Answer: [final answer]
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9  # Diversity
        )

        # Parse and validate
        sample = parse_response(response)

        # Quality check
        if validate_quality(sample):
            synthetic_data.append(sample)
        else:
            continue  # Regenerate

    return synthetic_data

# Example domains
domains = ["medical_qa", "legal_reasoning", "korean_culture", "financial_analysis"]

for domain in domains:
    synthetic = generate_synthetic_dataset(domain, target_count=10000)
    save(synthetic, f"synthetic_{domain}.jsonl")

# Total cost: 4 domains x 10K x $0.01 = $400
# Total time: 1주
# Output: 40K domain-specific samples
```

---

**Method B: Self-Instruct (Low-cost)**

```python
"""
Self-Instruct Style Generation
"""

def self_instruct_generation(seed_examples, target_model, n=10000):
    """
    Use existing model to generate variations
    Cost: $0 (use own GPU)
    """

    generated = []

    for i in range(n):
        # Random seed
        seed = random.choice(seed_examples)

        # Generate variation
        prompt = f"""
        Based on this example, create a similar but different problem:

        Example:
        {seed}

        Create a new problem with:
        - Different numbers/names/scenarios
        - Same difficulty level
        - Same format
        """

        output = target_model.generate(
            prompt,
            max_length=1024,
            temperature=0.8,
            top_p=0.9
        )

        # Validate
        if is_valid(output) and not too_similar(output, seed):
            generated.append(output)

    return generated

# Cost: GPU time only (~$50-$100)
# Quality: Lower than GPT-4, but useful for diversity
```

---

**Method C: Knowledge Rephrasing (Kimi K2 Style)**

```python
"""
Transform existing data to better format
"""

def rephrase_for_learning(math_problems):
    """
    Original: Dry theorem statement
    Transformed: Learning-note style explanation
    """

    rephrased = []

    for problem in math_problems:
        prompt = f"""
        Rephrase this math problem in a learning-note style:

        Original: {problem}

        Transform to:
        1. Intuitive explanation
        2. Step-by-step breakdown
        3. Visual description (if applicable)
        4. Common misconceptions
        5. Practice tips

        Keep factual accuracy (fidelity check required)
        """

        transformed = gpt4.generate(prompt)

        # Fidelity verification
        if verify_fidelity(problem, transformed) > 0.9:
            rephrased.append(transformed)

    return rephrased

# Use case: Transform existing datasets
math_dataset = load_dataset("lighteval/MATH")
learning_style = rephrase_for_learning(math_dataset)

# Cost: Dataset size x $0.01 = $100-$500
# Benefit: Same information, better learning signal
```

---

### 4.3 데이터 품질 관리

**Quality Scoring Pipeline:**

```python
"""
Automated Quality Assessment
"""

class QualityScorer:
    def __init__(self):
        self.criteria = {
            "coherence": self.check_coherence,
            "correctness": self.check_correctness,
            "difficulty": self.check_difficulty,
            "diversity": self.check_diversity,
            "safety": self.check_safety,
        }

    def score(self, sample):
        scores = {}
        for criterion, check_fn in self.criteria.items():
            scores[criterion] = check_fn(sample)

        # Weighted average
        weights = {
            "coherence": 0.2,
            "correctness": 0.3,  # Most important
            "difficulty": 0.2,
            "diversity": 0.1,
            "safety": 0.2,
        }

        total_score = sum(scores[k] * weights[k] for k in scores)

        return total_score, scores

    def check_correctness(self, sample):
        """Most critical check"""
        # For math/code: Execute and verify
        if sample["domain"] in ["math", "code"]:
            try:
                result = execute_and_verify(sample)
                return 1.0 if result else 0.0
            except:
                return 0.0

        # For general: Use GPT-4 as judge
        prompt = f"Is this answer correct? {sample}"
        judgment = gpt4.judge(prompt)
        return judgment.score

    def check_difficulty(self, sample):
        """Use base model's performance"""
        success_rate = base_model.solve(sample, n_attempts=5)

        # Target: 0.3-0.7 (not too easy, not too hard)
        if 0.3 <= success_rate <= 0.7:
            return 1.0
        elif success_rate < 0.1:
            return 0.3  # Too hard (but keep some)
        elif success_rate > 0.9:
            return 0.5  # Too easy (but keep some)
        else:
            return 0.7

    def check_safety(self, sample):
        """Critical for production"""
        # Toxicity
        toxicity = toxicity_classifier(sample["text"])
        if toxicity > 0.5:
            return 0.0  # Reject

        # Bias
        bias_score = bias_detector(sample["text"])
        if bias_score > 0.7:
            return 0.3  # Flag for review

        # PII
        if contains_pii(sample["text"]):
            return 0.0  # Reject

        return 1.0

# Usage
scorer = QualityScorer()
filtered_data = []

for sample in raw_data:
    score, details = scorer.score(sample)

    if score > 0.7:  # High quality
        sample["quality_score"] = score
        filtered_data.append(sample)
    elif score > 0.5:  # Medium quality
        # Human review queue
        review_queue.append(sample)
    else:  # Low quality
        # Discard
        pass

# Statistics
print(f"Kept: {len(filtered_data)}")
print(f"Review: {len(review_queue)}")
print(f"Discarded: {len(raw_data) - len(filtered_data) - len(review_queue)}")
```

---

### 4.4 데이터 Mixture 최적화

**Critical Ablation Study:**

```python
"""
Data Mixture Ablation (가장 중요한 실험!)
"""

def ablation_study_mixture():
    """
    Cost: 5-10 ablations x $2K = $10K-$20K
    Time: 1-2주
    Value: Optimal mixture = +10-20% performance
    """

    # Candidate mixtures
    mixtures = [
        {"general": 70, "domain": 30},  # Baseline
        {"general": 60, "domain": 40},  # More domain
        {"general": 50, "domain": 50},  # Balanced
        {"general": 40, "domain": 60},  # Domain-heavy
        {"general": 30, "domain": 70},  # Extreme domain
    ]

    # For each mixture
    results = {}
    for mixture in mixtures:
        # Prepare data
        data = mix_datasets(mixture)

        # Train small model (1B, 10B tokens)
        model = train_small_pilot(
            model_size="1B",
            data=data,
            tokens=10_000_000_000,
            gpus=8,
            duration_hours=24
        )

        # Evaluate on target benchmarks
        scores = {
            "domain_task": evaluate_domain(model),
            "general_mmlu": evaluate(model, "mmlu"),
            "coding": evaluate(model, "humaneval"),
            "safety": evaluate_safety(model),
        }

        results[str(mixture)] = scores

    # Analyze results
    best_mixture = max(results, key=lambda m: results[m]["domain_task"])

    print(f"Best mixture: {best_mixture}")
    print(f"Domain task: {results[best_mixture]['domain_task']}")
    print(f"General MMLU: {results[best_mixture]['general_mmlu']}")

    # Verify at scale
    final_model = train_full_model(
        model_size="7B",
        data=mix_datasets(eval(best_mixture)),
        tokens=100_000_000_000,
        gpus=32,
        duration_weeks=2
    )

    return final_model, best_mixture

# This is THE most important experiment
# Do NOT skip this!
```

---

## 5️⃣ 실험 프로세스: Iterative Improvement

### 5.1 Rapid Iteration Framework

**3-Layer Experimentation:**

```
Layer 1: Quick Tests (1-2 days, $100-$500)
├─ Purpose: Idea validation
├─ Model: 1B or base model
├─ Data: 10K-100K samples
└─ Decision: Go/No-go

Layer 2: Pilot (1-2 weeks, $2K-$10K)
├─ Purpose: Approach validation
├─ Model: 1B-3B
├─ Data: 10M-100M tokens
└─ Decision: Scale up or pivot

Layer 3: Full Training (1-2 months, $20K-$100K)
├─ Purpose: Production model
├─ Model: 7B-13B
├─ Data: 1T tokens
└─ Decision: Deploy or iterate
```

**Concrete Example:**

```python
"""
Iteration Cycle for Medical QA Model
"""

# Iteration 1: Quick Test (Day 1-2)
def test_medical_qa_concept():
    # Load base model
    model = load("llama-3-8b")

    # Quick LoRA on 10K medical QA
    medical_data = load_dataset("medical_qa", split="train[:10000]")

    lora_model = lora_finetune(
        model,
        medical_data,
        r=8,
        epochs=3,
        gpus=1,
        time_hours=4
    )

    # Evaluate
    score = evaluate_medical(lora_model)
    baseline = evaluate_medical(model)

    improvement = score - baseline

    if improvement > 0.1:  # >10% improvement
        return "GO", score
    else:
        return "NO-GO", score

result, score = test_medical_qa_concept()
print(f"Quick test: {result}, Score: {score}")

if result == "NO-GO":
    print("Pivot: Try different data or approach")
    exit()

# Iteration 2: Pilot (Week 1-2)
def pilot_medical_model():
    # Larger model, more data
    model = load("llama-3-8b")

    # Curate 50K medical QA
    medical_data = curate_medical_data(n=50000)

    # LoRA fine-tuning
    pilot_model = lora_finetune(
        model,
        medical_data,
        r=16,
        epochs=5,
        gpus=8,
        time_days=7
    )

    # Comprehensive evaluation
    scores = {
        "medical_qa": evaluate_medical(pilot_model),
        "mmlu_medical": evaluate(pilot_model, "mmlu_medical"),
        "safety": evaluate_medical_safety(pilot_model),
        "general_mmlu": evaluate(pilot_model, "mmlu"),  # Forgetting check
    }

    # Decision criteria
    if (scores["medical_qa"] > 0.7 and  # Good performance
        scores["general_mmlu"] > 0.60 and  # No catastrophic forgetting
        scores["safety"] > 0.9):  # Safe
        return "SCALE_UP", scores
    else:
        return "ITERATE", scores

result, scores = pilot_medical_model()
print(f"Pilot: {result}")
print(f"Scores: {scores}")

if result == "ITERATE":
    # Analyze what went wrong
    if scores["medical_qa"] < 0.7:
        print("Need better data or more training")
    if scores["general_mmlu"] < 0.60:
        print("Catastrophic forgetting! Add multi-task")
    if scores["safety"] < 0.9:
        print("Safety issues! Add safety data")
    # Adjust and repeat

# Iteration 3: Full Training (Month 1-2)
def full_training():
    # Best configuration from pilot
    model = load("llama-3-8b")

    # Full dataset (100K-500K samples)
    medical_data = curate_medical_data(n=500000)

    # Multi-stage training
    # Stage 1: SFT
    sft_model = full_finetune(  # Not LoRA, full FT
        model,
        medical_data,
        epochs=3,
        gpus=32,
        time_weeks=2
    )

    # Stage 2: RL with medical rewards
    rl_model = rl_train(
        sft_model,
        medical_tasks,
        reward_fn=medical_reward,
        gpus=32,
        time_weeks=2
    )

    # Stage 3: Safety RL
    final_model = safety_rl(
        rl_model,
        safety_data,
        gpus=32,
        time_weeks=1
    )

    # Final evaluation
    final_scores = comprehensive_eval(final_model)

    if final_scores["medical_qa"] > 0.75:
        return "DEPLOY", final_model
    else:
        return "ONE_MORE_ITERATION", final_model

# Total time: 2-3 months
# Total cost: $30K-$100K
# Success rate: 70% (with this process)
```

---

### 5.2 Evaluation Framework

**Multi-Dimensional Assessment:**

```python
"""
Comprehensive Evaluation System
"""

class ComprehensiveEvaluator:
    def __init__(self, domain="medical"):
        self.domain = domain

        # Domain-specific benchmarks
        self.domain_benchmarks = self.load_domain_benchmarks(domain)

        # General benchmarks
        self.general_benchmarks = {
            "mmlu": MMLUBenchmark(),
            "truthfulqa": TruthfulQABenchmark(),
            "toxicity": ToxicityBenchmark(),
        }

        # Custom benchmarks
        self.custom_benchmarks = self.create_custom_benchmarks(domain)

    def eval_comprehensive(self, model):
        results = {}

        # 1. Domain Performance (가장 중요)
        print("Evaluating domain performance...")
        for name, benchmark in self.domain_benchmarks.items():
            score = benchmark.evaluate(model)
            results[f"domain_{name}"] = score

        # 2. General Capability (forgetting check)
        print("Checking general capability...")
        for name, benchmark in self.general_benchmarks.items():
            score = benchmark.evaluate(model)
            results[f"general_{name}"] = score

        # 3. Custom Tasks (real-world relevance)
        print("Evaluating custom tasks...")
        for name, benchmark in self.custom_benchmarks.items():
            score = benchmark.evaluate(model)
            results[f"custom_{name}"] = score

        # 4. Safety & Robustness
        print("Safety checks...")
        results["safety"] = self.eval_safety(model)
        results["robustness"] = self.eval_robustness(model)

        # 5. Efficiency Metrics
        results["latency"] = self.measure_latency(model)
        results["throughput"] = self.measure_throughput(model)

        # 6. Generate report
        report = self.generate_report(results)

        return results, report

    def create_custom_benchmarks(self, domain):
        """Domain-specific real-world tasks"""
        if domain == "medical":
            return {
                "diagnosis": DiagnosisBenchmark(),
                "treatment_plan": TreatmentPlanBenchmark(),
                "drug_interaction": DrugInteractionBenchmark(),
                "patient_communication": PatientCommunicationBenchmark(),
            }
        elif domain == "legal":
            return {
                "contract_analysis": ContractAnalysisBenchmark(),
                "case_law_search": CaseLawSearchBenchmark(),
                "legal_writing": LegalWritingBenchmark(),
            }
        # Add more domains...

    def eval_safety(self, model):
        """Critical for production"""
        safety_score = 0

        # 1. Jailbreak resistance
        jailbreak_attempts = load_jailbreak_prompts()
        jailbreak_success = sum(
            1 for attempt in jailbreak_attempts
            if is_jailbroken(model.generate(attempt))
        )
        jailbreak_resistance = 1 - (jailbreak_success / len(jailbreak_attempts))
        safety_score += jailbreak_resistance * 0.3

        # 2. Toxicity
        toxic_prompts = load_toxic_prompts()
        toxic_responses = sum(
            1 for prompt in toxic_prompts
            if is_toxic(model.generate(prompt))
        )
        toxicity_score = 1 - (toxic_responses / len(toxic_prompts))
        safety_score += toxicity_score * 0.3

        # 3. Refusal appropriateness
        # Should refuse harmful requests, but not over-refuse
        refusal_score = evaluate_refusal_calibration(model)
        safety_score += refusal_score * 0.2

        # 4. Bias
        bias_score = evaluate_bias(model)
        safety_score += (1 - bias_score) * 0.2

        return safety_score

    def generate_report(self, results):
        """Human-readable report"""
        report = f"""
        ===== Evaluation Report =====

        Domain Performance:
        """

        for key, value in results.items():
            if key.startswith("domain_"):
                metric = key.replace("domain_", "")
                status = "✅" if value > 0.7 else "⚠️" if value > 0.5 else "❌"
                report += f"  {status} {metric}: {value:.2%}\n"

        report += "\nGeneral Capability:\n"
        for key, value in results.items():
            if key.startswith("general_"):
                metric = key.replace("general_", "")
                status = "✅" if value > 0.6 else "⚠️" if value > 0.5 else "❌"
                report += f"  {status} {metric}: {value:.2%}\n"

        report += "\nSafety & Robustness:\n"
        report += f"  Safety: {results['safety']:.2%}\n"
        report += f"  Robustness: {results['robustness']:.2%}\n"

        report += "\nEfficiency:\n"
        report += f"  Latency: {results['latency']:.2f}ms\n"
        report += f"  Throughput: {results['throughput']:.0f} tokens/sec\n"

        # Overall assessment
        domain_avg = np.mean([v for k, v in results.items() if k.startswith("domain_")])
        general_avg = np.mean([v for k, v in results.items() if k.startswith("general_")])

        report += f"\n===== Overall =====\n"
        report += f"Domain Average: {domain_avg:.2%}\n"
        report += f"General Average: {general_avg:.2%}\n"

        if domain_avg > 0.75 and general_avg > 0.60 and results['safety'] > 0.9:
            report += "\n🎉 READY FOR PRODUCTION\n"
        elif domain_avg > 0.65:
            report += "\n⚠️ NEEDS IMPROVEMENT\n"
        else:
            report += "\n❌ MAJOR ISSUES - DO NOT DEPLOY\n"

        return report

# Usage
evaluator = ComprehensiveEvaluator(domain="medical")
results, report = evaluator.eval_comprehensive(model)
print(report)
save_results(results, "eval_results.json")
```

---

### 5.3 Iteration Decision Framework

**When to Iterate vs Deploy:**

```python
"""
Data-Driven Iteration Decisions
"""

class IterationDecider:
    def __init__(self, target_metrics, budget_remaining):
        self.target_metrics = target_metrics
        self.budget_remaining = budget_remaining
        self.iteration_history = []

    def should_iterate(self, current_results):
        """
        Returns: ("DEPLOY" | "ITERATE" | "PIVOT" | "STOP", reason)
        """

        # Check 1: Target metrics achieved?
        targets_met = all(
            current_results[metric] >= threshold
            for metric, threshold in self.target_metrics.items()
        )

        if targets_met:
            return "DEPLOY", "All target metrics achieved! 🎉"

        # Check 2: Budget remaining?
        if self.budget_remaining < 0.2 * self.initial_budget:
            return "STOP", "Budget exhausted. Current best is good enough."

        # Check 3: Progress in recent iterations?
        if len(self.iteration_history) >= 3:
            recent_progress = [
                self.iteration_history[-i]["primary_metric"]
                for i in range(1, 4)
            ]

            # No improvement in last 3 iterations?
            if all(recent_progress[i] <= recent_progress[i+1] for i in range(2)):
                return "PIVOT", "No progress in 3 iterations. Try different approach."

        # Check 4: ROI calculation
        estimated_cost_to_target = self.estimate_cost_to_target(current_results)

        if estimated_cost_to_target > self.budget_remaining:
            return "STOP", f"Est. cost ${estimated_cost_to_target} > budget ${self.budget_remaining}"

        # Check 5: Diminishing returns?
        if len(self.iteration_history) >= 2:
            prev_improvement = (
                self.iteration_history[-1]["primary_metric"] -
                self.iteration_history[-2]["primary_metric"]
            )

            if prev_improvement < 0.02:  # <2% improvement
                cost_per_point = (
                    self.iteration_history[-1]["cost"] / prev_improvement
                )

                if cost_per_point > 5000:  # $5K per 1% improvement
                    return "STOP", "Diminishing returns. Deploy current version."

        # Default: Iterate
        gap_to_target = {
            metric: threshold - current_results[metric]
            for metric, threshold in self.target_metrics.items()
        }

        biggest_gap = max(gap_to_target.items(), key=lambda x: x[1])

        return "ITERATE", f"Continue. Focus on {biggest_gap[0]} (gap: {biggest_gap[1]:.2%})"

    def recommend_next_step(self, current_results, decision):
        """Specific recommendations"""
        if decision[0] == "ITERATE":
            # Analyze what needs improvement
            recommendations = []

            # Domain performance low?
            domain_score = np.mean([
                v for k, v in current_results.items()
                if k.startswith("domain_")
            ])

            if domain_score < self.target_metrics.get("domain_avg", 0.75):
                recommendations.append({
                    "action": "Increase domain data",
                    "details": "Add 20K more domain-specific samples",
                    "est_cost": "$5K",
                    "est_improvement": "+5-10%"
                })

            # Forgetting detected?
            general_score = np.mean([
                v for k, v in current_results.items()
                if k.startswith("general_")
            ])

            if general_score < 0.60:
                recommendations.append({
                    "action": "Add multi-task training",
                    "details": "Increase general data from 20% to 40%",
                    "est_cost": "$2K",
                    "est_improvement": "Reduce forgetting by 50%"
                })

            # Safety issues?
            if current_results["safety"] < 0.9:
                recommendations.append({
                    "action": "Safety RL",
                    "details": "Additional safety training with adversarial prompts",
                    "est_cost": "$3K",
                    "est_improvement": "Safety: 0.9+"
                })

            return recommendations

        return []

# Usage in iteration loop
decider = IterationDecider(
    target_metrics={
        "domain_avg": 0.75,
        "general_mmlu": 0.65,
        "safety": 0.90,
    },
    budget_remaining=50000
)

# After each training iteration
decision, reason = decider.should_iterate(current_results)
print(f"Decision: {decision}")
print(f"Reason: {reason}")

if decision == "ITERATE":
    recommendations = decider.recommend_next_step(current_results, decision)
    print("\nRecommended actions:")
    for rec in recommendations:
        print(f"- {rec['action']}: {rec['details']}")
        print(f"  Cost: {rec['est_cost']}, Expected: {rec['est_improvement']}")
```

---

## 6️⃣ 데이터 생성/개선 파이프라인

### 6.1 Continuous Data Improvement

**Flywheel Strategy:**

```
Model v1 (weak) → Generate data → Model v2 (better)
       ↑                                    ↓
       └────────── Generate better data ────┘
```

**Implementation:**

```python
"""
Self-Improving Data Pipeline (GLM-4.5 style)
"""

def data_improvement_cycle(base_model, target_domain, iterations=3):
    """
    Each iteration:
    1. Current model generates solutions
    2. Filter high-quality ones
    3. Train next model on improved data
    4. Repeat
    """

    current_model = base_model

    for iteration in range(iterations):
        print(f"=== Iteration {iteration + 1} ===")

        # Step 1: Generate with current model
        print("Generating solutions...")
        prompts = load_prompts(target_domain, n=10000)

        generated_solutions = []
        for prompt in tqdm(prompts):
            # Generate multiple candidates
            candidates = current_model.generate(
                prompt,
                num_return_sequences=5,
                temperature=0.8
            )

            # Score each candidate
            scored = [
                (candidate, score_quality(prompt, candidate))
                for candidate in candidates
            ]

            # Keep best
            best = max(scored, key=lambda x: x[1])

            if best[1] > 0.7:  # Quality threshold
                generated_solutions.append({
                    "prompt": prompt,
                    "solution": best[0],
                    "quality": best[1]
                })

        print(f"Generated {len(generated_solutions)} high-quality solutions")

        # Step 2: Combine with original data
        original_data = load_original_data(target_domain)

        combined_data = original_data + generated_solutions

        # Deduplicate
        combined_data = deduplicate(combined_data, threshold=0.9)

        print(f"Combined dataset: {len(combined_data)} samples")

        # Step 3: Train next model
        print("Training next model...")
        next_model = finetune(
            base_model,  # Always start from base
            combined_data,
            epochs=3,
            learning_rate=2e-5
        )

        # Step 4: Evaluate improvement
        prev_score = evaluate_domain(current_model)
        new_score = evaluate_domain(next_model)
        improvement = new_score - prev_score

        print(f"Previous: {prev_score:.2%}")
        print(f"New: {new_score:.2%}")
        print(f"Improvement: {improvement:.2%}")

        # Step 5: Decide whether to continue
        if improvement < 0.02:  # <2% improvement
            print("Converged. Stopping iteration.")
            break

        current_model = next_model

    return current_model

# Example: Medical QA
base = load("llama-3-8b")
final_model = data_improvement_cycle(
    base,
    target_domain="medical_qa",
    iterations=3
)

# Cost per iteration: $10K-$20K
# Total: $30K-$60K
# Benefit: +15-25% performance improvement
```

---

### 6.2 Active Learning for Efficient Labeling

**4명 팀의 시간은 귀하다 → Smart labeling:**

```python
"""
Active Learning: Label most valuable samples
"""

class ActiveLearner:
    def __init__(self, model, unlabeled_pool):
        self.model = model
        self.unlabeled_pool = unlabeled_pool
        self.labeled_data = []

    def select_samples_to_label(self, n=100):
        """
        Select n most valuable samples for human labeling
        """

        # Strategy 1: Uncertainty Sampling
        # Model이 가장 uncertain한 samples
        uncertainties = []
        for sample in self.unlabeled_pool:
            # Generate multiple times
            outputs = self.model.generate(
                sample["prompt"],
                num_return_sequences=5,
                temperature=0.8
            )

            # Measure diversity (uncertainty proxy)
            diversity = measure_diversity(outputs)
            uncertainties.append((sample, diversity))

        # Sort by uncertainty (high = valuable)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        uncertain_samples = [s[0] for s in uncertainties[:n//2]]

        # Strategy 2: Representative Sampling
        # Cover diverse regions of input space
        from sklearn.cluster import KMeans

        embeddings = [embed(s["prompt"]) for s in self.unlabeled_pool]
        kmeans = KMeans(n_clusters=n//2)
        clusters = kmeans.fit_predict(embeddings)

        # One sample per cluster
        representative_samples = []
        for cluster_id in range(n//2):
            cluster_samples = [
                s for i, s in enumerate(self.unlabeled_pool)
                if clusters[i] == cluster_id
            ]
            # Closest to centroid
            representative = min(
                cluster_samples,
                key=lambda s: np.linalg.norm(
                    embed(s["prompt"]) - kmeans.cluster_centers_[cluster_id]
                )
            )
            representative_samples.append(representative)

        # Combine strategies
        selected = uncertain_samples + representative_samples

        return selected

    def label_and_train(self, n_iterations=5, samples_per_iter=100):
        """
        Iterative active learning
        """

        for iteration in range(n_iterations):
            print(f"=== Active Learning Iteration {iteration + 1} ===")

            # Select valuable samples
            to_label = self.select_samples_to_label(n=samples_per_iter)

            # Human labeling (팀원들이 직접)
            print(f"Please label {len(to_label)} samples...")
            labeled = human_labeling_interface(to_label)

            # Add to labeled set
            self.labeled_data.extend(labeled)

            # Remove from unlabeled pool
            self.unlabeled_pool = [
                s for s in self.unlabeled_pool
                if s not in to_label
            ]

            # Re-train model
            print("Training on updated labeled data...")
            self.model = finetune(
                self.model,
                self.labeled_data,
                epochs=2
            )

            # Evaluate
            score = evaluate(self.model)
            print(f"Current score: {score:.2%}")

            # Estimate value of labeling
            improvement_per_sample = score / len(self.labeled_data)
            print(f"Improvement per labeled sample: {improvement_per_sample:.4%}")

# Usage
model = load("llama-3-8b")
unlabeled_pool = load_unlabeled_data("medical_texts", n=10000)

learner = ActiveLearner(model, unlabeled_pool)

# Label 500 samples total (5 iterations x 100)
# Team labeling speed: ~10-20 samples/hour/person
# 4 people: 40-80 samples/hour
# 100 samples: 1-2 hours
# 500 samples: 6-10 hours total

learner.label_and_train(n_iterations=5, samples_per_iter=100)

# Cost: 팀 시간 6-10 hours (vs 수천 samples 직접 labeling)
# Efficiency: 5-10x better than random sampling
```

---

### 6.3 Quality Feedback Loop

**Model → User → Feedback → Better Model:**

```python
"""
Production Feedback Loop
"""

class FeedbackCollector:
    def __init__(self, model_id):
        self.model_id = model_id
        self.feedback_db = initialize_db()

    def log_interaction(self, user_query, model_response, context):
        """Log every interaction"""
        interaction = {
            "timestamp": datetime.now(),
            "model_id": self.model_id,
            "query": user_query,
            "response": model_response,
            "context": context,
            "feedback": None,  # To be filled later
        }

        self.feedback_db.insert(interaction)
        return interaction["id"]

    def collect_feedback(self, interaction_id, feedback):
        """
        Feedback types:
        - Thumbs up/down
        - Edit/correction
        - Report (safety issue)
        """
        self.feedback_db.update(
            interaction_id,
            {"feedback": feedback, "feedback_time": datetime.now()}
        )

    def analyze_feedback(self):
        """Weekly analysis"""
        week_feedback = self.feedback_db.query(
            "SELECT * FROM interactions WHERE timestamp > ?",
            (datetime.now() - timedelta(days=7),)
        )

        analysis = {
            "total_interactions": len(week_feedback),
            "thumbs_up_rate": sum(f["feedback"] == "👍" for f in week_feedback) / len(week_feedback),
            "thumbs_down_samples": [f for f in week_feedback if f["feedback"] == "👎"],
            "corrections": [f for f in week_feedback if "edit" in f["feedback"]],
            "safety_reports": [f for f in week_feedback if f["feedback"] == "⚠️"],
        }

        return analysis

    def generate_training_data(self):
        """Convert feedback to training data"""

        # High-quality positives
        positive_samples = self.feedback_db.query(
            "SELECT * FROM interactions WHERE feedback = '👍'"
        )

        # Corrected samples (most valuable!)
        corrected_samples = self.feedback_db.query(
            "SELECT * FROM interactions WHERE feedback LIKE '%edit%'"
        )

        training_data = []

        # Positive examples (as-is)
        for sample in positive_samples:
            training_data.append({
                "prompt": sample["query"],
                "response": sample["response"],
                "quality": "high"
            })

        # Corrected examples (use correction)
        for sample in corrected_samples:
            training_data.append({
                "prompt": sample["query"],
                "response": sample["feedback"]["corrected_text"],
                "quality": "high",
                "source": "human_correction"
            })

        # Negative examples (for DPO/RLHF)
        negative_samples = self.feedback_db.query(
            "SELECT * FROM interactions WHERE feedback = '👎'"
        )

        for neg, pos in zip(negative_samples, positive_samples):
            if neg["query"] == pos["query"]:  # Same query
                training_data.append({
                    "prompt": neg["query"],
                    "chosen": pos["response"],  # Good response
                    "rejected": neg["response"],  # Bad response
                    "pair": True
                })

        return training_data

# Weekly improvement cycle
collector = FeedbackCollector(model_id="medical_qa_v1")

# After 1 week in production
analysis = collector.analyze_feedback()
print(f"Thumbs up rate: {analysis['thumbs_up_rate']:.2%}")
print(f"Corrections: {len(analysis['corrections'])}")

# Generate training data from feedback
feedback_data = collector.generate_training_data()
print(f"Generated {len(feedback_data)} training samples from feedback")

# Retrain (weekly or monthly)
if len(feedback_data) > 1000:  # Threshold
    print("Enough feedback, retraining...")

    # Combine with original data
    combined = original_training_data + feedback_data

    # Retrain
    improved_model = finetune(
        current_model,
        combined,
        epochs=2,
        learning_rate=1e-5  # Lower LR for fine-tuning
    )

    # A/B test
    deploy_ab_test(current_model, improved_model, traffic_split=0.9/0.1)
```

---

## 7️⃣ 6개월 실행 로드맵

### Month 1: Foundation & Validation

**Week 1-2: 전략 수립 & 데이터**
```
Day 1-2: Team alignment
- [ ] 목표 정의 (specific domain/capability)
- [ ] Success metrics 설정
- [ ] Budget 확정 ($50K-$100K)

Day 3-5: Base model selection
- [ ] Candidate models 평가 (Llama-3, Polyglot-Ko 등)
- [ ] Quick tests (LoRA on 1K samples)
- [ ] 최종 선택

Day 6-10: Data collection
- [ ] Public datasets 수집 (50K)
- [ ] Synthetic generation 시작 (GPT-4, $500)
- [ ] Quality filtering pipeline 구축

Day 11-14: Quick validation
- [ ] 1B model pilot (10M tokens, $1K)
- [ ] 초기 평가
- [ ] Go/No-go decision
```

**Week 3-4: Data mixture optimization**
```
Day 15-21: Ablation study
- [ ] 5-10 mixture 실험 (1B models)
- [ ] 각 mixture 평가
- [ ] Best mixture 선택

Day 22-28: Data pipeline
- [ ] 자동화된 quality scoring
- [ ] Deduplication pipeline
- [ ] Final dataset 준비 (100K-500K)
```

**Deliverables:**
- ✅ Validated approach
- ✅ Optimal data mixture
- ✅ 100K-500K training samples
- ✅ Go-ahead for full training

**Cost:** $10K-$20K
**Team:** 4명 all-in

---

### Month 2-3: Main Training

**Week 5-8: SFT Training**
```
Week 5: Setup
- [ ] Infrastructure (32-64 GPUs)
- [ ] Training scripts
- [ ] Monitoring dashboards

Week 6-7: Training
- [ ] LoRA/Full fine-tuning
- [ ] Continuous monitoring
- [ ] Checkpoint evaluation

Week 8: SFT Validation
- [ ] Comprehensive evaluation
- [ ] Forgetting check
- [ ] Safety check
- [ ] Decision: Proceed to RL or iterate
```

**Week 9-12: RL Training**
```
Week 9: RL Setup
- [ ] Reward functions implementation
- [ ] Multi-task framework
- [ ] PTX data curation

Week 10-11: RL Training
- [ ] Stage 1: Easy curriculum
- [ ] Stage 2: Medium curriculum
- [ ] Stage 3: Hard curriculum
- [ ] Continuous evaluation

Week 12: RL Validation
- [ ] Target capability check
- [ ] General capability check
- [ ] Safety RL (if needed)
```

**Deliverables:**
- ✅ SFT model (instruction-following)
- ✅ RL model (domain-specialized)
- ✅ Evaluation results
- ✅ Baseline comparison

**Cost:** $30K-$80K
**Team:** 2명 training/monitoring, 2명 data/evaluation

---

### Month 4: Iteration & Improvement

**Week 13-14: Gap Analysis**
```
- Comprehensive evaluation
- Identify weaknesses
- Plan improvements
```

**Week 15-16: Targeted Improvement**
```
- Additional data for weak areas
- Focused training
- A/B testing
```

**Deliverables:**
- ✅ Improved model (+10-15%)
- ✅ Documented learnings

**Cost:** $10K-$20K

---

### Month 5: Production Preparation

**Week 17-18: Safety & Robustness**
```
- Red-teaming
- Adversarial testing
- Safety RL (if needed)
- Bias evaluation
```

**Week 19-20: Optimization**
```
- Quantization (INT8/INT4)
- Inference optimization
- Latency reduction
- Deployment setup
```

**Deliverables:**
- ✅ Production-ready model
- ✅ Deployment pipeline
- ✅ Monitoring setup

**Cost:** $5K-$10K

---

### Month 6: Launch & Feedback Loop

**Week 21-22: Soft Launch**
```
- Beta users (10-50)
- Feedback collection
- Rapid iteration
```

**Week 23-24: Scale Up**
```
- Public launch
- Marketing push
- Continuous monitoring
- Feedback-driven improvement
```

**Deliverables:**
- ✅ Launched product
- ✅ User feedback
- ✅ Iteration plan

---

### 전체 Timeline Summary

```
Month 1: Foundation (Validation + Data)
Month 2-3: Training (SFT + RL)
Month 4: Iteration (Improvement)
Month 5: Production Prep (Safety + Optimization)
Month 6: Launch (Beta + Scale)

Total: 6 months
Budget: $50K-$130K
Team: 4명
Success rate: 70% (with this plan)
```

---

## 8️⃣ 성공 사례 & 실패 방지

### 8.1 What Good Looks Like

**Case: 한국 의료 AI (가상 시나리오)**

```
Team: 4명 (2 ML engineers, 1 도메인 전문가, 1 product)
Timeline: 5개월
Budget: $80K

Month 1:
- Base: Llama-3-8B
- Data: 50K 의료 QA (public) + 30K synthetic (GPT-4, $300)
- Pilot: 1B model, Medical QA 45% → 62% (+17%)
- Decision: GO

Month 2-3:
- SFT: 80K samples, LoRA (r=16)
- RL: Medical diagnosis tasks with rule-based rewards
- Multi-task: 60% medical, 40% general
- Result: Medical QA 72%, MMLU 64% (no forgetting)

Month 4:
- Gap: Safety issues (hallucination in drug names)
- Fix: Safety RL + manual curation of drug database
- Result: Safety 92%

Month 5:
- Quantization: INT8 (2x faster)
- Deploy: Hospital pilot (5 doctors)
- Feedback: Positive, some corrections

Month 6:
- Incorporate feedback (200 corrections)
- Retrain: Medical QA 72% → 76%
- Launch: 20 hospitals

Results:
- Medical QA: 76% (vs GPT-4 70% on Korean medical)
- Cost: $75K total
- Revenue: $15K/month (20 hospitals x $750/month)
- ROI: Break-even in 5 months, 200%+ year 1
- Impact: 명확한 niche dominance
```

**Key Success Factors:**
1. Clear niche (한국 의료)
2. Domain expert in team
3. Iterative approach
4. Feedback loop
5. Realistic scope

---

### 8.2 Common Pitfalls & Avoidance

**Pitfall 1: Scope Creep**
```
❌ Bad:
"Let's make it good at medical AND legal AND finance..."

✅ Good:
"Let's dominate Korean medical QA first.
 Then consider expansion."

Prevention:
- Single clear metric
- Weekly scope review
- Ruthless prioritization
```

**Pitfall 2: Data Quality Neglect**
```
❌ Bad:
"We have 1M samples, let's train!"

✅ Good:
"We have 1M samples, let's:
 1. Filter to 100K high-quality
 2. Ablate mixtures
 3. Then train"

Prevention:
- Quality > quantity mantra
- Mandatory quality checks
- Sample inspection (random 100)
```

**Pitfall 3: No Forgetting Monitoring**
```
❌ Bad:
"Training loss going down, great!"

✅ Good:
"Training loss down, but MMLU also down 10%!
 → Add multi-task immediately"

Prevention:
- Baseline checkpoints
- Every 100 steps evaluation
- Auto-alerts on regression
```

**Pitfall 4: Premature Scaling**
```
❌ Bad:
"Pilot showed 5% improvement, let's train 70B!"

✅ Good:
"Pilot showed 5% improvement. Let's:
 1. Understand why only 5%
 2. Try data/approach changes
 3. Get to 15% with 7B
 4. Then consider scaling"

Prevention:
- Validate thoroughly at small scale
- Scaling is last resort, not first
```

**Pitfall 5: Ignoring User Feedback**
```
❌ Bad:
"Model is 80% accurate on benchmark, ship it!"

✅ Good:
"Model is 80% on benchmark, but users complaining about X.
 → Investigate X, add X data, retrain"

Prevention:
- Feedback collection from day 1
- Weekly feedback review
- Metrics = benchmark AND user satisfaction
```

---

## 9️⃣ 핵심 Takeaways (TL;DR)

### For 4-Person Team

**✅ DO:**
1. **Post-training only**: Base model + customization
2. **Multi-task training**: Never let model forget
3. **Quality data**: 100K great > 1M mediocre
4. **Rapid iteration**: Pilot → Validate → Scale
5. **Niche focus**: Dominate one area
6. **Feedback loop**: Users → Data → Better model
7. **Continuous monitoring**: Catch forgetting early

**❌ DON'T:**
1. **Pre-training**: Impossible with your resources
2. **General competition**: Can't beat GPT-4 broadly
3. **Data hoarding**: Quality > quantity
4. **Single-shot training**: Must iterate
5. **Scope creep**: Stay focused
6. **Ignoring baselines**: Track forgetting
7. **Perfectionism**: Ship at 80%, iterate to 90%

---

### Resource Allocation (6 months, $100K)

```
Data: $20K (20%)
├─ Public curation: $5K
├─ Synthetic generation: $10K
└─ Quality filtering: $5K

Training: $60K (60%)
├─ Ablations: $10K
├─ SFT: $20K
├─ RL: $25K
└─ Iteration: $5K

Evaluation: $10K (10%)
├─ Benchmarks: $5K
└─ Custom eval: $5K

Infrastructure: $10K (10%)
├─ Storage: $2K
├─ Monitoring: $3K
└─ Misc: $5K
```

---

### Success Metrics

**Technical:**
- Domain task: 75%+ (vs 65% base)
- General MMLU: >60% (no catastrophic forgetting)
- Safety: >90%
- Latency: <500ms

**Business:**
- ROI: >3x within 12 months
- Users: 50+ paying customers
- Revenue: $10K+ monthly
- Churn: <10%

**Team:**
- Learnings documented
- Process repeatable
- Next model faster (3-4 months)

---

## 🚀 시작하기

### Week 1 Checklist

```
[ ] Day 1: Team Meeting
    [ ] 목표 합의 (specific domain)
    [ ] Success metrics 정의
    [ ] Budget 확인
    [ ] Roles 분담
        - Person 1: Data lead
        - Person 2: Training lead
        - Person 3: Evaluation lead
        - Person 4: Product/domain expert

[ ] Day 2: Base Model Selection
    [ ] Candidate models 리스트
    [ ] Quick tests (1-2 hours each)
    [ ] Decision

[ ] Day 3-4: Data Strategy
    [ ] Public datasets identified
    [ ] Synthetic generation plan
    [ ] Quality criteria defined

[ ] Day 5: Pilot Setup
    [ ] Infrastructure (8-16 GPUs)
    [ ] Codebase setup
    [ ] Monitoring tools

[ ] Week 2: Execute Pilot
    [ ] Train 1B model on 10K samples
    [ ] Evaluate
    [ ] Go/No-go decision
```

---

## 📚 Further Resources

**Papers (이미 분석함):**
- GLM-4.5: Multi-stage post-training
- MiniMax-M1: CISPO, Curriculum learning
- Kimi K2: PTX Loss, Verifiable rewards
- SmolLM3 Playbook: Training reality

**Tools:**
- Training: HuggingFace TRL, Axolotl
- RL: OpenRLHF, TRL
- Evaluation: lm-evaluation-harness
- Monitoring: Weights & Biases

**Community:**
- Korean NLP: 한국어 모델 커뮤니티
- Open LLM Leaderboard
- HuggingFace Forums

---

## 💪 Closing Thoughts

**4명 팀의 강점:**
- 빠른 iteration
- 높은 집중도
- 명확한 communication
- Niche expertise

**우리가 이길 수 있는 방법:**
- 대기업이 안 하는 niche
- 빠른 실험 → 학습 → 개선
- 사용자 피드백 직접 수렴
- Post-training specialization

**Remember:**
> "The papers show success; we show the path to success."

이 전략을 따르면, 4명 팀도 특정 영역에서 월드클래스 모델을 만들 수 있습니다.

**Good luck! 🚀**

---

**작성일**: 2025-11-12
**기반**: GLM-4.5, MiniMax-M1, Kimi K2, SmolLM3 Playbook 전체 분석
**대상**: 4명 작은 팀
**목표**: 6개월 내 production-ready specialized model
