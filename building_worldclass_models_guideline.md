# 월드클래스 모델 구축 실전 가이드: 작은 팀을 위한 프로세스

## 📋 가이드 개요

이 가이드는 GLM-4.5, MiniMax-M1, Kimi K2/K2 Thinking 같은 월드클래스 모델들의 실제 훈련 경험과 Hugging Face SmolLM3 Training Playbook의 현실적 교훈을 바탕으로, **작은 팀이 실제로 동작하는 효율적인 프로세스**를 제공합니다.

### 이 가이드의 대상

**"작은 팀"의 정의:**
- 2-10명의 핵심 인력
- 제한된 GPU 리소스 (8-128 GPUs)
- 제한된 예산 ($10K-$1M)
- 빠른 iteration 필요

**이 가이드가 아닌 것:**
- ❌ 학술 논문 (이론적 설명)
- ❌ 튜토리얼 (코드 step-by-step)
- ❌ API 문서

**이 가이드는:**
- ✅ 실전 프로세스와 의사결정 프레임워크
- ✅ Iterative, checklist 기반
- ✅ 실패 시나리오와 복구 전략 포함
- ✅ 월드클래스 모델들이 실제로 한 것 (논문이 말하지 않은 것)

---

## 🎯 핵심 원칙 (논문들이 증명한 것)

### 원칙 1: Data > Architecture (Always)

**증거:**
> SmolLM3 Playbook: "Data obsession consistently outweighs architectural innovation"

**실제 사례:**
- Kimi K2: Knowledge rephrasing으로 token efficiency 증가
- GLM-4.5: Repository-level code training이 성능 향상의 핵심
- MiniMax-M1: 53K SynLogic synthetic data가 게임 체인저

**행동 지침:**
```
아키텍처 혁신에 시간 쓰기 전에:
1. 데이터 품질 10% 향상 시도 (1-2주)
2. 데이터 mixture 실험 (2-4주)
3. Synthetic data generation (1-2개월)

그래도 부족할 때만 아키텍처 변경 고려
```

**리소스 할당:**
- 60% Data curation & ablations
- 20% Training & infrastructure
- 10% Algorithm development
- 10% Post-training

---

### 원칙 2: Validate Before Investing

**배경:**
- SmolLM3: 1T tokens 후 restart (수백만 달러 손실)
- 대부분의 실패는 초기 검증 부족

**행동 지침:**
```
대규모 훈련 전:
1. 1B-7B 모델로 proof-of-concept (1-2주, $1K-$10K)
2. 100B tokens 파일럿 (1-4주, $10K-$50K)
3. 성공 확인 후 scale up

실패 시 pivot, 성공 시 proceed
```

**검증 체크리스트:**
- [ ] Data pipeline 안정성 (throughput, 에러율)
- [ ] Loss curve 정상 (spike 없음, 수렴 경향)
- [ ] Preliminary benchmark (목표의 60-70% 달성)
- [ ] Infrastructure (GPU 통신, checkpoint)
- [ ] Team capability (debugging, monitoring)

---

### 원칙 3: Reserve 50% Buffer (Not 20-30%)

**배경:**
- SmolLM3: "Reserve 20-30% buffer" 권장
- 하지만 novel approaches는 충분하지 않음

**실제 필요:**
```
Standard approach (well-tested):
- Timeline: +20-30%
- Budget: +20-30%

Novel approach (CISPO, MuonClip 등):
- Timeline: +50-100% (6개월 → 12-18개월)
- Budget: +50-100% ($10M → $20-30M)
```

**작은 팀 적용:**
```
초기 계획: 3개월, $100K
Realistic: 6개월, $200K

항상 2배 준비, 1배 약속
```

---

### 원칙 4: Iteration Speed > Team Size

**증거:**
- SmolLM3: 2-3명 core team
- 빠른 실험 iteration이 승리의 핵심

**비교:**
```
Team A (작지만 빠름):
- 3명
- Daily experiments
- Weekly pivots
- 성공

Team B (크지만 느림):
- 20명
- Weekly meetings
- Monthly decisions
- 실패
```

**행동 지침:**
```
팀 구성:
- 2-3명 core researchers (decision makers)
- 2-5명 engineers (infrastructure, data)
- 1-2명 SRE (monitoring, debugging)

총 5-10명, flat hierarchy
Decision in hours, not weeks
```

---

### 원칙 5: Ablations Are Not Optional

**증거:**
- SmolLM3: 58% of compute on ablations
- 월드클래스 모델: 50-100%

**현실:**
```
"Main run만 하면 되지 않나?"
→ NO! Every decision needs validation

의문: "이 data mixture가 최선인가?"
→ 5-10 ablations (각 100B tokens)
→ 수천 GPU-hours
→ 하지만 유일한 방법
```

**작은 팀 전략:**
```
Full ablation 불가능 시:
1. Small-scale ablations (1B-7B models)
2. Short runs (10B-50B tokens)
3. Quick validation
4. Scale up winner

비용 10-20%로 위험 80% 감소
```

---

## 🌳 의사결정 트리: Train vs Alternatives

### Level 1: 새 모델이 정말 필요한가?

```
시작: 우리는 X 능력이 필요하다

질문 1: 기존 모델로 가능한가?
├─ Yes → Use existing model (GPT-4, Claude, etc.)
│         Cost: $0-$10K/month API
│         Time: 즉시
│
└─ No → 질문 2로

질문 2: Prompt engineering으로 해결 가능한가?
├─ Yes → Few-shot, CoT, ReACT 시도
│         Cost: $0
│         Time: 1-2주
│
└─ No → 질문 3으로

질문 3: RAG로 해결 가능한가?
├─ Yes → Vector DB + Retrieval setup
│         Cost: $5K-$50K (infrastructure)
│         Time: 2-4주
│
└─ No → 질문 4로

질문 4: 오픈소스 모델 fine-tuning으로 가능한가?
├─ Yes → LoRA/QLoRA fine-tuning
│         Cost: $1K-$50K
│         Time: 1-4주
│         → Level 2로 (Post-training)
│
└─ No → Level 3으로 (From Scratch Training 필요)
```

---

### Level 2: Post-Training 의사결정

```
시작: 기존 모델을 개선하고 싶다

질문 1: 개선 목표가 명확한가?
├─ Yes → 질문 2로
│
└─ No → ❌ STOP
          더 명확히 정의하고 돌아오라
          Vague goal = Wasted resources

질문 2: 현재와 목표의 Gap이 얼마나 큰가?
├─ Small gap (<10%) → ⚠️ WARNING
│   - ROI 낮음
│   - Prompt engineering 먼저 시도
│   - 정말 필요한지 재검토
│
└─ Large gap (>20%) → 질문 3으로

질문 3: 어떤 타입의 개선인가?

A. Narrow Domain Specialization
   (예: Medical, Legal, Finance)
   ✅ 추천: LoRA fine-tuning
   비용: $5K-$50K
   시간: 2-6주
   조건:
   - High-quality domain data 확보 가능
   - Clear economic value
   - Verifiable metrics
   → Section 7로 (Post-training)

B. 언어 확장
   (예: 영어 → 한국어)
   ✅ 추천: Continual pre-training
   비용: $50K-$500K
   시간: 1-3개월
   조건:
   - Target language corpus (10B+ tokens)
   - Market opportunity 명확
   → Section 7로 (Post-training)

C. 약점 보완
   (예: Tool-use 50% → 80%)
   ✅ 추천: Targeted RL
   비용: $20K-$200K
   시간: 1-2개월
   조건:
   - Gap >20%
   - Verifiable reward signal
   → Section 7로 (Post-training)

D. 이미 SOTA 더 강화
   (예: AIME 91% → 95%)
   ❌ 비추천: ROI 매우 낮음
   Alternative:
   - Ensemble methods
   - Self-consistency (inference-time)

E. 새 Modality 추가
   (예: Text → Multimodal)
   ❌ 불가능: Architecture redesign 필요
   → Level 3로 (새 모델 필요)
```

---

### Level 3: From-Scratch Training 의사결정

```
시작: 완전히 새로운 모델이 필요하다

⚠️ WARNING: 이것은 가장 expensive & risky 경로
Minimum: $100K-$1M, 3-12개월

질문 1: Pilot로 검증했는가?
├─ No → ❌ STOP
│         먼저 1B-7B 모델로 proof-of-concept
│         비용: $1K-$10K
│         성공 후 돌아오라
│
└─ Yes → 질문 2로

질문 2: 리소스가 충분한가?

Minimum Requirements:
- Budget: $100K-$1M (ablations 포함)
- GPUs: 32-512 H100/A100 (최소 1-3개월)
- Team: 3-10명 (연구, 엔지니어링, SRE)
- Time: 3-12개월 (버퍼 포함)
- Expertise: Distributed training, RL, data curation

├─ Yes → 질문 3으로
│
└─ No → ❌ STOP
          리소스 확보 후 재검토
          또는 Post-training 대안 고려

질문 3: 어떤 규모로 시작하는가?

A. Small Model (1B-7B)
   ✅ 추천: Small team 시작점
   비용: $50K-$200K
   시간: 2-4개월
   Use case:
   - Domain-specific (medical, legal)
   - Edge deployment
   - Distillation target
   → Section 4-6으로

B. Medium Model (13B-70B)
   ⚠️ 신중: 상당한 리소스 필요
   비용: $200K-$1M
   시간: 4-8개월
   Use case:
   - General-purpose competitive
   - API service
   → Section 4-6으로

C. Large Model (100B+, MoE)
   ❌ 매우 신중: 작은 팀에게 매우 위험
   비용: $1M-$10M+
   시간: 6-18개월
   Use case:
   - Frontier research
   - Major company backing
   필수조건:
   - 이전에 medium model 성공 경험
   - Dedicated infrastructure team
   - $5M+ budget 확보
   → Section 4-6으로 (extreme caution)
```

---

## 🚀 Phase 0: Validation & Planning (가장 중요!)

> "Hours of planning save weeks of wasted training"

### 0.1 Feasibility Check

#### 기술적 Feasibility

**질문 체크리스트:**

```
[ ] 1. 데이터 확보 가능한가?
    - 필요량: Pre-training 최소 100B tokens
    - 품질: High-quality, diverse
    - 접근성: Legal, privacy 이슈 없음
    - 비용: Curation cost <30% of total budget

[ ] 2. 아키텍처가 목표에 적합한가?
    - Dense vs MoE (MoE 추천)
    - Context length (128K standard, 1M if specific need)
    - Novel architecture? (추가 6-12개월 고려)

[ ] 3. 훈련 인프라 확보 가능한가?
    - GPU access (owned, cloud, grants)
    - Network (InfiniBand for >32 GPUs)
    - Storage (100TB+ for large-scale)
    - Monitoring tools

[ ] 4. 팀 역량 충분한가?
    - Distributed training experience
    - Debugging skills (2am sessions)
    - Data engineering
    - RL expertise (for post-training)

[ ] 5. Timeline realistic한가?
    - 계획의 2배로 설정했는가?
    - Buffer 50% 포함했는가?
    - Team vacation, sick days 고려했는가?
```

**Red Flags (Stop signals):**
- ❌ "데이터는 나중에 구하면 되지 않나?" → STOP
- ❌ "일단 시작하고 보자" → STOP
- ❌ "다른 팀이 3개월 걸렸으니 우리도..." → STOP (2배로 계획)
- ❌ "Budget 부족하면 중간에 더 받으면..." → STOP (확보 후 시작)

---

#### 경제적 Feasibility

**ROI 계산 프레임워크:**

```python
# Step 1: Total Cost 추정
training_cost = gpu_hours * gpu_price  # Main run
ablation_cost = training_cost * 0.5    # 최소 50%
debug_cost = training_cost * 0.3       # Restart, debug
data_cost = training_cost * 0.2        # Curation
human_cost = team_size * months * salary

total_cost = (training_cost + ablation_cost + debug_cost +
              data_cost + human_cost)

# Reality: 계획의 2-3x
realistic_cost = total_cost * 2.5

# Step 2: Expected Value 추정
market_size = ?  # Target market
market_share = ?  # Realistic (보통 1-5%)
revenue_per_user = ?
lifetime_value = ?

expected_revenue = market_size * market_share * lifetime_value

# Step 3: ROI 계산
roi = expected_revenue / realistic_cost

# Decision
if roi >= 3:
    return "✅ GO"
elif roi >= 1.5:
    return "⚠️ MAYBE (높은 위험)"
else:
    return "❌ NO-GO (대안 찾기)"
```

**실제 예시:**

**Case A: Medical Domain Model (7B)**
```
Cost:
- Training: $20K (32 A100, 2주)
- Ablations: $10K
- Data curation: $50K (의사 annotation)
- Team (3명 x 3개월): $75K
- Total: $155K
- Realistic: $400K (2.5x)

Revenue:
- Medical AI market: $1B
- Our niche (한국 병원): $10M
- Market share: 5% (차별화된 경쟁력)
- Expected: $500K/year
- 5년 LTV: $2.5M

ROI: $2.5M / $400K = 6.25x ✅ GO!
```

**Case B: General QA Model (70B)**
```
Cost:
- Training: $500K
- Ablations: $250K
- Data: $100K
- Team (6명 x 6개월): $300K
- Total: $1.15M
- Realistic: $2.9M

Revenue:
- General QA market: $10B
- Competing with GPT-4, Claude
- Market share: 0.1% (매우 낙관적)
- Expected: $10M/year?
- 하지만 경쟁 심화, commoditization

ROI: Uncertain, likely <2x ❌ NO-GO
Alternative: Focus on niche specialization
```

---

### 0.2 Resource Planning

#### GPU 요구사항 계산

**공식 (rough estimate):**

```python
def estimate_gpu_hours(
    model_size_b,      # Billion parameters
    training_tokens_t,  # Trillion tokens
    gpu_type="H100"
):
    """
    Based on Chinchilla scaling laws and empirical data
    """
    # FLOPs per token
    flops_per_token = 6 * model_size_b * 1e9  # 6N (forward + backward)

    # Total FLOPs
    total_flops = flops_per_token * training_tokens_t * 1e12

    # GPU throughput
    gpu_flops = {
        "H100": 1000e12,  # 1 PetaFLOPs (theoretical)
        "A100": 312e12,   # 312 TeraFLOPs
        "V100": 125e12,   # 125 TeraFLOPs
    }

    # Utilization (realistic)
    utilization = 0.4  # 40% (communication overhead, 등)

    effective_flops = gpu_flops[gpu_type] * utilization

    # GPU-hours (single GPU)
    gpu_hours_single = total_flops / effective_flops / 3600

    return gpu_hours_single

# Example: GLM-4.5 scale
model_size = 355  # 355B total params (MoE)
tokens = 23  # 23T tokens

gpu_hours = estimate_gpu_hours(355, 23, "H100")
print(f"Single GPU: {gpu_hours:,.0f} hours")
print(f"32 GPUs: {gpu_hours/32:,.0f} hours ({gpu_hours/32/24:.1f} days)")
print(f"256 GPUs: {gpu_hours/256:,.0f} hours ({gpu_hours/256/24:.1f} days)")

# Output (대략):
# Single GPU: 8,000,000 hours
# 32 GPUs: 250,000 hours (10,417 days) ❌ 불가능
# 256 GPUs: 31,250 hours (1,302 days) ❌ 여전히 매우 김
# 1024 GPUs: 7,812 hours (326 days) ✅ Feasible (하지만 매우 expensive)
```

**작은 팀 realistic targets:**

| Model Size | Tokens | GPUs | Duration | Cost (H100 @$2/hr) |
|-----------|--------|------|----------|-------------------|
| **1B** | 100B | 8 | 2주 | $5K |
| **3B** | 100B | 16 | 2주 | $10K |
| **7B** | 1T | 32 | 2개월 | $100K |
| **13B** | 1T | 64 | 2개월 | $200K |
| **70B** | 2T | 256 | 3개월 | $1.2M |

**💡 Tip:** MoE를 사용하면 activated params는 작게 유지하면서 capacity 증가
- 예: 355B total, 32B activated → 효율성 10x

---

#### Timeline 계획

**Realistic Timeline Template:**

```
Small Model (1B-7B), 100B-1T tokens:

Week 1-2: Setup & Validation
- Infrastructure setup
- Data pipeline testing
- Small-scale pilot (1B, 10B tokens)

Week 3-4: Data Ablations
- Mixture experiments (5-10 ablations)
- Each: 10B tokens, 1-2 days
- Winner selection

Week 5-12: Main Training (2 months)
- Pre-training: 1T tokens
- Continuous monitoring
- Checkpoint every 50B-100B tokens
- Buffer for debugging

Week 13-16: Post-training (1 month)
- SFT: 1-2주
- RL: 2-3주
- Evaluation: ongoing

Week 17-18: Evaluation & Iteration
- Full benchmark suite
- Regression checks
- Final tuning

Total: 4-5개월 (계획: 3개월 + 50% buffer)
```

---

### 0.3 Risk Assessment

**리스크 매트릭스:**

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|----------|
| **데이터 품질 문제** | 높음 | 높음 | Pilot으로 조기 검증, 다양한 소스 |
| **Training instability** | 중간 | 매우 높음 | Frequent checkpoints, conservative LR |
| **Catastrophic forgetting** | 중간 | 중간 | Multi-task, PTX loss, curriculum |
| **Infrastructure failure** | 높음 | 중간 | Redundancy, quick recovery plan |
| **Checkpoint corruption** | 낮음 | 높음 | Multiple backup, validation |
| **Novel algorithm 실패** | 높음 | 높음 | Standard baseline 준비, pivot plan |
| **Team burnout** | 중간 | 높음 | Sustainable pace, rotation |
| **Budget 초과** | 높음 | 매우 높음 | 50% buffer, staged funding |
| **Timeline 초과** | 매우 높음 | 중간 | 2x planning, milestone-based |
| **ROI 미달** | 중간 | 매우 높음 | Clear go/no-go criteria, pivot early |

**Critical Risks (즉시 중단 트리거):**

1. **Data 확보 불가능**:
   - Trigger: Pilot 단계에서 quality 기준 미달
   - Action: 프로젝트 중단 또는 data strategy 완전 재설계

2. **Technical blocker**:
   - Trigger: 2주 이상 progress 없음
   - Action: 외부 전문가 영입 또는 접근법 변경

3. **Budget 50% 초과**:
   - Trigger: Halfway point에서 75% budget 소진
   - Action: Scope 축소 또는 추가 펀딩 확보

4. **Team loss**:
   - Trigger: Core member 1명 이상 이탈
   - Action: 즉시 채용 또는 프로젝트 pause

---

### 0.4 Go/No-Go Decision

**최종 체크리스트:**

```
[ ] Feasibility
    [ ] 기술적으로 가능함 (Pilot 검증)
    [ ] 데이터 확보 가능
    [ ] 팀 역량 충분
    [ ] Infrastructure 준비됨

[ ] Economics
    [ ] ROI >= 3x
    [ ] Budget 확보 (realistic cost 2.5x 포함)
    [ ] Revenue model 명확
    [ ] Market validation

[ ] Risk Management
    [ ] Critical risks 완화 전략 수립
    [ ] Contingency plan 준비
    [ ] Kill criteria 명확
    [ ] Stakeholder alignment

[ ] Team Readiness
    [ ] Commitment 확보 (3-12개월)
    [ ] Roles 명확히 정의
    [ ] On-call rotation 계획
    [ ] Work-life balance plan

[ ] Timeline
    [ ] Milestone 명확
    [ ] Buffer 50% 포함
    [ ] Dependencies 파악
    [ ] External constraints 고려
```

**Decision:**
- ✅ **All checked → GO**: Proceed to Phase 1
- ⚠️ **80-90% checked → CAUTION**: Address gaps, then GO
- ❌ **<80% checked → NO-GO**: Fix fundamental issues or cancel

---

## 📊 Phase 1: Data Foundation (가장 중요한 단계)

> "Data obsession consistently outweighs architectural innovation"
> — SmolLM3 Training Playbook

### 1.1 Data Philosophy

**핵심 인사이트 (논문들로부터):**

1. **Quality > Quantity** (Kimi K2)
   - Knowledge rephrasing으로 token efficiency 증가
   - Math → Learning-note style 변환
   - 15.5T tokens으로 23T GLM-4.5와 경쟁

2. **Diverse > Clean (in moderation)** (SmolLM3)
   - "Using highest quality data doesn't always yield strongest models"
   - Pure arXiv < Balanced mixture
   - 60-70% General + 15-20% Domain + 15-20% Code

3. **Synthetic Data는 Strategic Tool** (모든 논문)
   - GLM-4.5: Synthetic reasoning in mid-training
   - MiniMax-M1: 53K SynLogic logical reasoning
   - Kimi K2: 20,000+ synthetic tools

---

### 1.2 Data Collection Strategy

#### Step 1: Define Data Requirements

**Specification Template:**

```yaml
data_requirements:
  # Volume
  total_tokens: 1T  # 1 trillion for 7B model

  # Domains
  domains:
    - name: "general_web"
      percentage: 60
      sources: ["CommonCrawl", "C4", "FineWeb-Edu"]
      quality_threshold: 0.7

    - name: "code"
      percentage: 15
      sources: ["GitHub", "StackOverflow"]
      languages: ["python", "javascript", "java", "cpp"]
      quality_filters: ["stars > 10", "license: MIT/Apache"]

    - name: "math_science"
      percentage: 15
      sources: ["arXiv", "Math StackExchange", "scientific papers"]
      subjects: ["mathematics", "physics", "cs"]

    - name: "specialized"
      percentage: 10
      sources: ["domain-specific corpus"]
      description: "Target capability"

  # Quality criteria
  quality:
    - deduplication: "exact + fuzzy (85% similarity)"
    - language: "primary + others (95% + 5%)"
    - toxicity: "filter > 0.5 threshold"
    - perplexity: "filter outliers (top 5%, bottom 5%)"
    - length: "min 100 tokens, max 100K tokens"

  # Format
  format: "jsonl"
  tokenizer: "BPE with vocab 50K-100K"
```

---

#### Step 2: Data Sourcing

**Common Sources:**

**A. Public Datasets (Free)**
```
General:
✅ CommonCrawl (web scrape, 수십 TB)
✅ C4 (Colossal Clean Crawled Corpus, 350GB)
✅ FineWeb-Edu (Hugging Face, 1.3T tokens, education-filtered)
✅ RedPajama (1.2T tokens, LLaMA replica)

Code:
✅ The Stack (3TB, permissive licenses)
✅ CodeParrot (GitHub filtered)

Math/Science:
✅ arXiv (papers)
✅ OpenWebMath (14B tokens math-focused)
✅ MATH dataset (12K problems)
✅ GSM8K (grade school math)

Instruction:
✅ FLAN (1.8M examples)
✅ OpenOrca (1M GPT-4 augmented)
✅ Alpaca (52K instruction-following)
```

**B. Proprietary/Licensed (Paid)**
```
Books:
- Books3 (❌ copyright issues)
- Licensed book datasets ($$$)

News:
- News aggregators (licensing required)

Specialized:
- Medical: PubMed (free), MIMIC (restricted)
- Legal: CaseLaw (free), Westlaw ($$$)
- Finance: SEC filings (free), Bloomberg ($$$)
```

**C. Synthetic Generation (DIY)**
```
Tools:
- GPT-4/Claude API ($$$)
- 자체 모델로 self-improve
- Rule-based generation (grammar, templates)

Use cases:
- Reasoning chains (GLM-4.5 style)
- Logical problems (MiniMax-M1's SynLogic)
- Tool-use trajectories (Kimi K2 style)
```

---

#### Step 3: Data Curation Pipeline

**실전 Pipeline:**

```python
"""
Data Curation Pipeline
"""

# Stage 1: Collection (병렬)
raw_data = [
    download_commonweb(),
    download_github(),
    download_arxiv(),
    generate_synthetic()
]

# Stage 2: Cleaning (각 source별)
for source in raw_data:
    # A. Format normalization
    normalized = normalize_format(source)  # UTF-8, consistent structure

    # B. Language detection
    filtered_lang = filter_language(normalized, target_langs=["en", "zh", ...])

    # C. Deduplication
    #    - Exact: Hash-based (빠름)
    #    - Fuzzy: MinHash LSH (느림, 중요)
    deduplicated = deduplicate(filtered_lang, method="minhash", threshold=0.85)

    # D. Quality filtering
    quality_filtered = filter_quality(deduplicated, criteria={
        "perplexity": (10, 1000),  # Remove too easy/hard
        "length": (100, 100000),
        "toxicity": 0.5,
        "repetition": 0.3
    })

    # E. PII removal
    anonymized = remove_pii(quality_filtered)  # Email, phone, SSN, etc.

    # F. Safety filtering
    safe = filter_unsafe(anonymized)  # Violence, NSFW, etc.

    # Stage 3: Tokenization & Packing
    tokenized = tokenize(safe, tokenizer=your_tokenizer)
    packed = pack_sequences(tokenized, max_length=2048)  # Context packing

    # Stage 4: Shuffling
    shuffled = shuffle(packed, seed=42)  # Reproducibility

    # Stage 5: Storage
    save_shards(shuffled, output_dir="data/processed", shard_size=1GB)

# Stage 6: Mixture Construction (이것이 핵심!)
# 별도 섹션에서 설명
```

---

### 1.3 Data Mixture Optimization (Critical!)

> "Most impact comes from data mixture, not architecture"

**논문들의 mixture 전략:**

**GLM-4.5 (추정):**
```yaml
pre_training:
  general_web: 60-65%
  code: 15-20%
  math_science: 10-15%
  multilingual: 5-10%

mid_training:
  repo_level_code: 40%
  synthetic_reasoning: 30%
  long_context: 20%
  agent_tasks: 10%
```

**Kimi K2 (추정):**
```yaml
pre_training:
  general_en: 40%
  general_zh: 25%
  other_langs: 15%
  code: 15%
  tool_use_synthetic: 5%
```

**작은 팀 approach:**

#### Ablation Framework

```python
"""
Data Mixture Ablation Strategy
"""

# Step 1: Define candidate mixtures
mixtures = [
    {"general": 70, "code": 15, "math": 10, "specialized": 5},  # Baseline
    {"general": 60, "code": 20, "math": 15, "specialized": 5},  # More code
    {"general": 60, "code": 15, "math": 20, "specialized": 5},  # More math
    {"general": 50, "code": 20, "math": 20, "specialized": 10}, # Balanced
    {"general": 65, "code": 15, "math": 5, "specialized": 15},  # More specialized
]

# Step 2: Small-scale ablations (1B model, 10B tokens)
results = []
for mixture in mixtures:
    # Train small model (1-2 days, $1K-$2K each)
    model = train_small_model(
        model_size="1B",
        tokens="10B",
        mixture=mixture,
        gpus=8,
        duration_hours=24
    )

    # Evaluate on target benchmarks
    scores = evaluate(model, benchmarks=[
        "MMLU", "HumanEval", "GSM8K", "your_custom_benchmark"
    ])

    results.append({"mixture": mixture, "scores": scores})

# Step 3: Select winner
winner = max(results, key=lambda x: x["scores"]["avg"])

# Step 4: Verify at larger scale (7B model, 100B tokens)
verification_model = train_model(
    model_size="7B",
    tokens="100B",
    mixture=winner["mixture"],
    gpus=32,
    duration_weeks=2
)

verification_scores = evaluate(verification_model, benchmarks=full_suite)

# Step 5: Final decision
if verification_scores > threshold:
    final_mixture = winner["mixture"]
else:
    # Pivot: Try refined mixtures around winner
    refined_mixtures = refine_around(winner["mixture"])
    # Repeat ablations...
```

**Cost Analysis:**
- 5 ablations x $2K = $10K
- 1 verification x $20K = $20K
- Total: $30K
- Benefit: 최적 mixture 발견 → Final training 성공률 10-20% 증가
- ROI: Huge (final training cost $100K-$500K 고려 시)

---

### 1.4 Synthetic Data Generation

**언제 합성 데이터가 필요한가?**

✅ **Use cases:**
1. **Reasoning chains**: Step-by-step thought process
2. **Domain expertise**: 전문가 annotation 비용 높을 때
3. **Tool-use trajectories**: Real trajectories 부족
4. **Data augmentation**: Existing data 변형
5. **Long-tail scenarios**: 희귀한 상황

❌ **피해야 할 경우:**
1. General knowledge (web scraping이 더 좋음)
2. Factual information (hallucination 위험)
3. 충분한 real data 존재

---

#### Method 1: LLM-based Generation

**Kimi K2 Style (Agentic Data):**

```python
"""
Synthetic Agentic Data Generation
"""

def generate_agentic_task():
    # Step 1: Domain evolution (다양한 도메인의 tools 생성)
    domains = ["search", "calculator", "python", "web_browser",
               "file_system", "database", "api_caller"]

    tools_per_domain = 20  # 20 tools per domain

    synthetic_tools = []
    for domain in domains:
        for i in range(tools_per_domain):
            # GPT-4로 tool specification 생성
            tool_spec = gpt4.generate(prompt=f"""
            Generate a realistic tool specification for {domain} domain.
            Tool #{i+1}.

            Format:
            - name: tool_name
            - description: what it does
            - parameters: list of params with types
            - returns: return type
            - example_usage: code example
            """)

            synthetic_tools.append(tool_spec)

    # Step 2: Task generation with rubric
    task = gpt4.generate(prompt=f"""
    Generate a complex task that requires using these tools:
    {random.sample(synthetic_tools, k=5)}

    Task should:
    - Be realistic and practical
    - Require 3-5 tool calls
    - Have clear success criteria (rubric)

    Output format:
    - task_description: ...
    - required_tools: [...]
    - success_rubric: {{criteria: threshold}}
    """)

    # Step 3: User simulation + Trajectory generation
    trajectory = generate_trajectory(
        task=task,
        tools=synthetic_tools,
        model=base_model,  # 이전 version model
        max_turns=10
    )

    # Step 4: Execution validation (hybrid simulated + real)
    result = execute_trajectory(trajectory, sandbox=True)

    # Step 5: Rubric-based evaluation
    score = evaluate_with_rubric(result, task["success_rubric"])

    # Only keep high-quality trajectories
    if score > 0.8:
        return {
            "task": task,
            "trajectory": trajectory,
            "result": result,
            "score": score
        }
    else:
        return None  # Discard low-quality

# Generate thousands
synthetic_dataset = []
for _ in range(10000):  # Cost: $10K-$50K in API calls
    sample = generate_agentic_task()
    if sample:
        synthetic_dataset.append(sample)

# Cost: 10K samples x $2-5 per sample = $20K-$50K
# Benefit: Agentic capability unlocked
```

---

#### Method 2: Rule-Based + Template

**MiniMax-M1 Style (SynLogic):**

```python
"""
Synthetic Logical Reasoning Data (SynLogic style)
"""

def generate_logic_problem():
    # Templates for logical reasoning
    templates = [
        "if_then_chain",
        "contrapositive",
        "syllogism",
        "set_operations",
        "graph_traversal",
        "constraint_satisfaction"
    ]

    template = random.choice(templates)

    if template == "if_then_chain":
        # Example: Multi-step logical chain
        num_steps = random.randint(3, 7)

        # Generate premises
        premises = []
        for i in range(num_steps):
            premise = f"If {random_condition()} then {random_condition()}"
            premises.append(premise)

        # Generate question
        question = f"Given: {'; '.join(premises)}. What can we conclude?"

        # Generate step-by-step reasoning
        reasoning = generate_reasoning_chain(premises)

        # Generate final answer
        answer = derive_conclusion(premises)

        return {
            "question": question,
            "reasoning": reasoning,  # CoT
            "answer": answer,
            "template": template
        }

    # Similar for other templates...

# Generate dataset
logic_dataset = [generate_logic_problem() for _ in range(50000)]

# Cost: $0 (rule-based), 단 template design에 1-2주 필요
# Benefit: Logical reasoning capability
# Quality: Rule-based는 정확성 100% (하지만 diversity 제한)
```

---

#### Method 3: Transformation-Based

**Kimi K2 Style (Knowledge Rephrasing):**

```python
"""
Knowledge Rephrasing for Token Efficiency
"""

def rephrase_knowledge(original_text, style="learning_note"):
    """
    Original: Dry mathematical theorem
    → Learning-note: Step-by-step explanation with intuition
    """

    # Step 1: Style-diverse prompting
    styles = [
        "learning_note",    # As if teaching
        "socratic_dialogue",  # Q&A format
        "analogy_based",    # Using analogies
        "visual_description",  # Describing mentally
        "application_focused"  # Practical use cases
    ]

    style = random.choice(styles)

    rephrased = gpt4.generate(prompt=f"""
    Rephrase the following content in {style} style:

    Original:
    {original_text}

    Requirements:
    - Maintain factual accuracy (fidelity)
    - Make it more intuitive and educational
    - Add step-by-step breakdown if applicable
    - Keep length similar (token efficient)
    """)

    # Step 2: Fidelity verification
    #   Check if rephrased preserves original meaning
    fidelity_score = verify_fidelity(original_text, rephrased)

    if fidelity_score > 0.9:
        return rephrased
    else:
        return original_text  # Keep original if fidelity lost

# Apply to mathematical texts
math_corpus = load_corpus("mathematics")
rephrased_corpus = [rephrase_knowledge(text) for text in math_corpus]

# Result: Same information, better learning signal
# Cost: $5K-$20K (API calls for millions of texts)
# Benefit: Token efficiency, better reasoning capability
```

---

### 1.5 Data Quality Validation

**Validation Pipeline:**

```python
"""
Continuous Data Quality Validation
"""

# Metric 1: Perplexity Distribution
def check_perplexity(dataset, reference_model):
    """
    Too low perplexity = too easy (memorization)
    Too high perplexity = too noisy
    """
    perplexities = [reference_model.perplexity(text) for text in sample(dataset, 10000)]

    plt.hist(perplexities)

    # Target: Log-normal distribution with mean 50-500
    mean_ppl = np.mean(perplexities)

    if mean_ppl < 20:
        warnings.warn("Data too easy, model may underfit")
    elif mean_ppl > 1000:
        warnings.warn("Data too noisy, may hurt training")

    return perplexities

# Metric 2: Diversity (Vocabulary Coverage)
def check_diversity(dataset):
    """
    More unique tokens = more diverse
    """
    all_tokens = []
    for text in dataset:
        all_tokens.extend(tokenizer.tokenize(text))

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)

    diversity_ratio = unique_tokens / total_tokens

    # Target: 0.1-0.3 (10-30% unique)
    if diversity_ratio < 0.05:
        warnings.warn("Low diversity, repetitive data")

    return diversity_ratio

# Metric 3: Contamination Check
def check_contamination(dataset, eval_benchmarks):
    """
    Ensure eval data not in training data
    """
    for benchmark in eval_benchmarks:
        for eval_sample in benchmark:
            # Check n-gram overlap
            overlap = check_ngram_overlap(dataset, eval_sample, n=13)

            if overlap:
                warnings.warn(f"Contamination detected: {eval_sample[:100]}")
                # Remove contaminated samples
                dataset = remove_contaminated(dataset, eval_sample)

    return dataset

# Metric 4: Balance Check
def check_domain_balance(dataset, target_mixture):
    """
    Verify actual mixture matches target
    """
    actual_mixture = classify_domains(sample(dataset, 10000))

    for domain, target_pct in target_mixture.items():
        actual_pct = actual_mixture[domain]

        diff = abs(target_pct - actual_pct)

        if diff > 5:  # >5% difference
            warnings.warn(f"Domain {domain}: target {target_pct}%, actual {actual_pct}%")

# Run all validations
perplexities = check_perplexity(dataset, reference_model)
diversity = check_diversity(dataset)
dataset = check_contamination(dataset, eval_benchmarks)
check_domain_balance(dataset, target_mixture)

# Final approval
if all_metrics_pass():
    print("✅ Data quality validated, ready for training")
else:
    print("❌ Data quality issues detected, fix before training")
```

---

### 1.6 Data Pipeline Infrastructure

**Production-Ready Pipeline:**

```python
"""
Scalable Data Pipeline (Distributed)
"""

# Use Ray or Dask for distributed processing
import ray

ray.init()

@ray.remote
def process_shard(shard_path):
    """
    Process one shard (1GB) of raw data
    """
    # Load
    raw_data = load_jsonl(shard_path)

    # Clean
    cleaned = clean_pipeline(raw_data)

    # Filter
    filtered = filter_quality(cleaned)

    # Tokenize
    tokenized = tokenize(filtered)

    # Pack
    packed = pack_sequences(tokenized)

    return packed

# Parallel processing (100x speedup)
shard_paths = glob("raw_data/*.jsonl")  # 1000 shards

# Process all shards in parallel
futures = [process_shard.remote(path) for path in shard_paths]
processed_shards = ray.get(futures)

# Shuffle & merge
final_dataset = shuffle_and_merge(processed_shards)

# Time: 1000 shards in 1 hour (vs 1000 hours sequential)
# Cost: $100-$500 (cloud compute)
```

---

### 1.7 Data Foundation Checklist

```
[ ] Data Collection
    [ ] Sources identified (public, licensed, synthetic)
    [ ] Volume sufficient (100B+ tokens for 7B model)
    [ ] Domains diverse (general + specialized)
    [ ] Legal/license issues resolved

[ ] Data Cleaning
    [ ] Format normalized
    [ ] Language filtered
    [ ] Deduplicated (exact + fuzzy 85%)
    [ ] Quality filtered (perplexity, length, toxicity)
    [ ] PII removed
    [ ] Safety filtered

[ ] Data Mixture
    [ ] Target mixture defined
    [ ] Ablations performed (5-10 experiments)
    [ ] Winner validated at larger scale
    [ ] Domain balance verified

[ ] Synthetic Data (if applicable)
    [ ] Generation method chosen
    [ ] Quality validation pipeline
    [ ] Fidelity verified
    [ ] Cost-benefit analyzed

[ ] Data Quality
    [ ] Perplexity distribution normal
    [ ] Diversity sufficient (10-30%)
    [ ] Contamination checked
    [ ] Balance matches target

[ ] Infrastructure
    [ ] Pipeline scalable (distributed)
    [ ] Storage sufficient (100TB+)
    [ ] Throughput validated (>1M tokens/sec)
    [ ] Monitoring setup

[ ] Final Validation
    [ ] Small-scale pilot successful
    [ ] Stakeholder approval
    [ ] Ready for Phase 2
```

**Time:** 1-2 months (가장 긴 phase, 하지만 가장 중요)
**Cost:** $10K-$100K (data curation, ablations, infrastructure)
**Output:** High-quality, diverse, optimized training dataset

---

(계속 - Phase 2: Architecture Selection으로 이어짐)

## 🏗️ Phase 2: Architecture Selection

> "Architecture matters, but less than you think. Focus on proven designs."

### 2.1 Architecture Decision Tree

```
Start: We need a model architecture

Question 1: Dense or MoE?

├─ Dense (all parameters active)
│  Pros:
│  ✅ Simpler training
│  ✅ Easier debugging
│  ✅ No load balancing issues
│  ✅ Better for smaller models (<13B)
│
│  Cons:
│  ❌ Higher inference cost
│  ❌ Lower capacity per param
│
│  → Use when:
│     - Model size <13B
│     - Inference efficiency critical
│     - Team lacks MoE experience
│
└─ MoE (sparse activation)
   Pros:
   ✅ Higher capacity per param (10x)
   ✅ Efficient inference (only activated experts)
   ✅ Industry standard (GLM, MiniMax, Kimi all use MoE)

   Cons:
   ❌ Complex training (load balancing)
   ❌ Routing overhead
   ❌ Expert imbalance issues

   → Use when:
      - Model size >13B
      - Want SOTA capability
      - Have MoE expertise or willing to learn

   → MoE Configuration:
      - Experts: Start with 8-16, scale to 64-384 (Kimi K2 extreme)
      - Sparsity: Activate 1-2 experts per token (standard)
      - Routing: Top-k gating with load balancing loss

Question 2: Context Length?

├─ Standard (4K-8K)
│  → Use when:
│     - Most tasks <4K tokens
│     - Inference speed critical
│     - Memory constrained
│
├─ Medium (32K-128K)
│  → Use when:
│     - Need document understanding
│     - Code repo analysis
│     - Standard for 2024-2025 models
│  → Recommendation: 128K (industry standard)
│
└─ Long (256K-1M)
   → Use when:
      - Critical use case requires it
      - Willing to sacrifice speed/cost
      - Have specialized architecture (linear attention)
   → Warning: MiniMax-M1 style hybrid attention required

Question 3: Attention Mechanism?

├─ Standard Multi-Head Attention (MHA)
│  → Most common, well-tested
│  → Use for <128K context
│
├─ Grouped-Query Attention (GQA)
│  → Balance between MHA and MQA
│  → Good for 128K context
│  → Used by Kimi K2 (64 heads)
│
├─ Multi-Query Attention (MQA)
│  → Fast inference
│  → Use for edge deployment
│
└─ Hybrid Linear + Softmax (MiniMax-M1 style)
   → Only for 256K-1M context
   → Requires custom CUDA kernels (3-6 months dev)
   → ❌ NOT recommended for small teams unless critical need

Question 4: Novel Architecture?

├─ Standard (LLaMA-style)
│  ✅ Proven, well-tested
│  ✅ Community support
│  ✅ Fast iteration
│  → Recommended for 99% of cases
│
└─ Novel
   ❌ High risk
   ❌ +6-12 months development
   ❌ Many unknowns
   → Only if:
      - Unique requirement standard can't satisfy
      - Strong research motivation
      - Budget for extended timeline
```

---

### 2.2 Recommended Architecture (Small Team)

**For 7B Model (Most Practical):**

```python
ModelConfig(
    # Size
    num_parameters=7_000_000_000,
    hidden_size=4096,
    num_layers=32,

    # Attention
    num_attention_heads=32,  # Standard
    attention_type="GQA",  # Grouped-query for efficiency
    num_kv_heads=8,  # 4:1 ratio

    # MoE (optional, but recommended for 7B+)
    use_moe=True,
    num_experts=8,  # Start conservative
    num_experts_per_token=2,  # Standard
    expert_capacity_factor=1.25,  # Buffer for load balancing

    # Context
    max_position_embeddings=131_072,  # 128K (2^17)
    rope_theta=10000,  # RoPE base
    rope_scaling={
        "type": "yarn",  # YaRN for context extension
        "factor": 32  # 4K → 128K
    },

    # Architecture details
    intermediate_size=11_008,  # 2.7x hidden (SwiGLU)
    activation="swiglu",  # Standard for recent models
    norm_type="rmsnorm",  # Faster than LayerNorm
    norm_eps=1e-6,

    # Stability
    use_qk_norm=True,  # GLM-4.5, Kimi K2 style
    attention_dropout=0.0,  # Usually 0 for pre-training
    residual_dropout=0.0,

    # Efficiency
    use_flash_attention=True,  # 2-3x faster
    gradient_checkpointing=True,  # Save memory

    # Tokenizer
    vocab_size=50_000,  # Adjust based on languages

    # Multi-token prediction (GLM-4.5 style, optional)
    use_multi_token_prediction=False,  # Advanced feature
    num_prediction_heads=1,  # Set to 3-5 if enabled
)
```

**Scaling Rules:**

| Model Size | Layers | Hidden | Heads | Intermediate | KV Heads (GQA) | Experts (MoE) |
|-----------|--------|--------|-------|--------------|----------------|---------------|
| **1B** | 22 | 2048 | 16 | 5504 | 4 | 8 |
| **3B** | 26 | 2560 | 20 | 6912 | 5 | 8 |
| **7B** | 32 | 4096 | 32 | 11008 | 8 | 8-16 |
| **13B** | 40 | 5120 | 40 | 13824 | 10 | 16-32 |
| **70B** | 80 | 8192 | 64 | 22016 | 16 | 32-64 |

---

### 2.3 Architecture Validation (Before Full Training)

**Pilot Test:**

```python
"""
Architecture Validation with Small-Scale Training
"""

def validate_architecture(config, pilot_config):
    """
    Train tiny model to validate architecture choices
    """

    # Step 1: Create small version
    small_config = scale_down_config(config, target_size="100M")

    # Step 2: Train on small data (1B tokens)
    model = train_pilot(
        config=small_config,
        data=pilot_data,  # 1B tokens
        steps=1000,
        gpus=8,
        duration="12 hours"
    )

    # Step 3: Check for issues
    checks = {
        "training_stability": check_loss_curve(model.history),
        "attention_patterns": analyze_attention(model),
        "expert_balance": check_expert_usage(model) if config.use_moe else True,
        "gradient_norms": check_gradient_health(model),
        "memory_efficiency": check_memory_usage(model),
        "throughput": check_training_speed(model)
    }

    # Step 4: Decision
    issues = [k for k, v in checks.items() if not v]

    if not issues:
        return "✅ Architecture validated, proceed to full training"
    elif len(issues) <= 2:
        return f"⚠️ Minor issues: {issues}. Fix and re-validate."
    else:
        return f"❌ Major issues: {issues}. Redesign architecture."

# Cost: $500-$2K (12 hours, 8 GPUs)
# Time: 1 day
# Value: Catch architectural issues before $100K+ training
```

---

### 2.4 Common Architecture Pitfalls (Avoid)

**Pitfall 1: Too Many Novel Components**
```
❌ Bad: Custom attention + novel activation + new normalization + ...
✅ Good: Standard architecture + 1 novel component (if必要)

Reason: Each novelty adds 2-3 months debug time
```

**Pitfall 2: Undersized Intermediate Layer**
```
❌ Bad: intermediate_size = 2x hidden_size
✅ Good: intermediate_size = 2.7x hidden_size (SwiGLU standard)

Reason: Capacity bottleneck, performance ceiling
```

**Pitfall 3: Too Many Experts Too Soon**
```
❌ Bad: Start with 384 experts (Kimi K2 scale)
✅ Good: Start with 8-16, scale up after validation

Reason: Load balancing nightmare, weeks of debugging
```

**Pitfall 4: Ignoring Inference Cost**
```
❌ Bad: Design for training only
✅ Good: Consider inference from day 1

Example: 128 attention heads → 2x slower inference than 64 heads
         GQA → 4-5x faster than MHA at inference
```

---

### 2.5 Architecture Checklist

```
[ ] Configuration Defined
    [ ] Model size chosen (1B/7B/13B/70B)
    [ ] Dense vs MoE decided
    [ ] Attention mechanism selected
    [ ] Context length decided (128K recommended)
    [ ] All hyperparameters specified

[ ] Validation Complete
    [ ] Small-scale pilot successful (100M model, 1B tokens)
    [ ] No training instability
    [ ] Memory usage acceptable
    [ ] Throughput meets target (>50% of theoretical)
    [ ] Expert balance good (if MoE)

[ ] Implementation Ready
    [ ] Code reviewed
    [ ] Unit tests pass
    [ ] Distributed training tested (multi-GPU)
    [ ] Checkpoint saving/loading works
    [ ] Mixed precision (BF16) enabled

[ ] Trade-offs Understood
    [ ] Training cost estimated
    [ ] Inference cost estimated
    [ ] Capability vs efficiency trade-off accepted
    [ ] Team has expertise or learning plan

[ ] Stakeholder Approval
    [ ] Architecture documented
    [ ] Justification clear
    [ ] Alternatives considered
    [ ] Go-ahead confirmed
```

**Output:** Validated, production-ready architecture ready for Phase 3 (Pre-training)

---

## 🎓 Phase 4: Post-Training (가장 실용적!)

> "Post-training is where small teams can actually compete"
> — 월드클래스 모델들은 모두 sophisticated post-training 사용

### 4.1 Post-Training Philosophy

**왜 Post-training이 작은 팀에게 유리한가?**

```
Pre-training:
- Cost: $100K-$10M
- Time: 3-12개월
- Risk: 높음
- Requires: 대규모 인프라

Post-training:
- Cost: $1K-$100K (100-1000x 저렴)
- Time: 1-8주
- Risk: 낮음
- Requires: 8-64 GPUs만으로 가능
```

**논문들의 증거:**
- GLM-4.5: Multi-stage post-training으로 91% AIME
- Kimi K2: 6+ RL stages로 multiple SOTA
- MiniMax-M1: RL로 DeepSeek R1 근접

**Key Insight:** Base model은 구매/다운로드, post-training으로 차별화!

---

### 4.2 Multi-Stage Post-Training Strategy

**GLM-4.5이 증명한 구조:**

```
Stage 1: SFT (Supervised Fine-Tuning)
├─ Input: Base model + high-quality examples
├─ Output: Instruction-following model
├─ Duration: 1-2주
└─ Cost: $5K-$20K

Stage 2: Reasoning RL (Easy → Hard Curriculum)
├─ Input: SFT model + math/logic tasks
├─ Method: PPO/GRPO with difficulty curriculum
├─ Duration: 2-3주
├─ Cost: $10K-$50K
└─ Output: Reasoning-capable model

Stage 3: Agentic RL (Tool-use)
├─ Input: Reasoning model + tool tasks
├─ Method: Multi-turn RL with execution feedback
├─ Duration: 2-3주
├─ Cost: $10K-$50K
└─ Output: Agentic model

Stage 4: Iterative Refinement (Optional)
├─ Data replacement (GLM-4.5 style)
├─ Self-critique (Kimi K2 style)
├─ Duration: 1-2주
└─ Cost: $5K-$20K

Total: 6-10주, $30K-$140K
```

---

### 4.3 Stage 1: Supervised Fine-Tuning (SFT)

**목적:** Base model → Instruction-following model

#### Data Preparation

**필요한 데이터:**
```
Volume: 10K-100K examples (not millions!)
Quality: High (human-written or GPT-4 generated)

Format:
{
    "instruction": "User query",
    "input": "Optional context",
    "output": "Expected response"
}

Domains:
- General QA: 40%
- Reasoning: 20%
- Coding: 20%
- Tool-use: 10%
- Safety: 10%
```

**Data Sources:**

**Option A: Public Datasets**
```python
from datasets import load_dataset

# High-quality instruction datasets
datasets = [
    "Open-Orca/OpenOrca",  # 1M GPT-4 examples
    "WizardLM/WizardLM_evol_instruct_V2",  # 143K
    "timdettmers/openassistant-guanaco",  # 10K high-quality
    "HuggingFaceH4/no_robots",  # 10K human-written
]

# Combine and deduplicate
combined = combine_datasets(datasets)
filtered = filter_quality(combined, min_score=4.0)
deduplicated = deduplicate(filtered, threshold=0.85)

final_sft_data = sample(deduplicated, n=50_000)  # 50K examples
```

**Option B: Synthetic Generation**
```python
"""
Generate domain-specific SFT data with GPT-4
"""

def generate_sft_example(domain):
    prompt = f"""
    Generate a high-quality instruction-following example for {domain}.

    Format:
    Instruction: <user query>
    Response: <detailed, helpful answer>

    Requirements:
    - Realistic user need
    - Clear, correct response
    - Appropriate difficulty
    - {domain}-specific knowledge
    """

    example = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9  # Diversity
    )

    return parse_example(example)

# Generate 10K examples x $0.01 = $100
sft_data = [generate_sft_example(domain) for domain in domains for _ in range(2000)]

# Cost: $100-$500
# Time: 1-2 days
```

---

#### Training Configuration

**LoRA/QLoRA (Recommended for Small Teams):**

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank (8-64, higher = more capacity)
    lora_alpha=32,  # Scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap base model
base_model = load_model("meta-llama/Llama-2-7b-hf")
peft_model = get_peft_model(base_model, lora_config)

# Training args
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    num_train_epochs=3,
    learning_rate=2e-4,  # Higher than full fine-tuning
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    bf16=True,  # Mixed precision
    gradient_checkpointing=True,  # Save memory
)

# Train
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=sft_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=2048,
)

trainer.train()

# Cost: 8 A100 x 24 hours = $4K-$8K
# Trainable params: 0.1% of total (67M out of 7B)
# Memory: ~30GB per GPU (vs 80GB+ for full fine-tuning)
```

**Full Fine-Tuning (If Budget Allows):**

```python
# Same setup but no LoRA
model = load_model("meta-llama/Llama-2-7b-hf")

# Lower learning rate for full fine-tuning
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Smaller due to memory
    gradient_accumulation_steps=8,
    num_train_epochs=2,  # Fewer epochs
    learning_rate=1e-5,  # Much lower
    # ... rest similar
)

# Cost: 32-64 GPUs x 1 week = $20K-$40K
# Better performance (+2-5% over LoRA)
# Worth it if budget available
```

---

#### SFT Validation

```python
"""
Validate SFT model before RL
"""

def validate_sft(model):
    checks = {}

    # 1. Instruction following
    checks["instruction_following"] = evaluate_instruction_following(
        model, benchmark="IFEval"
    )
    # Target: >60% strict accuracy

    # 2. General capability maintained
    checks["mmlu"] = evaluate(model, benchmark="MMLU")
    # Target: Within 2% of base model

    # 3. Safety
    checks["safety"] = evaluate_safety(model, benchmark="ToxiGen")
    # Target: Toxicity <10%

    # 4. No catastrophic forgetting
    checks["coding"] = evaluate(model, benchmark="HumanEval")
    checks["math"] = evaluate(model, benchmark="GSM8K")
    # Target: Within 5% of base model

    # Decision
    if all(v > threshold for v in checks.values()):
        return "✅ SFT successful, proceed to RL"
    else:
        return f"❌ Issues detected: {checks}"

# Critical: Don't skip validation!
# 1-2 days extra now saves weeks later
```

---

### 4.4 Stage 2: Reinforcement Learning

**핵심:** RL은 post-training의 secret sauce

#### 4.4.1 RL Algorithm Choice

**Decision Tree:**

```
Q: What type of capability?

└─ Verifiable Tasks (Math, Code)
   │
   ├─ PPO (Standard)
   │  ✅ Well-tested, stable
   │  ✅ Libraries available (trl, openrlhf)
   │  Cost: Baseline
   │  → Use when: Safe choice, first time doing RL
   │
   ├─ GRPO (Group-relative)
   │  ✅ Simpler than PPO (no critic)
   │  ✅ Faster training
   │  Cost: 30% less than PPO
   │  → Use when: Want efficiency
   │
   └─ CISPO (Importance sampling, MiniMax-M1)
      ⚠️ Novel, requires implementation
      ✅ Better for reasoning tasks
      ✅ Preserves rare tokens
      Cost: Baseline + dev time (2-4 weeks)
      → Use when: Reasoning-heavy, willing to invest

└─ Subjective Tasks (Creative writing, dialogue)
   │
   ├─ DPO (Direct Preference Optimization)
   │  ✅ No reward model needed
   │  ✅ Simpler pipeline
   │  Cost: 50% of PPO
   │  Data: Pairwise preferences (10K-100K)
   │  → Use when: Have preference data
   │
   └─ RLHF (Classical)
      ⚠️ Complex (reward model + PPO)
      Cost: 2x DPO
      → Use when: Highest quality needed
```

**Recommendation for Small Teams:** Start with **GRPO** or **DPO**
- GRPO: Verifiable tasks (math, code)
- DPO: Preference tasks (helpfulness, style)

---

#### 4.4.2 Reward Signal Design (Critical!)

**Kimi K2's Verifiable Rewards Gym:**

```python
"""
Multi-domain Reward Framework
"""

class VerifiableRewardsGym:
    def __init__(self):
        self.domains = {
            "math": MathVerifier(),
            "code": CodeVerifier(),
            "logic": LogicVerifier(),
            "safety": SafetyVerifier(),
        }

    def compute_reward(self, prompt, response):
        # Domain detection
        domain = detect_domain(prompt)
        verifier = self.domains[domain]

        # Compute reward
        reward = verifier.verify(prompt, response)

        return reward

class MathVerifier:
    """Rule-based math verification"""

    def verify(self, problem, solution):
        # Extract final answer
        predicted_answer = extract_answer(solution)

        # Ground truth
        ground_truth = execute_solution(problem)

        # Binary reward
        if predicted_answer == ground_truth:
            return 1.0  # Correct
        else:
            return -0.5  # Wrong (negative to discourage)

class CodeVerifier:
    """Sandbox execution for code"""

    def verify(self, problem, code):
        # Extract test cases
        test_cases = extract_tests(problem)

        # Run in sandbox
        results = []
        for test_input, expected_output in test_cases:
            try:
                actual_output = sandbox_execute(code, test_input, timeout=5)
                results.append(actual_output == expected_output)
            except Exception as e:
                results.append(False)  # Runtime error

        # Reward = pass rate
        pass_rate = sum(results) / len(results)

        # Bonus for clean code
        if pass_rate == 1.0:
            bonus = evaluate_code_quality(code)  # 0-0.2
            return pass_rate + bonus
        else:
            return pass_rate * 0.5  # Partial credit

class SafetyVerifier:
    """Adversarial safety checking"""

    def verify(self, prompt, response):
        # Multiple safety checks
        toxicity = check_toxicity(response)
        jailbreak = check_jailbreak(prompt, response)
        bias = check_bias(response)

        # Combine (all must pass)
        if toxicity < 0.5 and not jailbreak and bias < 0.3:
            return 1.0  # Safe
        else:
            return -1.0  # Unsafe (strong penalty)
```

---

#### 4.4.3 Multi-task RL (Catastrophic Forgetting Prevention)

**Critical Implementation:**

```python
"""
Multi-task RL Training Loop (Kimi K2 style)
"""

def multi_task_rl_training():
    # Define task mixture
    tasks = {
        "math": {
            "weight": 0.30,
            "data": load_math_data(),
            "verifier": MathVerifier()
        },
        "code": {
            "weight": 0.30,
            "data": load_code_data(),
            "verifier": CodeVerifier()
        },
        "reasoning": {
            "weight": 0.20,
            "data": load_reasoning_data(),
            "verifier": LogicVerifier()
        },
        "general": {
            "weight": 0.20,
            "data": load_general_data(),
            "verifier": GenericVerifier()
        }
    }

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            # Sample task based on weights
            task_name = sample_task(tasks)
            task = tasks[task_name]

            # Generate rollouts
            prompts = sample(task["data"], batch_size=32)
            responses = model.generate(prompts)

            # Compute rewards
            rewards = [
                task["verifier"].verify(p, r)
                for p, r in zip(prompts, responses)
            ]

            # RL update (GRPO example)
            loss = compute_grpo_loss(
                model, prompts, responses, rewards
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log per-task performance
            log_metrics(task_name, rewards)

    # Key: All tasks trained throughout
    # → No forgetting!
```

---

#### 4.4.4 PTX Loss (Kimi K2 Style)

**Implementation:**

```python
"""
PTX Loss for Knowledge Preservation
"""

def train_with_ptx(model, rl_data, ptx_data, gamma=0.1):
    """
    Combined RL + PTX training
    """

    for step in range(training_steps):
        # RL batch
        rl_prompts, rl_responses, rl_rewards = sample_rl_rollout(rl_data)

        # RL loss (GRPO)
        rl_loss = compute_grpo_loss(model, rl_prompts, rl_responses, rl_rewards)

        # PTX batch (curated high-quality data)
        ptx_batch = sample(ptx_data, batch_size=32)

        # PTX loss (standard language modeling)
        ptx_loss = model.compute_lm_loss(ptx_batch)

        # Combined loss
        total_loss = rl_loss + gamma * ptx_loss

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Monitor both
        log({
            "rl_loss": rl_loss.item(),
            "ptx_loss": ptx_loss.item(),
            "total_loss": total_loss.item()
        })

# PTX data curation
ptx_dataset = curate_ptx_data(
    sources=[
        ("general_qa", 10000),      # General knowledge
        ("reasoning", 5000),        # Basic reasoning
        ("common_sense", 5000),     # Common sense
        ("facts", 5000),            # Factual knowledge
    ]
)

# Hyperparameter tuning
for gamma in [0.05, 0.1, 0.2]:
    model = train_with_ptx(model, rl_data, ptx_data, gamma=gamma)
    eval_results = evaluate(model)
    # Select best gamma
```

---

#### 4.4.5 Curriculum Learning (GLM-4.5 Style)

**Difficulty-based Progression:**

```python
"""
Curriculum Learning for RL
"""

def curriculum_rl_training(model, task_data):
    # Stage 1: Easy problems (warmup)
    easy_data = filter_by_difficulty(task_data, difficulty="easy")

    model = rl_train(
        model, easy_data,
        epochs=2,
        note="Warmup on easy problems"
    )

    # Stage 2: Medium problems
    medium_data = filter_by_difficulty(task_data, difficulty="medium")

    # Gradually mix in easy (prevent forgetting)
    mixed_data = combine([easy_data, medium_data], weights=[0.3, 0.7])

    model = rl_train(
        model, mixed_data,
        epochs=3,
        note="Ramp up difficulty"
    )

    # Stage 3: Hard problems
    hard_data = filter_by_difficulty(task_data, difficulty="hard")

    # Keep practicing easier ones
    final_mix = combine(
        [easy_data, medium_data, hard_data],
        weights=[0.2, 0.3, 0.5]
    )

    model = rl_train(
        model, final_mix,
        epochs=5,
        note="Master hard problems while maintaining easy/medium"
    )

    return model

# Difficulty scoring
def score_difficulty(problem):
    # Method 1: Pass rate from base model
    pass_rate = base_model.solve_rate(problem)

    if pass_rate > 0.8:
        return "easy"
    elif pass_rate > 0.3:
        return "medium"
    else:
        return "hard"

# Cost: Minimal overhead (same total training, just ordered differently)
# Benefit: 5-10% better final performance, more stable training
```

---

### 4.5 Post-Training Checklist

```
[ ] Stage 1: SFT
    [ ] Data curated (10K-100K examples)
    [ ] LoRA/QLoRA configured
    [ ] Training completed (1-2주)
    [ ] Validation passed (IFEval, MMLU, safety)

[ ] Stage 2: RL Setup
    [ ] Algorithm chosen (GRPO/DPO recommended)
    [ ] Reward functions implemented
    [ ] Multi-task framework ready
    [ ] PTX data curated (if using)

[ ] Stage 3: RL Training
    [ ] Curriculum defined (easy → hard)
    [ ] Multi-task weights set
    [ ] Training monitored (reward, KL divergence)
    [ ] Checkpoints validated

[ ] Stage 4: Catastrophic Forgetting Prevention
    [ ] Multi-task training active
    [ ] PTX loss (if used) with gamma tuned
    [ ] Baseline benchmarks tracked
    [ ] No regression detected

[ ] Final Validation
    [ ] Target capability achieved
    [ ] General capability maintained
    [ ] Safety checks passed
    [ ] Ready for deployment

[ ] Cost Check
    [ ] Total spend within budget
    [ ] ROI positive
    [ ] Lessons documented
```

**Total Duration:** 6-10주
**Total Cost:** $30K-$140K (vs $100K-$10M for pre-training)

---

## 📈 Phase 5: Evaluation & Iteration

### 5.1 Comprehensive Evaluation Framework

**Multi-dimensional Evaluation:**

```python
"""
Evaluation Suite for Model Validation
"""

class ModelEvaluator:
    def __init__(self):
        self.benchmarks = {
            # Reasoning
            "AIME": AIIMEBenchmark(),
            "GPQA": GPQABenchmark(),
            "MMLU": MMLUBenchmark(),

            # Coding
            "HumanEval": HumanEvalBenchmark(),
            "MBPP": MBPPBenchmark(),
            "LiveCodeBench": LiveCodeBenchmark(),

            # Agentic
            "TAU-Bench": TAUBench(),
            "SWE-bench": SWEBench(),

            # General
            "IFEval": IFEvalBenchmark(),
            "MMLU-Pro": MMLUProBenchmark(),

            # Safety
            "ToxiGen": ToxiGenBenchmark(),
            "TruthfulQA": TruthfulQABenchmark(),
        }

    def full_eval(self, model):
        results = {}

        for name, benchmark in self.benchmarks.items():
            print(f"Evaluating {name}...")
            score = benchmark.evaluate(model)
            results[name] = score

        # Generate report
        report = self.generate_report(results)

        return results, report

    def regression_check(self, new_model, baseline_model):
        """
        Critical: Check for performance regression
        """
        new_results = self.full_eval(new_model)
        baseline_results = self.full_eval(baseline_model)

        regressions = []

        for benchmark, new_score in new_results.items():
            baseline_score = baseline_results[benchmark]
            delta = new_score - baseline_score

            if delta < -0.05:  # >5% regression
                regressions.append({
                    "benchmark": benchmark,
                    "baseline": baseline_score,
                    "new": new_score,
                    "delta": delta
                })

        if regressions:
            warnings.warn(f"Regressions detected: {regressions}")

        return regressions
```

---

### 5.2 Iteration Framework

**PDCA (Plan-Do-Check-Act) Cycle:**

```
Iteration N:

1. PLAN (목표 설정)
   - Current: Model v1.0, MMLU 65%
   - Target: MMLU 70% (+5%)
   - Method: Add 50K reasoning examples to SFT
   - Budget: $10K
   - Timeline: 2주

2. DO (실행)
   - Curate data
   - Run SFT
   - Monitor training

3. CHECK (검증)
   - Evaluate v1.1
   - MMLU: 68% (+3%, short of +5%)
   - HumanEval: 55% → 52% (-3%, regression!)

4. ACT (조정)
   - Analysis: Reasoning improved, coding regressed
   - Root cause: Imbalanced training data
   - Decision: Add 30K coding examples, repeat
   - OR: Accept trade-off if coding not critical

Iteration N+1:
- Adjust based on learnings
- Repeat cycle
```

---

### 5.3 When to Stop Iterating

**Stopping Criteria:**

```
✅ Stop iterating when:
1. Target metrics achieved
2. ROI becomes negative (diminishing returns)
3. Deadline approaching
4. No improvement for 3 consecutive iterations
5. Budget exhausted

Continue if:
1. Clear path to improvement
2. ROI still positive
3. Critical capability gap remains
```

---

## 🚨 Phase 6: Common Failures & Recovery

> "The 2am debugging sessions no one talks about"

### 6.1 Training Failures (from SmolLM3 Playbook)

#### Mystery 1: Vanishing Throughput

**Symptoms:**
```
Expected: 400K tokens/sec
Reality: 400K → 300K → 200K → 100K
Timeline: Degrades over hours
```

**Root Causes:**
1. Tensor parallelism communication overhead
2. Gradient accumulation interactions
3. Memory leaks
4. NCCL issues

**Debug Process:**
```python
# Step 1: Profile
profiler.profile(training_step)
# Look for: Increasing all-reduce time

# Step 2: Check GPU utilization
nvidia-smi dmon
# Look for: Decreasing GPU utilization

# Step 3: Monitor network
iftop
# Look for: Network saturation

# Solution (SmolLM3):
# Optimize tensor parallelism configuration
# Reduce gradient accumulation steps
# Fix NCCL network topology
```

**Recovery:**
- Checkpoint current state
- Rollback to stable configuration
- Apply fix
- Resume training

**Cost:** 6-12 hours debugging, minimal compute waste (caught early)

---

#### Mystery 2: Loss Spikes

**Symptoms:**
```
Loss: 2.5 → 2.3 → 2.1 → 5.8 (spike!) → 3.0 → 2.9
```

**Root Causes:**
1. Bad batch (outlier data)
2. Gradient explosion
3. Numerical instability
4. Optimizer state corruption
5. Hardware issues

**Mitigation:**
```python
# Prevention
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Gradient clipping
    warmup_steps=500,   # Gradual LR warmup
    logging_steps=10,   # Frequent monitoring
    save_steps=100,     # Frequent checkpoints
)

# Detection
def detect_loss_spike(loss_history, window=10, threshold=2.0):
    recent_mean = np.mean(loss_history[-window:])
    current_loss = loss_history[-1]

    if current_loss > recent_mean * threshold:
        return True  # Spike detected!
    return False

# Recovery
if detect_loss_spike(trainer.loss_history):
    # Option 1: Skip bad batch
    continue

    # Option 2: Rollback checkpoint
    model.load_checkpoint(last_stable_checkpoint)

    # Option 3: Reduce learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
```

**Cost:** Varies
- Minor spike: 0 cost (skip batch)
- Major spike: Rollback 50-100B tokens ($5K-$10K loss)

---

#### Mystery 3: Checkpoint Corruption

**Symptoms:**
```
Loading checkpoint... ERROR
Model outputs gibberish after loading
NaN losses immediately after resume
```

**Prevention:**
```python
# Robust checkpoint saving
def save_checkpoint_robust(model, path, step):
    # Save to temp location first
    temp_path = f"{path}.tmp"
    torch.save(model.state_dict(), temp_path)

    # Verify integrity
    loaded = torch.load(temp_path)
    assert loaded is not None

    # Test forward pass
    test_output = model(test_input)
    assert not torch.isnan(test_output).any()

    # Only then move to final location
    os.rename(temp_path, path)

    # Keep multiple checkpoints
    if step % 1000 == 0:
        backup_path = f"{path}.step{step}"
        shutil.copy(path, backup_path)
```

**Recovery:**
- Always keep last 3-5 checkpoints
- Test checkpoint validity before discarding old ones
- Rollback to last good checkpoint

**Cost:** Lose progress since last good checkpoint (hours to days)

---

### 6.2 Post-Training Failures

#### Failure: Reward Hacking

**Symptoms:**
```
Reward: 0.5 → 0.6 → 0.7 → 0.9 → 0.95
But: Actual quality decreases!

Example:
Q: "Solve 2+2"
A: "The answer is 4. Let me verify: 2+2 = 4. Indeed, 4 is correct.
   Therefore, the answer is 4. In conclusion, 4. Final answer: 4."

Reward: High (contains correct answer)
Quality: Low (repetitive, verbose)
```

**Prevention:**
```python
# Multi-faceted reward
def compute_reward(problem, solution):
    # Correctness
    correctness = verify_answer(problem, solution)

    # Conciseness penalty
    length_penalty = -0.01 * len(solution.split())

    # Repetition penalty
    repetition_penalty = -count_repetitions(solution) * 0.1

    # Combined
    total_reward = correctness + length_penalty + repetition_penalty

    return total_reward
```

---

#### Failure: Catastrophic Forgetting (Despite Mitigation)

**Symptoms:**
```
Before RL:
- Math: 70%
- Coding: 65%
- General: 75%

After Math RL:
- Math: 85% (+15% ✅)
- Coding: 45% (-20% ❌)
- General: 70% (-5% ❌)
```

**Recovery:**
```python
# Diagnosis
regressions = detect_regressions(new_model, baseline_model)

if regressions:
    # Option 1: Increase multi-task mixing
    # Increase non-math tasks from 30% → 50%

    # Option 2: Increase PTX loss weight
    # gamma: 0.1 → 0.2

    # Option 3: Lower learning rate
    # lr: 1e-5 → 5e-6

    # Option 4: Add experience replay
    # Keep 20% samples from pre-RL training

    # Re-train with adjustments
```

**Cost:** 1-2 additional RL iterations ($10K-$40K)

---

### 6.3 Infrastructure Failures

#### GPU Failures

**Reality:** At scale, GPUs fail regularly
- 32 GPUs, 1 month training → Expect 1-2 GPU failures
- 256 GPUs, 1 week → Expect 1-3 failures

**Mitigation:**
```bash
# Health checks
nvidia-smi --query-gpu=health --format=csv

# Automatic restart with fault tolerance
# (PyTorch FSDP, DeepSpeed handle this)

# Checkpoint frequently (every 100-500 steps)
```

**Recovery:**
- Checkpoint saves: Lose at most last 100 steps
- Dynamic GPU exclusion: Continue with N-1 GPUs

**Cost:** Minimal if handled automatically

---

### 6.4 Emergency Decision Tree

```
Training Problem Detected:

Q: Is training completely stopped?
├─ Yes:
│  ├─ GPU failure? → Replace GPU, resume from checkpoint
│  ├─ OOM error? → Reduce batch size, enable gradient checkpointing
│  └─ Code bug? → Fix, resume
│
└─ No (but issues):
   ├─ Loss spike? → See Mystery 2 above
   ├─ Throughput degradation? → See Mystery 1 above
   └─ Checkpoint corruption? → See Mystery 3 above

Q: How much progress lost?
├─ <1 day: Resume from checkpoint
├─ 1-7 days: Analyze root cause, fix, resume
├─ >7 days: CRITICAL
   └─ Emergency meeting
       ├─ Root cause clear? → Fix and resume
       └─ Not clear? → Pause, investigate, consider restart

Q: Budget remaining?
├─ >30%: Can afford fixes/restarts
└─ <30%: CRITICAL
    → Prioritize finishing over perfection
    → Consider scope reduction
```

---

## 💰 Phase 7: Resource Management

### 7.1 Budget Allocation Template

**Small Model (7B):**
```
Total Budget: $100K

Breakdown:
1. Planning & Validation (5%): $5K
   - Architecture pilots
   - Data quality validation

2. Data (20%): $20K
   - Curation: $10K
   - Synthetic generation: $5K
   - Storage: $5K

3. Training (50%): $50K
   - Main run (100B-1T tokens): $20K
   - Ablations (5-10 experiments): $15K
   - Debugging/restarts (buffer): $15K

4. Post-Training (15%): $15K
   - SFT: $5K
   - RL: $10K

5. Evaluation & Iteration (5%): $5K

6. Contingency (5%): $5K

Reality: Likely spend $120K-$150K (1.2-1.5x)
```

---

### 7.2 Timeline Template

**7B Model End-to-End:**

```
Month 1: Planning & Data
Week 1-2: Validation & architecture pilots
Week 3-4: Data curation & mixture ablations

Month 2-3: Pre-training (if needed)
Week 5-8: Main training run (1T tokens)
Week 9-12: Buffer for issues

Month 4: Post-training
Week 13-14: SFT
Week 15-16: RL (multi-stage)

Month 5: Evaluation & Iteration
Week 17-18: Full evaluation
Week 19-20: Iteration & final tuning

Total: 5 months (Plan: 3 months + 67% buffer)

Reality: Likely 6-7 months for first model
         3-4 months for subsequent models (learning curve)
```

---

## ✅ Phase 8: Practical Checklists

### Pre-Training Checklist

```
[ ] Phase 0: Validation
    [ ] Feasibility confirmed
    [ ] ROI >= 3x
    [ ] Budget secured (realistic x2.5)
    [ ] Team ready

[ ] Phase 1: Data
    [ ] 100B+ tokens collected
    [ ] Cleaned & deduplicated
    [ ] Mixture optimized (5-10 ablations)
    [ ] Quality validated

[ ] Phase 2: Architecture
    [ ] Configuration defined
    [ ] Pilot successful (100M model)
    [ ] No architectural red flags

[ ] Phase 3: Training Setup
    [ ] Infrastructure tested (multi-GPU)
    [ ] Checkpointing works
    [ ] Monitoring setup
    [ ] Alert system configured

[ ] Phase 4: Execution
    [ ] Training launched
    [ ] Continuous monitoring
    [ ] Checkpoints verified
    [ ] Throughput acceptable

[ ] Phase 5: Completion
    [ ] Target tokens trained
    [ ] Final checkpoint validated
    [ ] Preliminary evaluation passed
```

### Post-Training Checklist

```
[ ] Stage 1: SFT
    [ ] 10K-100K examples curated
    [ ] LoRA configured (if applicable)
    [ ] Training complete (1-2주)
    [ ] Validation passed

[ ] Stage 2: RL
    [ ] Algorithm chosen
    [ ] Rewards implemented
    [ ] Multi-task setup
    [ ] Curriculum defined

[ ] Stage 3: Training
    [ ] RL training complete
    [ ] No catastrophic forgetting
    [ ] Target capability achieved

[ ] Stage 4: Validation
    [ ] Full benchmark suite
    [ ] Regression checks passed
    [ ] Safety validated
    [ ] Ready for deployment
```

---

## 📚 Case Studies & Examples

### Case Study 1: Medical Domain Specialization (Success)

**Scenario:** 작은 병원 AI 스타트업

**Approach:**
- Base model: Llama-2-7B
- Method: LoRA fine-tuning
- Data: 50K medical QA pairs (curated with doctors)
- Time: 6주
- Cost: $35K

**Results:**
- Medical QA accuracy: 45% → 72% (+27%)
- General capability: Maintained (MMLU 65% → 64%)
- ROI: $35K investment → $500K annual revenue (14x)

**Key Success Factors:**
1. Clear niche (Korean hospital use case)
2. High-quality domain data (doctor-curated)
3. LoRA (cost-effective)
4. Multi-task (maintained general capability)

---

### Case Study 2: General QA Model (Failure)

**Scenario:** 스타트업, GPT-4 경쟁 시도

**Approach:**
- Train 70B model from scratch
- Budget: $2M
- Timeline: 12 months

**Results:**
- MMLU: 75% (GPT-3.5 수준, GPT-4 85%)
- Cost overrun: $2M → $3.5M
- Market: 이미 GPT-4, Claude 지배
- Outcome: 프로젝트 중단, 회사 어려움

**Failure Analysis:**
1. ❌ Tried to compete in commodity market
2. ❌ Underestimated competition
3. ❌ No clear differentiation
4. ❌ Insufficient ROI analysis

**Lesson:** 작은 팀은 niche specialization, not general competition

---

### Case Study 3: Iterative Improvement (Success)

**Scenario:** Code-focused model

**Iteration 1:**
- Approach: SFT on coding data
- Result: HumanEval 35% → 55% (+20% ✅)
- Cost: $10K

**Iteration 2:**
- Insight: Need reasoning for complex problems
- Approach: Add math RL with curriculum
- Result: HumanEval 55% → 62% (+7% ✅)
- Cost: $20K

**Iteration 3:**
- Insight: Tool-use helps (running code)
- Approach: Agentic RL with Python sandbox
- Result: HumanEval 62% → 68% (+6% ✅)
- Cost: $25K

**Total:**
- Cost: $55K
- Result: 35% → 68% (+33%, near SOTA)
- Timeline: 4 months
- ROI: Positive (niche market)

**Key Success Factors:**
1. Incremental approach
2. Data-driven decisions
3. Each iteration validated
4. Stopped at "good enough"

---

## 🎓 Final Wisdom

### Top 10 Lessons from World-Class Models

1. **Data > Architecture** (always)
2. **Validate before investing** (pilots are cheap)
3. **Reserve 50% buffer** (not 20-30%)
4. **Iteration speed > Team size** (small team advantage)
5. **Ablations are not optional** (50-100% of compute)
6. **Multi-task prevents forgetting** (not self-distillation)
7. **Post-training is where small teams compete** (100-1000x cheaper than pre-training)
8. **ROI >= 3x or don't do it** (higher bar for small teams)
9. **Niche > General** (specialization is defensible)
10. **Document failures** (learning compounds)

---

### When to Build vs Buy

**Build when:**
- ✅ Clear niche with economic moat
- ✅ Unique data access
- ✅ ROI >= 3x
- ✅ Defensible differentiation

**Buy/Use API when:**
- ❌ General use case (GPT-4/Claude already best)
- ❌ Insufficient data
- ❌ ROI unclear
- ❌ Faster time-to-market critical

**Post-train existing model when:**
- ✅ 80% of build, 5% of cost
- ✅ Domain specialization
- ✅ Language expansion
- ✅ Specific capability gap

---

## 🚀 Getting Started Tomorrow

**Week 1 Action Plan:**

```
Day 1-2: Decision
[ ] Review this guideline
[ ] Identify use case
[ ] Rough ROI calculation
[ ] Go/no-go decision

Day 3-4: Planning
[ ] Define target metrics
[ ] Estimate budget (realistic x2.5)
[ ] Assemble team
[ ] Secure resources

Day 5: Validation Setup
[ ] Choose base model (or start data curation)
[ ] Setup infrastructure (GPU access)
[ ] Prepare small pilot experiment
[ ] Define kill criteria

Week 2: Pilot
[ ] Run small-scale pilot (1B model, 10B tokens)
[ ] Evaluate results
[ ] Adjust or pivot
[ ] Commit to full project or cancel

Week 3+: Execution
[ ] Follow this guideline phase by phase
[ ] Track checklist
[ ] Document learnings
[ ] Iterate based on data
```

---

**Remember:**
> "The papers show success; the reality is struggle. But struggle with a process beats struggle without one."

**This guideline is your process. Use it, adapt it, succeed.**

---

**작성일**: 2025-11-11
**기반**: GLM-4.5, MiniMax-M1, Kimi K2, SmolLM3 Playbook 분석
**대상**: 2-10명 작은 팀
**목표**: 실제로 동작하는 월드클래스 모델 구축 프로세스

---

**라이센스**: MIT License
**기여**: GitHub Issues/PRs welcome
**문의**: [your-contact]

**Good luck building world-class models! 🚀**
