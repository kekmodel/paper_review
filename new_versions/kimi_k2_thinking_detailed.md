# Kimi K2 Thinking: 공식 발표 상세 분석

## 📋 공식 정의

**Kimi K2 Thinking**: "Our best open-source thinking model"

**핵심 컨셉:**
> "Built as a **thinking agent**, it reasons **step by step** while **using tools**"

---

## 🎯 핵심 혁신

### 1. Test-Time Scaling의 두 축

**전통적 모델:**
```
Test-time scaling = More inference compute
(예: O1의 longer chain-of-thought)
```

**K2 Thinking:**
```
Test-time scaling = Thinking tokens + Tool calling steps
                   ↓                    ↓
              Internal reasoning    External actions
```

**획기적인 점:**
- **Thinking tokens**: 내부 추론 (O1-style)
- **Tool calling steps**: 외부 도구 사용 (최대 200-300 sequential calls!)
- **결합**: Reasoning과 Action을 interleave

---

### 2. 200-300 Sequential Tool Calls

**규모의 혁신:**
```
기존 모델: 5-10 tool calls per task
K2 Thinking: 200-300 sequential tool calls
→ 20-60배 증가!
```

**의미:**
- **Without human interference**: 완전 자율적
- **Coherent reasoning across hundreds of steps**: 일관된 추론 유지
- **Complex problem solving**: 매우 복잡한 문제 해결 가능

**프로세스 예시 (PhD-level math problem):**
```
23 interleaved reasoning and tool calls:
Think → Search → Think → Python → Think → Search → Think → ...
(반복 23회!)
```

---

## 📊 성능 하이라이트

### State-of-the-Art 달성

| Benchmark | Category | K2 Thinking | Comparison |
|-----------|----------|-------------|------------|
| **HLE (with tools)** | Agentic Reasoning | **44.9%** | GPT-5: 41.7%, Claude 4.5: 32.0% |
| **BrowseComp** | Agentic Search | **60.2%** | GPT-5: 54.9%, Human: 29.2% |
| **SWE-bench Verified** | Agentic Coding | **71.3%** | GPT-5: 74.9%, Claude: 77.2% |
| **AIME 2025 (w/ python)** | Math Reasoning | **99.1%** | GPT-5: 99.6% (거의 동일!) |
| **LiveCodeBench V6** | Competitive Programming | **83.1%** | GPT-5: 87.0% |

### 특징적 강점

**1. Agentic Search (압도적):**
- BrowseComp: 60.2% vs Human 29.2% (**2배 이상**)
- BrowseComp-ZH: 62.3%
- Seal-0: 56.3%
- FinSearchComp-T3: 47.4%

**2. Multilingual Coding:**
- SWE-Multilingual: **61.1%** (GPT-5: 55.3%)
- Multi-SWE-bench: 41.9%

**3. Heavy Mode (병렬 전략):**
- HLE (heavy): **51.0%** (vs 44.9% standard)
- AIME 2025: **100.0%** (완벽!)
- HMMT 2025: **97.5%**

---

## 🔬 기술적 세부사항

### Architecture: Original K2 기반

**Base Model:**
- 1.04T total params, 32B activated (MoE)
- 384 experts, sparsity=48
- 128K context → **256K context** (K2 Thinking에서 확장!)

**추가 능력:**
- **Thinking mode**: Extended reasoning with visible thinking tokens
- **Tool integration**: Search, Python, Web browsing natively

---

### Training: Quantization-Aware Training (QAT)

**문제:**
```
Thinking models → Excessive decoding lengths
Low-bit quantization → Substantial performance drops
```

**해결책: QAT during Post-training**
```
INT4 weight-only quantization on MoE components
↓
Native INT4 inference with ~2x speed
↓
State-of-the-art performance maintained!
```

**결과:**
- ✅ **2배 빠른 generation speed**
- ✅ Performance 유지 (모든 benchmark INT4로 측정)
- ✅ GPU memory 절약

**혁신적 이유:**
- 대부분의 thinking models: Quantization 시 성능 급락
- K2 Thinking: QAT로 문제 해결
- → Long-context thinking + Efficiency 동시 달성

---

### Inference Configuration

**Temperature:** 1.0 (creative, diverse reasoning)

**Context Length:** 256K tokens

**Thinking Token Budgets:**
- HLE, AIME, HMMT, GPQA: **96K tokens**
- IMO, LiveCodeBench, OJ-Bench: **128K tokens**
- Longform Writing: **32K tokens**

**Tool Call Limits:**
- HLE: **120 steps** (48K tokens per step)
- Agentic search: **300 steps** (24K tokens per step)

**Context Management:**
- Tool output이 256K 초과 시 이전 outputs 숨김 (simple strategy)

---

## 🎨 실제 사례 분석

### 사례 1: PhD-level Mathematics (23 Tool Calls)

**문제:** Hyperbolic space sampling의 log pdf 계산

**K2 Thinking의 프로세스:**
```
1. Think: Parse the complex problem
2. Search: "hyperbolic normal distribution pdf Lorentz model"
3. Think: Analyze search results
4. Search: "hyperbolic wrapped normal pdf"
5. Think: Need to derive pdf ourselves
6. Search: Papers on wrapped normal distribution
7. Think: Found relevant paper (Nagano et al., 2019)
8. Python: Compute matrix inverse
9. Python: Calculate log determinant
10. Think: Verify numeric patterns
11. Python: Test coefficients for multiple n
12. Think: Identify closed form
13. Python: Verify formula
... (총 23 interleaved steps)
```

**최종 답:**
```
ℓ_k(n) = (n-1)ln(k) + (1/2)ln(n+1) - (2 - 1/n)k²
```

**특징:**
- Symbolic math + Numeric verification
- Literature search + Derivation
- Hypothesis generation + Testing
- 완전히 자율적으로 PhD-level 문제 해결

---

### 사례 2: Agentic Search (Jimmy Gary Jr.)

**문제:**
> "이 사람을 찾아라:
> - 1860-1890 설립 대학 졸업
> - 대학 운동선수, NFL 짧게 활동
> - 2010-2020 외계인 침공 SF 영화 출연
> - 2010-2020 교도소 드라마에서 교정관 역할 (잘못된 수감자 석방 에피소드)
> - 인터뷰에서 2001-2010 의학 드라마 캐릭터가 가장 좋아하는 역할이라고 말함
>
> SF 영화의 캐릭터 이름은?"

**K2 Thinking의 프로세스 (일부):**
```
1. Think: Parse constraints, 5가지 조건 분석
2. Search: "Corrections Officer releases wrong inmate episode"
3. Think: Narrow down to OITNB
4. Search: "University of Kansas football player actor"
5. Think: Not matching, need different approach
6. Search: "Orange Is the New Black wrong inmate release"
7. Think: Found episode "We Can Be Heroes", CO Sikowitz
8. Search: "Eden Malyn former football player"
9. Think: Not NFL background, wrong person
10. Search: "Jimmy Gary Jr. football player"
11. Think: Match! WVU (founded 1867), Seattle Seahawks
12. Search: "Jimmy Gary Jr. By Dawn film"
13. Think: "By Dawn" (2018) alien invasion film confirmed
14. Search: "Jimmy Gary Jr. By Dawn character"
... (계속)
```

**최종 답:** **Rudy Cox** (Jimmy Gary Jr. played in "By Dawn")

**특징:**
- 복잡한 multi-constraint search
- Dead ends에서 pivot (Eden Malyn → Jimmy Gary Jr.)
- Cross-reference multiple sources
- Systematic verification

---

### 사례 3: Creative Writing (Science Fiction)

**Prompt:** "Cloud가 주인공인 SF 단편 (planetary weather-control system, free will during lightning storm)"

**K2 Thinking 버전:**
- Title: "Cumulus-7's First Choice"
- **Research-based**: Köhler theory, terminal velocity, Wegener-Bergeron-Findeisen diffusion
- **Scientific accuracy**: 구체적 수치 (128-bit register, 32 kA lightning, 30,000 K temperature)
- **Philosophical depth**: Consciousness as phase transition, free will vs programming
- **Narrative structure**: First-person, gradual awakening, climactic choice

**Quality:**
- Not generic SF
- Deep scientific grounding
- Emotional resonance
- Thematically coherent

---

## 🆚 비교 분석

### K2 Thinking vs GPT-5 (High)

| Aspect | K2 Thinking | GPT-5 (High) | Winner |
|--------|-------------|--------------|--------|
| **HLE (with tools)** | 44.9% | 41.7% | ✅ K2 |
| **BrowseComp** | 60.2% | 54.9% | ✅ K2 |
| **SWE-bench Verified** | 71.3% | 74.9% | GPT-5 |
| **AIME 2025 (w/ python)** | 99.1% | 99.6% | ≈ Tie |
| **HMMT 2025 (w/ python)** | 95.1% | 96.7% | ≈ Tie |
| **GPQA-Diamond** | 84.5% | 85.7% | ≈ Tie |
| **LiveCodeBench V6** | 83.1% | 87.0% | GPT-5 |
| **SWE-Multilingual** | 61.1% | 55.3% | ✅ K2 |
| **OJ-Bench (cpp)** | 48.7% | 56.2% | GPT-5 |
| **Status** | Open-source | Proprietary | ✅ K2 |

**결론:**
- **K2 강점**: Agentic search, multilingual coding, open-source
- **GPT-5 강점**: Pure coding, competitive programming
- **Overall**: Competitive across the board, some areas K2 leads

---

### K2 Thinking vs Claude Sonnet 4.5 (Thinking)

| Benchmark | K2 Thinking | Claude 4.5 | Winner |
|-----------|-------------|------------|--------|
| **HLE (with tools)** | 44.9% | 32.0% | ✅ K2 (+40%) |
| **BrowseComp** | 60.2% | 24.1% | ✅ K2 (+150%) |
| **SWE-bench Verified** | 71.3% | 77.2% | Claude |
| **AIME 2025 (w/ python)** | 99.1% | 100.0% | ≈ Tie |
| **GPQA-Diamond** | 84.5% | 83.4% | ≈ Tie |
| **Longform Writing** | 73.8% | 79.8% | Claude |

**결론:**
- **K2 압도**: Agentic tasks (특히 search)
- **Claude 우위**: Coding, writing
- **Gap 크기**: Agentic search에서 K2가 압도적 (2배+ 차이)

---

### K2 0905 (Original) vs K2 Thinking

| Benchmark | K2 0905 | K2 Thinking | Improvement |
|-----------|---------|-------------|-------------|
| **HLE (with tools)** | 21.7% | **44.9%** | **+107%** |
| **BrowseComp** | 7.4% | **60.2%** | **+713%** |
| **SWE-bench Verified** | 69.2% | **71.3%** | +3% |
| **AIME 2025 (w/ python)** | 75.2% | **99.1%** | **+32%** |
| **HMMT 2025 (w/ python)** | 70.4% | **95.1%** | **+35%** |
| **LiveCodeBench V6** | 56.1% | **83.1%** | **+48%** |

**결론:**
- **Massive gains**: Agentic tasks (7배+ BrowseComp!)
- **Major gains**: Math reasoning (2배 AIME)
- **Modest gains**: Coding (이미 높았음)

---

## 💡 Heavy Mode: 병렬 전략

### 정의

**Heavy Mode:**
> "Employs an efficient parallel strategy: first rolls out **8 trajectories simultaneously**, then **reflectively aggregates** all outputs to generate the final result."

**프로세스:**
```
1. Parallel Rollout: 8 independent reasoning paths
2. Generate 8 candidate answers
3. Reflective Aggregation: Analyze all 8 outputs
4. Final Answer: Synthesized best result
```

### 성능 (Heavy vs Standard)

| Benchmark | Standard | Heavy Mode | Improvement |
|-----------|----------|------------|-------------|
| **HLE (with tools)** | 44.9% | **51.0%** | +14% |
| **AIME 2025 (w/ python)** | 99.1% | **100.0%** | +0.9% (완벽!) |
| **HMMT 2025 (w/ python)** | 95.1% | **97.5%** | +2.4% |

**특징:**
- ✅ Significant gains on hard problems (HLE)
- ✅ Ceiling performance on easier benchmarks (AIME 100%)
- ✅ Efficient parallelization (not just averaging)

**비교:**
- GPT-5 Pro: Similar concept (heavy mode)
- K2 Thinking Heavy: 8 trajectories + reflective aggregation

---

## 🎓 Capability Breakdown

### 1. Agentic Reasoning (최강)

**Humanity's Last Exam (HLE):**
- Text-only, with tools: **44.9%** (SOTA among open-source)
- Expert-level questions, 100+ subjects
- Long-horizon reasoning + tool use

**특징:**
- Planning → Reasoning → Executing → Adapting
- Hundreds of steps coherently
- Deep structured reasoning

---

### 2. Agentic Search (압도적)

**BrowseComp:**
- Score: **60.2%**
- Human baseline: 29.2%
- **2배 이상 human performance**

**능력:**
- Continuously browse, search, reason
- Hard-to-find real-world web information
- Goal-directed, web-based reasoning
- Dynamic, information-rich environments

**프로세스:**
```
Think → Search → Browse → Think → Code → Think → ...
(Interleaved reasoning cycles)
```

**특징:**
- Dynamic hypothesis generation and refinement
- Evidence verification
- Coherent answer construction
- Ambiguous, open-ended problems → Clear, actionable subtasks

---

### 3. Agentic Coding (강력)

**SWE-bench Verified:**
- Score: 71.3%
- Close to GPT-5 (74.9%), Claude 4.5 (77.2%)

**SWE-Multilingual:**
- Score: **61.1%**
- Beats GPT-5 (55.3%)

**강점:**
- HTML, React, component-intensive front-end
- Translating ideas → Fully functional, responsive products
- Multi-step development workflows
- Precision and adaptability

**Examples from single prompt:**
- Component-heavy websites
- Word clone
- Complex interactive UIs

---

### 4. General Capabilities

**Creative Writing:**
- Completeness and richness
- Strong command of style and instruction
- Vivid, imaginative (poetic imagery, deeper associations)
- Human, emotional, purposeful
- Thematic depth and resonance

**Practical Writing:**
- Reasoning depth, perspective breadth
- High precision instruction adherence
- Systematic, thorough coverage
- Rigorous, logically coherent
- Academic, research, long-form analytical excellence

**Personal & Emotional:**
- Empathy and balance
- Thoughtful, specific reflections
- Nuanced perspectives
- Actionable next steps
- Grounded, practical, genuinely human tone

---

## 🔧 Technical Implementation

### Tool Suite

**Available Tools:**
1. **Search**: Web search for information
2. **Python**: Code interpreter for computation
3. **Web Browsing**: Navigate and extract web content

**Integration:**
- Native tool calling (not external API)
- 200-300 sequential calls without human interference
- Reasoning integrated with tool use

### Context Management

**Challenge:**
```
256K context limit
+
300 tool calls with outputs
→ Context overflow!
```

**Solution:**
```
Simple strategy: Hide all previous tool outputs
when accumulated input exceeds 256K
```

**Trade-off:**
- ✅ Prevents overflow
- ⚠️ Loses historical context
- Future: More sophisticated strategies (e.g., summarization)

### Safety Measures

**Data Leakage Prevention:**
- Blocked Hugging Face access during testing
  - Reason: HLE questions might be on HF
  - Without blocking: 51.3% HLE
  - With blocking: 44.9% HLE (reported score)

**Fair Comparison:**
- Rigorous methodology
- o3-mini as judge (HLE standard)
- Verbatim official prompts

---

## 📈 Evaluation Methodology

### Benchmark Categories

**1. Reasoning Tasks:**
- HLE (w/ and w/o tools)
- AIME, HMMT, IMO
- GPQA-Diamond

**2. General Tasks:**
- MMLU-Pro, MMLU-Redux
- Longform Writing
- HealthBench

**3. Agentic Search Tasks:**
- BrowseComp (EN & ZH)
- Seal-0
- FinSearchComp-T3
- Frames

**4. Coding Tasks:**
- SWE-bench (Verified, Multilingual, Multi)
- SciCode
- LiveCodeBench V6
- OJ-Bench (cpp)
- Terminal-Bench

### Testing Details

**Sampling Strategy:**
- AIME, HMMT (no tools): **avg@32** (32 runs averaged)
- AIME, HMMT (w/ python): **avg@16** (16 runs)
- IMO-AnswerBench: **avg@8** (8 runs)
- Agentic search: **avg@4** (4 runs)
- Coding tasks: **avg@5** (5 runs)

**Rationale:**
- High variance in reasoning tasks → Multiple runs
- Stochastic tool use → Averaging essential

### Evaluation Infrastructure

**In-house Harness:**
- Derived from SWE-agent
- Clamped context windows (Bash, Edit tools)
- Rewritten system prompts for task semantics

**Agent Framework:**
- Terminal-Bench: Terminus-2 (default)
- JSON parser provided

---

## 🌟 Key Innovations Summary

### 1. Dual Test-Time Scaling

**Traditional:**
```
More thinking tokens → Better reasoning
```

**K2 Thinking:**
```
More thinking tokens + More tool calls → Better agentic reasoning
```

**Impact:**
- First model to systematically scale both axes
- 200-300 tool calls (unprecedented scale)
- Coherent long-horizon reasoning

---

### 2. Quantization-Aware Training (QAT)

**Problem:**
```
Thinking models → Long sequences
Quantization → Performance drop
```

**Solution:**
```
QAT during post-training
→ INT4 inference
→ 2x speed, maintained performance
```

**Impact:**
- First thinking model with native INT4 support
- Deployment efficiency dramatically improved
- Open-source model competitive with proprietary

---

### 3. Interleaved Reasoning & Tool Use

**Not:**
```
Think → Act → Think → Act (simple alternation)
```

**But:**
```
Think (plan) → Search → Think (analyze) → Python (verify)
→ Think (refine) → Browse → Think (synthesize) → ...
```

**Impact:**
- Truly integrated reasoning and action
- Not just tool-calling, but tool-integrated thinking
- Enables solving problems impossible for reasoning-only or tool-only

---

### 4. Heavy Mode (Parallel Reasoning)

**Strategy:**
```
8 parallel trajectories → Reflective aggregation → Final answer
```

**Impact:**
- 51% HLE (vs 44.9% standard, +14%)
- 100% AIME (perfect score!)
- Demonstrates effective test-time scaling

---

## 🎯 Use Case Analysis

### When to Use K2 Thinking

#### ✅ Ideal For:

**1. Complex Research Tasks**
- Literature review
- Multi-source information synthesis
- Hard-to-find information (BrowseComp 60%)

**2. Long-Horizon Problem Solving**
- PhD-level math/science
- Multi-step reasoning (200-300 steps)
- Hypothesis generation and testing

**3. Agentic Coding**
- Full-stack development from prompt
- Component-heavy front-end
- Multilingual projects (61.1% SWE-Multilingual)

**4. Exploratory Analysis**
- Open-ended questions
- Ambiguous problems → Structured solutions
- Evidence-based reasoning

#### ⚠️ Overkill For:

**1. Simple Queries**
- Factual QA (use standard K2 0905)
- Quick calculations
- Short responses

**2. Real-Time Applications**
- Latency-sensitive tasks
- Even with INT4, thinking mode slower

**3. Cost-Sensitive Tasks (if API pricing high)**
- 200-300 tool calls expensive
- Consider standard mode first

---

### K2 0905 vs K2 Thinking 선택 가이드

| Task Type | Recommend | Why |
|-----------|-----------|-----|
| **Simple coding** | K2 0905 | 69.2% SWE-bench already good, faster |
| **Complex multi-file coding** | K2 Thinking | 71.3% SWE-bench, better planning |
| **Web search** | K2 Thinking | 60% BrowseComp (vs 7.4% standard!) |
| **Math competition** | K2 Thinking | 99.1% AIME (vs 75.2% standard) |
| **Quick chat** | K2 0905 | Low latency, good enough |
| **Research task** | K2 Thinking | Long-horizon reasoning essential |
| **Front-end development** | K2 Thinking | Component-heavy strength |

---

## 🔮 Future Implications

### 1. Test-Time Scaling의 새로운 패러다임

**Before:**
```
Pre-training compute → Model capability (fixed at inference)
```

**OpenAI O1:**
```
Test-time compute → Extended reasoning → Better performance
```

**K2 Thinking:**
```
Test-time compute → Reasoning + Tool use → Agentic performance
```

**Next:**
```
Test-time compute → Reasoning + Tool use + Multi-agent collaboration?
```

---

### 2. Open-Source의 경쟁력

**K2 Thinking vs Proprietary:**
- HLE: Beats GPT-5 (44.9% vs 41.7%)
- BrowseComp: Beats GPT-5 (60.2% vs 54.9%)
- Overall: Competitive across board

**의미:**
- Open-source can match or exceed closed-source
- Community can build on K2 Thinking
- Democratization of advanced AI

---

### 3. Agentic AI의 미래

**현재 (K2 Thinking):**
- 200-300 sequential tool calls
- Think → Search → Code → Browse cycles
- Fully autonomous

**미래 (추측):**
```
1000+ tool calls?
Multi-agent collaboration?
Self-improving agents (learning from experience)?
Real-world deployment (not just benchmarks)?
```

---

## 📊 성능 비교 시각화

### HLE (with tools) - Agentic Reasoning

```
K2 Thinking:  ████████████████████████████████████████████ 44.9%
GPT-5 (High): ████████████████████████████████████████   41.7%
Grok-4:       ████████████████████████████████████████   41.0%
Claude 4.5:   ████████████████████████████████         32.0%
K2 0905:      █████████████████████                     21.7%
DeepSeek-V3:  ████████████████████                      20.3%
```

### BrowseComp - Agentic Search

```
K2 Thinking:  ████████████████████████████████████████████████████████████ 60.2%
GPT-5:        ███████████████████████████████████████████████████         54.9%
DeepSeek-V3:  ████████████████████████████████████████                    40.1%
Human:        █████████████████████████████                               29.2%
Claude 4.5:   ████████████████████████                                    24.1%
K2 0905:      ███████                                                       7.4%
```

### SWE-bench Verified - Agentic Coding

```
Claude 4.5:   █████████████████████████████████████████████████████████████████████████████ 77.2%
GPT-5:        ██████████████████████████████████████████████████████████████████████████  74.9%
K2 Thinking:  █████████████████████████████████████████████████████████████████████     71.3%
K2 0905:      ███████████████████████████████████████████████████████████████████       69.2%
DeepSeek-V3:  ██████████████████████████████████████████████████████████████████        67.8%
```

---

## 🎓 배울 점

### 1. **Reasoning과 Action은 분리될 수 없다**

**Traditional AI:**
```
Think (internal) → Act (external)
Separate phases
```

**K2 Thinking:**
```
Think-Act-Think-Act... (interleaved)
Reasoning is informed by action results
Actions are guided by ongoing reasoning
```

**교훈:** True intelligence requires tight integration

---

### 2. **Test-Time Scaling의 다차원성**

**Dimensions:**
1. Thinking tokens (internal reasoning)
2. Tool calls (external actions)
3. Parallel trajectories (Heavy Mode)

**교훈:** Scaling multiple dimensions > Scaling one dimension

---

### 3. **Quantization ≠ Performance Loss (with QAT)**

**Myth:**
```
INT4 quantization → 5-10% performance drop
Especially bad for long sequences
```

**Reality (K2 Thinking):**
```
QAT during post-training
→ INT4 with maintained performance
→ 2x speed bonus
```

**교훈:** Proper training can eliminate quantization gap

---

### 4. **Open-Source Can Lead**

**K2 Thinking:**
- Beats GPT-5 on HLE, BrowseComp
- Competitive on most benchmarks
- Fully open-source

**교훈:** Open-source is not second-class

---

## 🔗 Availability

**Chat Mode:** kimi.com
- Live now
- Subset of tools, reduced turns (for speed)
- May not reproduce benchmark scores

**Agentic Mode:** Coming soon
- Full capabilities
- 200-300 tool calls
- Benchmark-matching performance

**API:** Kimi K2 Thinking API
- Available now
- Full capabilities

**Weights:**
- Likely on Hugging Face (following K2 0905 pattern)
- INT4 quantized version

---

## 📝 결론

### Kimi K2 Thinking의 의의

**기술적:**
1. ✅ Dual test-time scaling (thinking + tool calling)
2. ✅ QAT로 INT4 efficiency 달성
3. ✅ 200-300 sequential tool calls (unprecedented)
4. ✅ Heavy Mode 병렬 전략

**성능적:**
1. ✅ SOTA on agentic tasks (HLE, BrowseComp)
2. ✅ Competitive on coding, math
3. ✅ Beats proprietary models in key areas
4. ✅ Open-source accessibility

**전략적:**
1. ✅ Proves open-source competitiveness
2. ✅ Establishes new paradigm (thinking + action)
3. ✅ Enables complex real-world applications
4. ✅ Community can build on top

### 미래 방향

**Short-term:**
- Full agentic mode release
- More tools integration
- Longer context (256K → 512K?)

**Long-term:**
- Multi-agent collaboration
- Self-improving loops
- Real-world deployment at scale
- Possibly multimodal (vision + thinking + action)

---

**공식 발표:** Moonshot AI
**출처:** https://moonshotai.github.io/Kimi-K2/thinking.html
**작성일:** 2025-11-10
