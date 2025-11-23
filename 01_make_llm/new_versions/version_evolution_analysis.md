# 버전 진화 분석: 이전 vs 새로운 버전 (전승과 변화)

## 📋 개요

이 문서는 세 모델 시리즈의 버전 간 진화를 상세히 분석합니다:

1. **GLM-4.5 → GLM-4.6**
2. **MiniMax-M1 → MiniMax-M2**
3. **Kimi K2 (0905) → Kimi K2 Thinking**

각 모델의 **전승(Inheritance)**, **변화(Changes)**, **전략적 방향(Strategic Direction)**을 분석하여, 차세대 LLM 개발의 트렌드를 파악합니다.

---

## 1️⃣ GLM 시리즈: 4.5 → 4.6

### 타임라인

```
GLM-4.5 (2024 중후반) → GLM-4.6 (2025년 9월 30일)
약 10-12개월 간격
```

### 전승 (Inheritance): 핵심 유지

#### Architecture (변화 없음)

| Component | GLM-4.5 | GLM-4.6 | Status |
|-----------|---------|---------|--------|
| **Total Params** | 355B | 355B (추정) | ✅ 유지 |
| **Activated Params** | 32B | 32B (추정) | ✅ 유지 |
| **MoE Architecture** | Yes | Yes | ✅ 유지 |
| **Attention Heads** | 96 | 96 (추정) | ✅ 유지 |
| **Architecture Type** | Deeper layers | Same | ✅ 유지 |

**근거:**
- 논문/블로그에서 architecture 변경 언급 없음
- 성능 improvement가 architecture보다 capabilities에 집중
- MoE 355B/32B는 GLM의 identity

#### Core Capabilities (유지)

```
✅ Multi-Token Prediction (speculative decoding)
✅ Hybrid Reasoning (thinking vs direct mode)
✅ QK-Norm (attention stability)
✅ Multilingual capabilities
```

**분석:**
- GLM-4.5의 핵심 혁신들은 모두 유지
- Architecture redesign 없이 capabilities 확장 전략

---

### 변화 (Changes): 능력 확장

#### 1. Context Window 확장

**GLM-4.5:**
```
128K tokens
→ Long-context tasks 가능
→ But not extremely long
```

**GLM-4.6:**
```
200K tokens (+56% increase)
→ More complex agentic tasks
→ Longer code files
→ Extended document processing
```

**의미:**
- 128K → 200K는 moderate increase (not extreme like 1M)
- Practical sweet spot targeting
- MiniMax-M1의 1M은 overkill이라는 인식 반영

**Use Cases:**
- 128K: ~50K words, ~200 pages
- 200K: ~78K words, ~312 pages
- → Full book processing, large codebase analysis

---

#### 2. Tool Use in Inference (혁신!)

**GLM-4.5:**
```
Function calling: Request → Model suggests tools → External execution
Separation: Internal reasoning vs External actions
```

**GLM-4.6:**
```
Tool use DURING inference
→ Model internally uses tools while reasoning
→ Tighter integration
```

**추정 메커니즘:**

**Before (4.5):**
```python
# Separate phases
reasoning_output = model.generate(prompt)
# reasoning_output contains tool calls

tool_results = execute_tools(reasoning_output.tool_calls)
final_output = model.generate(prompt + tool_results)
```

**After (4.6, 추정):**
```python
# Integrated
output = model.generate_with_tools(
    prompt,
    available_tools=[search, python, browser],
    max_tool_calls=10
)
# Model internally calls tools during generation
```

**GLM-4.6 블로그 언급:**
> "Tool use during inference... strengthening overall capability"

**비교:**
- O1-style: Internal thinking (no external tools)
- GLM-4.5: External tool calling (separated)
- **GLM-4.6: Internal + External** (integrated)
- K2 Thinking: 200-300 sequential tool calls (most extreme)

**의미:**
- This is a paradigm shift
- Reasoning ↔ Tool use가 interleaved
- 더 sophisticated agentic behavior 가능

---

#### 3. Coding Performance 향상

**GLM-4.5:**
```
SWE-bench Verified: 64.2%
Focus: General coding capability
```

**GLM-4.6:**
```
CC-Bench: 48.6% win rate vs Claude Sonnet 4
Better real-world: Claude Code, Cline, Roo Code, Kilo Code
특히 "visually polished front-end pages" 생성 강화
```

**차이점 분석:**

| Aspect | GLM-4.5 | GLM-4.6 |
|--------|---------|---------|
| **Benchmark** | SWE-bench (academic) | CC-Bench (real-world) |
| **Focus** | General coding | Developer workflow integration |
| **Strength** | ? | Front-end, UI components |
| **Integration** | Standalone | Claude Code, Cline, etc. |

**의미:**
- Academic benchmark → Real-world performance
- Component-heavy tasks 강화
- Developer tool integration 중시
- → Practical usefulness 우선

---

#### 4. Token Efficiency

**GLM-4.5:**
```
Baseline token usage
```

**GLM-4.6:**
```
15% fewer tokens to complete same tasks
```

**달성 방법 (추정):**

1. **Better Planning:**
   ```
   Before: Trial-and-error → More tokens
   After: Better initial plan → Direct solution
   ```

2. **Concise Output:**
   ```
   Before: Verbose explanations
   After: More direct, still clear
   ```

3. **Efficient Tool Use:**
   ```
   Before: Multiple tool calls for same info
   After: Single comprehensive tool call
   ```

**의미:**
- Cost reduction (fewer tokens = lower cost)
- Faster inference (less generation)
- Better reasoning (efficiency ≈ better understanding)

**Impact:**
```
같은 비용으로 15% 더 많은 작업 가능
또는
같은 작업을 15% 저렴하게
```

---

#### 5. Refined Writing & Agent Performance

**Writing Quality:**
```
GLM-4.5: Good alignment
GLM-4.6: Better alignment with human preferences
         More natural role-playing
         Improved style & readability
```

**Agent Performance:**
```
GLM-4.5: TAU-Bench 70.1%, Tool-calling 90.6%
GLM-4.6: "Enhanced tool-using & search-based agent capabilities"
         Tool use IN inference (game-changer)
```

---

### 전략적 분석

#### Focus Shift

**GLM-4.5:**
```
🎯 Goal: Balanced ARC (Agentic, Reasoning, Coding)
📊 Strength: Pure math (91% AIME)
🔬 Approach: Architecture innovation (deeper, more heads)
```

**GLM-4.6:**
```
🎯 Goal: Enhanced Agentic + Practical coding
📊 Strength: Real-world developer tools, Tool inference
🔬 Approach: Capability expansion (context, integration)
```

#### Market Positioning

**GLM-4.5:**
```
Academic/Research: Strong pure reasoning
General Purpose: Balanced capabilities
Competitive Edge: Efficiency (355B/32B)
```

**GLM-4.6:**
```
Developer-Focused: IDE integration (Claude Code, Cline)
Agentic Applications: Tool use in inference
Enterprise Ready: 200K context, refined writing
```

#### Technology Maturity

**GLM-4.5:**
```
Stage: Research → Production
Focus: Prove architecture works
Metric: Benchmark performance
```

**GLM-4.6:**
```
Stage: Production → Optimization
Focus: Real-world usefulness
Metric: Developer productivity, Win rate vs competitors
```

---

### 성능 비교 (가용 데이터)

| Benchmark | GLM-4.5 | GLM-4.6 | Change |
|-----------|---------|---------|--------|
| **Context** | 128K | 200K | +56% ✅ |
| **SWE-bench V** | 64.2% | ? | ? |
| **CC-Bench** | 90.6% (tool) | 48.6% win vs Sonnet 4 | Different metric |
| **AIME 24** | 91.0% | ? (likely maintained) | ≈ |
| **TAU-Bench** | 70.1% | ? (enhanced) | ⬆️ |
| **Token Efficiency** | Baseline | -15% | ✅ |
| **Tool Use** | Function calling | **In inference** | 🚀 |

**해석:**
- Core strengths 유지 (math, reasoning)
- New capabilities 추가 (tool inference, extended context)
- Real-world focus (coding, agent, efficiency)

---

### 배울 점

#### 1. "Incremental Innovation > Radical Redesign"

**교훈:**
- Architecture redesign 없이도 major improvement 가능
- Context 확장 (128K → 200K)
- Tool integration (external → internal)
- Efficiency (token reduction 15%)

**의미:**
- Not every version needs new architecture
- Capabilities can evolve on stable foundation
- Cost-effective development

---

#### 2. "Tool Use in Inference"의 중요성

**Before (Separate):**
```
Think → Request tool → Execute → Think → ...
Latency: High (external calls)
Control: Limited (wait for results)
```

**After (Integrated, GLM-4.6):**
```
Think-with-tools → ...
Latency: Lower (internal)
Control: Better (adaptive use)
```

**Similar to:**
- Kimi K2 Thinking: 200-300 tool calls
- O1 + Tools = GLM-4.6 approach

---

#### 3. Real-World > Academic Benchmarks

**GLM-4.5:**
- AIME 91.0% (impressive!)
- SWE-bench 64.2%
- TAU-Bench 70.1%

**GLM-4.6:**
- CC-Bench 48.6% win vs Sonnet 4
- "Better real-world performance"
- Developer tool integration

**Lesson:**
- Academic benchmarks saturating
- Real-world performance matters more
- Integration > Standalone capability

---

## 2️⃣ MiniMax 시리즈: M1 → M2

### 타임라인

```
MiniMax-M1 (2025년 초-중반?) → MiniMax-M2 (2025년 10월 27일)
약 6-10개월 간격 (추정)
```

### 전승 (Inheritance): 부분적 유지 (추정)

**⚠️ 주의**: M2의 architecture details 미공개, 많은 부분이 추정

#### Likely Inherited

| Component | M1 | M2 (추정) | Confidence |
|-----------|----|-----------|-----------|
| **MoE Architecture** | Yes (456B/45.9B) | Likely Yes | ⭐⭐⭐⭐ High |
| **Attention Mechanism** | Hybrid (Linear + Softmax) | Likely modified | ⭐⭐⭐ Medium |
| **Efficiency Focus** | Yes (25% FLOPs) | Yes (8% price) | ⭐⭐⭐⭐⭐ Very High |
| **RL Algorithm** | CISPO | Possibly evolved | ⭐⭐ Low |

**근거:**

**MoE 유지 가능성 높음:**
- M2도 large model (Hugging Face 언급)
- MoE는 efficiency의 기반
- Activated params 적게 유지 필요 (cost)

**Attention 변형 가능성:**
- M1: 1M context (hybrid linear attention)
- M2: ? context (likely 128K-512K for efficiency)
- Linear attention 부분 유지, but scale down

**Efficiency DNA:**
- M1: 25% FLOPs of DeepSeek R1
- M2: 8% price of Claude, 2x speed
- → Efficiency가 identity

---

### 변화 (Changes): Radical Pivot

#### 1. Strategic Focus 전환

**MiniMax-M1: Research-Oriented**
```
🎯 Goal: Prove technical innovation
🔬 Innovation: Lightning attention (1M context)
🔬 Innovation: CISPO RL algorithm
📊 Metric: FLOPs efficiency, Long-context benchmarks
🎓 Audience: Research community, Technical users
```

**MiniMax-M2: Production-Oriented**
```
🎯 Goal: Real-world deployment, Market adoption
💼 Focus: "A model born for Agents and code"
💰 Innovation: Cost efficiency (8% Claude price)
⚡ Innovation: Speed (2x faster, ~100 tokens/sec)
🛠️ Integration: Claude Code, Cursor, Cline, Kilo Code
👥 Audience: Individual developers, Small teams
```

**Pivot 규모:**
```
Research paper (M1) → Product launch (M2)
Academic benchmarks → Developer workflows
FLOPs → Price per token
Theory → Practice
```

---

#### 2. Context Length (추정)

**MiniMax-M1:**
```
1M tokens (1,000,000)
→ 8x longer than DeepSeek R1
→ Unprecedented scale
→ Hybrid linear attention 필수
```

**MiniMax-M2 (추정):**
```
? tokens (미공개)

가능성 1: 256K-512K (moderate)
근거: Efficiency focus, Most use cases don't need 1M

가능성 2: 128K (standard)
근거: Cost optimization, 대부분 충분

가능성 3: Still 1M
근거: Differentiation, Technical moat
→ 하지만 unlikely (cost 강조와 모순)
```

**추정: 256K-512K가 가능성 높음**

**이유:**
1. **Cost-performance balance**
   - 1M context는 expensive to serve
   - 256K-512K는 충분히 길면서 practical

2. **Use case analysis**
   - Agent tasks: 대부분 128K-256K 충분
   - Coding: Full repository ~256K
   - 1M은 niche use cases만 필요

3. **Efficiency claim**
   - 8% price, 2x speed
   - 1M context 유지하면서 달성 어려움
   - Context reduction이 cost reduction의 key

---

#### 3. Pricing & Speed

**MiniMax-M1:**
```
Pricing: Not disclosed (likely expensive for 1M context)
Speed: Baseline
Training: 3주, 512 H800 GPUs, ~$534K
```

**MiniMax-M2:**
```
Pricing: $0.30/M input, $1.20/M output
         → 8% of Claude Sonnet price
Speed: ~100 tokens/second
       → "Nearly double" inference speed vs Claude
       → 2x faster than M1 (likely)
```

**Cost Comparison:**

| Model | Input (per M) | Output (per M) | vs Claude |
|-------|---------------|----------------|-----------|
| Claude Sonnet | ~$3 | ~$15 | 100% |
| **MiniMax-M2** | **$0.30** | **$1.20** | **8%** |
| GPT-4 Turbo | ~$10 | ~$30 | ~200% |

**혁명적 이유:**
```
같은 $10로:
Claude: ~3.3M input or ~667K output
M2: ~33M input or ~8.3M output
→ 10배 더 많은 작업 가능!
```

**Speed Comparison:**

| Model | Tokens/sec | Relative |
|-------|------------|----------|
| Claude Sonnet | ~50 (추정) | 1x |
| **MiniMax-M2** | **~100** | **2x** |

**달성 방법 (추정):**

1. **Quantization:**
   - INT8 or INT4 inference
   - K2 Thinking의 QAT 참고

2. **Context Reduction:**
   - 1M → 256K-512K
   - Less memory, faster attention

3. **Optimized Serving:**
   - Better batching
   - Efficient kernels
   - Flash Attention variants

4. **Smaller Activated Params:**
   - M1: 45.9B activated
   - M2: Possibly less (30B-40B?)
   - MoE sparsity increase

---

#### 4. Developer Tool Integration

**MiniMax-M1:**
```
❌ No specific tool integration mentioned
🎓 Research-focused evaluation
📊 Academic benchmarks
```

**MiniMax-M2:**
```
✅ Claude Code integration
✅ Cursor integration
✅ Cline integration
✅ Kilo Code integration
✅ Droid integration

✅ Shell, Browser, Python interpreters
✅ MCP tools support
```

**의미:**

**M1:**
```
General model
→ Users need to integrate themselves
→ API-only
```

**M2:**
```
Developer-first model
→ Native IDE support
→ Drop-in replacement for Claude
→ "We use it ourselves" philosophy
```

**Development Philosophy:**

**M2 블로그 언급:**
> "To create a model meeting requirements, we must first be able to use it ourselves."

**Process:**
```
1. Internal developers use M2 in daily workflow
2. Identify pain points, iterate
3. Accumulated methods → Benchmarks
4. Real-world proven → Public release
```

**Lesson:**
- Dogfooding (eating own dog food)
- Real usage > Theoretical capability
- Developer experience matters

---

#### 5. Performance Profile

**MiniMax-M1: Research Strengths**
```
✅ Long-context: OpenAI o3, Claude 4 Opus 능가
✅ Extended reasoning
✅ SWE-bench: 56.0%
✅ AIME 2024: 86.0%
⚠️ Pure math: Lags DeepSeek-R1 (86% vs 90%+)
```

**MiniMax-M2: Production Strengths**
```
✅ Top-5 globally (Artificial Analysis 10-test)
✅ Tool use & search: "Very close to best overseas"
✅ Programming: "Among best in domestic market"
⚠️ "Slightly behind top overseas in pure programming"
```

**Trade-off 분석:**

| Aspect | M1 | M2 | Winner |
|--------|----|----|--------|
| **Long-context** | 1M (SOTA) | ? (practical) | M1 (technical) |
| **Pure Math** | 86% (good) | ? | ? |
| **Cost** | Expensive | 8% Claude | **M2** 🏆 |
| **Speed** | Baseline | 2x | **M2** 🏆 |
| **Integration** | None | Extensive | **M2** 🏆 |
| **Tool-use** | Good | "Very close to best" | ≈ |
| **Target User** | Researchers | Developers | Different |

**Strategic Choice:**
```
Sacrifice: Some benchmark performance (maybe)
Gain: Cost (12.5x cheaper), Speed (2x), Integration
Value: Market adoption > Technical excellence
```

---

### 전략적 분석

#### Technology Lifecycle

**M1: Technology Demonstration**
```
Stage: Proof of Concept
Goal: Show it's possible (1M context, CISPO)
Metric: FLOPs, Benchmark scores
Audience: Academia, Tech enthusiasts
Success: Published paper, Recognition
```

**M2: Market Capture**
```
Stage: Product-Market Fit
Goal: Get users (developers)
Metric: Adoption rate, User satisfaction
Audience: Developers, Startups
Success: Revenue, Market share
```

#### Competitive Strategy

**M1: Technical Differentiation**
```
vs DeepSeek-R1: "25% FLOPs"
vs Others: "1M context"
Moat: Technical innovation
```

**M2: Economic Disruption**
```
vs Claude: "8% price"
vs Others: "2x speed"
Moat: Cost efficiency
Strategy: Price war → Market capture
```

**Similar Playbooks:**
- AWS vs Traditional hosting: 10x cheaper → Dominance
- Chinese smartphone brands: Better specs, 50% price
- MiniMax-M2: Better integration, 8% price

**Risk:**
- "Cheap = Low quality" perception
- But: "Among best domestic", "Close to best overseas"
- → Quality + Price = Winning combination

---

#### Market Positioning Evolution

**M1:**
```
Position: "Research innovation leader"
Message: "Most efficient long-context model"
Competitors: DeepSeek-R1, O3 (technical)
```

**M2:**
```
Position: "Developer productivity tool"
Message: "Affordable Claude alternative for developers"
Competitors: Claude Sonnet, GPT-4o (commercial)
```

**Target Market Shift:**

**M1:**
```
Market: Research labs, Advanced users
Size: Small (thousands)
Value: Technical reputation
```

**M2:**
```
Market: Individual developers, Small teams, Startups
Size: Large (millions)
Value: Revenue, Ecosystem
```

---

### 배울 점

#### 1. "Research → Product Pivot"

**Pattern:**
```
Phase 1: Publish innovation (M1)
        → Build reputation
        → Prove technical capability

Phase 2: Productize (M2)
        → Simplify for users
        → Optimize for cost
        → Integrate with tools
```

**Examples:**
- Google BERT → Distilled BERT → Production APIs
- OpenAI GPT-3 → ChatGPT (simpler UX)
- **MiniMax M1 → M2** (cost focus)

**Lesson:**
- Research proves it works
- Product makes it accessible
- Different success metrics

---

#### 2. "Cost as Competitive Moat"

**Traditional Moat:**
```
Better algorithms → Performance lead → Moat
```

**MiniMax-M2 Moat:**
```
Better engineering → Cost lead → Price disruption → Market share
```

**Why Effective:**
- LLM performance converging (85-90% on most tasks)
- Cost becomes differentiator
- 8% price = 12.5x more value
- Developers are cost-sensitive

**Sustainability:**
```
Question: Can they maintain 8% price as models improve?
Answer: Yes, if efficiency focus continues
        - Quantization advances
        - Serving optimization
        - Architecture efficiency
```

---

#### 3. "Dogfooding > External Testing"

**M2 블로그:**
> "Developers across teams integrated M2 into daily workflows"

**Benefits:**
1. **Real pain points discovered**
   - Not hypothetical use cases
   - Actual developer friction

2. **Faster iteration**
   - Internal feedback immediate
   - No user study needed

3. **Better product-market fit**
   - Team is the target user
   - If they love it, others will too

**Lesson:**
- Use your own product intensively
- Internal users = Best beta testers
- Real usage > Benchmark optimization

---

#### 4. "1M Context Was Too Much"

**M1's 1M context:**
- ✅ Technical achievement
- ✅ Differentiation
- ❌ Overkill for most tasks
- ❌ Expensive to serve
- ❌ Slow inference

**M2's (추정) 256K-512K:**
- ✅ Practical for 90% use cases
- ✅ Cost-effective
- ✅ Faster inference
- ✅ Better UX

**General Lesson:**
```
More is not always better
Optimal > Maximum
Practical > Theoretical
```

**Applied to LLMs:**
- Context: 128K-512K sweet spot
- Parameters: Activated < Total (MoE)
- Precision: INT4-INT8 often sufficient
- Speed: 50-100 tokens/sec good enough

---

## 3️⃣ Kimi 시리즈: K2 (0905) → K2 Thinking

### 타임라인

```
Kimi K2 0905 (2025년 9월 5일?) → K2 Thinking (2025년 후반)
약 2-4개월 간격
```

### 전승 (Inheritance): Architecture 완전 유지

#### Base Model (100% Same)

| Component | K2 0905 | K2 Thinking | Status |
|-----------|---------|-------------|--------|
| **Total Params** | 1.04T | 1.04T | ✅ Same |
| **Activated Params** | 32B | 32B | ✅ Same |
| **MoE Experts** | 384 | 384 | ✅ Same |
| **Sparsity** | 48 | 48 | ✅ Same |
| **Attention Heads** | 64 | 64 | ✅ Same |
| **Optimizer** | MuonClip | MuonClip | ✅ Same |

**명확한 근거:**
- K2 Thinking = K2 0905 + Thinking mode
- "Built on" original K2
- Architecture changes 언급 없음
- Same model, different inference mode

---

#### Core Training (Inherited)

```
✅ Pre-training: 15.5T tokens (MuonClip)
✅ Knowledge rephrasing strategy
✅ Math → learning-note transformation
✅ PTX loss (NOT self-distillation)
✅ Self-critique mechanism
✅ Verifiable Rewards Gym
✅ Ultra-sparse MoE (384 experts)
```

**의미:**
- K2 Thinking은 new model이 아님
- K2 0905의 **inference mode extension**
- Post-training 추가 (thinking-specific)

---

### 변화 (Changes): Thinking Mode 추가

#### 1. Context Length 확장

**K2 0905:**
```
128K tokens (original)
→ 4K base → YaRN extension
```

**K2 Thinking:**
```
256K tokens (2x)
→ Longer reasoning chains
→ More tool call history
```

**필요성:**
```
Thinking tokens: 96K-128K per task
+
Tool outputs: Can be large
→ 128K insufficient
→ 256K needed
```

---

#### 2. Thinking Mode (핵심 혁신!)

**K2 0905: Reflex-Grade**
```
GitHub: "A reflex-grade model without long thinking"
→ Fast, direct responses
→ Minimal internal reasoning visible
→ Standard LLM behavior
```

**K2 Thinking: Extended Reasoning**
```
"Thinking mode" enabled
→ Internal reasoning visible (like O1)
→ Chain-of-thought generated
→ Step-by-step problem solving
```

**Thinking Token Budgets:**

| Task Type | Budget | Purpose |
|-----------|--------|---------|
| **HLE, AIME, HMMT, GPQA** | 96K tokens | Complex reasoning |
| **IMO, LiveCodeBench, OJ-Bench** | 128K tokens | Very hard problems |
| **Longform Writing** | 32K tokens | Creative generation |

**Example Thinking Process:**
```
User: "Solve this PhD-level math problem"

K2 Thinking:
[Thinking - 5K tokens]
Let me parse this problem. It involves hyperbolic space sampling...
The challenge is computing the log pdf at a specific point...
I need to search for relevant papers first...
[Search: "hyperbolic normal distribution pdf"]
[Thinking - 3K tokens]
Found Nagano et al. (2019) paper. The formula is...
But I need to verify with numerical computation...
[Python: matrix inverse calculation]
[Thinking - 2K tokens]
The pattern suggests a closed form. Let me test multiple values of n...
[Python: loop through n=3 to n=10]
[Thinking - 4K tokens]
Aha! The coefficient is -(2 - 1/n) for the k² term...
...
[Total: ~20K thinking tokens, 23 tool calls]
```

**vs K2 0905:**
```
User: "Solve this PhD-level math problem"

K2 0905:
Let me solve this step by step.
[Attempts direct solution, may fail on very hard problems]
[Maybe 1-2K tokens, few tool calls]
```

**Key Difference:**
- **K2 0905**: Direct problem solving
- **K2 Thinking**: Explicit reasoning process visible

---

#### 3. Dual Test-Time Scaling

**Traditional (O1-style):**
```
Test-time compute = More thinking tokens
→ Longer internal reasoning
→ Better performance
```

**K2 Thinking: Dual Scaling**
```
Test-time compute = Thinking tokens + Tool calling steps
                    ↓                  ↓
              Internal reasoning   External actions
```

**Concrete Numbers:**

**Thinking Tokens:**
- Per step: 24K-48K tokens
- Total: 96K-128K tokens
- Purpose: Internal reasoning

**Tool Calling Steps:**
- HLE: Max 120 steps
- Agentic search: Max 300 steps
- Purpose: External information/computation

**Interleaved Execution:**
```
Step 1: Think (24K tokens) → Tool call → Get result
Step 2: Think (24K tokens) → Tool call → Get result
...
Step 120: Think (24K tokens) → Final answer
```

**Total Compute:**
```
K2 0905: ~5K tokens per task
K2 Thinking: 96K-128K thinking + 300 tool calls
→ 20-30x more compute per task!
```

---

#### 4. Tool Integration Scale

**K2 0905:**
```
Tool calls: ~5-20 per task (typical)
Sequential: Yes, but limited
Autonomous: Partially
```

**K2 Thinking:**
```
Tool calls: 200-300 per task (max)
Sequential: Fully (no human interference)
Autonomous: Completely
```

**Example (PhD Math):**
```
23 interleaved reasoning & tool calls:
1. Think (parse problem)
2. Search (literature)
3. Think (analyze results)
4. Search (more specific)
5. Think (need formula)
6. Python (compute matrix)
7. Think (verify pattern)
8. Python (test multiple n)
9. Think (identify pattern)
10. Python (verify formula)
...
23. Think (final answer)
```

**Example (Agentic Search - Jimmy Gary Jr.):**
```
~30+ steps:
1. Think (parse constraints)
2. Search (prison drama corrections officer)
3. Think (narrow to OITNB)
4. Search (wrong inmate release)
5. Think (identify CO Sikowitz, but wrong person)
6. Search (football player actor OITNB)
7. Think (pivot to different approach)
8. Search (Jimmy Gary Jr. football)
9. Think (verify university founded 1867)
10. Search (Jimmy Gary Jr. alien invasion film)
...
Answer: Rudy Cox in "By Dawn"
```

**Autonomy Level:**
```
K2 0905: User → Model → Result (one shot)
K2 Thinking: User → Model explores 200-300 steps → Result
            (fully autonomous multi-step reasoning)
```

---

#### 5. QAT for Efficiency

**K2 0905:**
```
Precision: FP16/BF16 (standard)
Inference: Standard speed
Memory: Standard
```

**K2 Thinking:**
```
Precision: INT4 (QAT trained)
Inference: ~2x faster generation
Memory: ~4x less (INT4 vs FP16)
```

**Innovation:**

**Problem:**
```
Thinking models → Very long sequences (96K-128K)
Quantization → Usually big performance drop
INT4 on long sequences → Historically bad
```

**Solution:**
```
Quantization-Aware Training (QAT) during post-training
→ MoE components INT4
→ Train with quantization noise
→ Model adapts to INT4
```

**Result:**
```
✅ 2x generation speed
✅ State-of-the-art performance (not degraded!)
✅ All benchmarks reported in INT4
```

**Why This Matters:**

**Without QAT:**
```
FP16 K2: 91% AIME
INT4 K2: ~85% AIME (typical 5-7% drop)
→ Not acceptable
```

**With QAT:**
```
FP16 K2: 91% AIME (hypothetical)
INT4 K2: 99% AIME (K2 Thinking)
→ Actually BETTER (due to thinking mode)
→ No quantization penalty
```

**Technical Achievement:**
- First thinking model with native INT4 support
- No performance degradation
- 2x speedup bonus

---

#### 6. Heavy Mode (병렬 전략)

**K2 0905:**
```
Single trajectory: One reasoning path
Sequential: Generate → Answer
```

**K2 Thinking Standard:**
```
Single trajectory: One thinking path
But longer: 96K-128K tokens
```

**K2 Thinking Heavy Mode:**
```
8 parallel trajectories:
1. Path 1: Think differently → Answer 1
2. Path 2: Think differently → Answer 2
...
8. Path 8: Think differently → Answer 8

Reflective Aggregation:
→ Analyze all 8 answers
→ Synthesize best elements
→ Final Answer
```

**Performance Gain:**

| Benchmark | Standard | Heavy | Improvement |
|-----------|----------|-------|-------------|
| **HLE (with tools)** | 44.9% | 51.0% | +14% |
| **AIME 2025 (w/ python)** | 99.1% | 100.0% | +0.9% (perfect!) |
| **HMMT 2025 (w/ python)** | 95.1% | 97.5% | +2.4% |

**Cost:**
```
Standard: 1x compute
Heavy: 8x compute (parallel) + aggregation
→ ~8-9x total cost
```

**Use Case:**
```
Standard: Most tasks
Heavy: Critical tasks (exams, important decisions)
```

**Comparison:**
- GPT-5 Pro: Similar concept (heavy reasoning)
- K2 Thinking Heavy: 8 trajectories explicit

---

### 성능 변화 (Dramatic Improvements)

#### Math Reasoning

| Benchmark | K2 0905 | K2 Thinking | Improvement |
|-----------|---------|-------------|-------------|
| **AIME 2025 (no tools)** | 51.0% | 94.5% | +85% |
| **AIME 2025 (w/ python)** | 75.2% | 99.1% | +32% |
| **AIME 2025 Heavy** | - | 100.0% | Perfect! |
| **HMMT 2025 (no tools)** | 38.8% | 89.4% | +130% |
| **HMMT 2025 (w/ python)** | 70.4% | 95.1% | +35% |
| **HMMT 2025 Heavy** | - | 97.5% | Near-perfect |
| **IMO-AnswerBench** | 45.8% | 78.6% | +72% |
| **GPQA-Diamond** | 74.2% | 84.5% | +14% |

**분석:**
- Without tools: 2x-3x improvement
- With python: 30-35% improvement
- Heavy mode: Additional 10-20%

---

#### Agentic Reasoning

| Benchmark | K2 0905 | K2 Thinking | Improvement |
|-----------|---------|-------------|-------------|
| **HLE (text, no tools)** | 7.9% | 23.9% | +203% |
| **HLE (with tools)** | 21.7% | 44.9% | +107% |
| **HLE Heavy** | - | 51.0% | +135% |

**분석:**
- HLE는 extremely hard (expert-level, 100+ subjects)
- 21.7% → 44.9%는 massive jump
- Tool use essential (7.9% → 44.9%, 5.7x)

---

#### Agentic Search

| Benchmark | K2 0905 | K2 Thinking | Improvement |
|-----------|---------|-------------|-------------|
| **BrowseComp** | 7.4% | 60.2% | **+713%** |
| **BrowseComp-ZH** | 22.2% | 62.3% | +181% |
| **Seal-0** | 25.2% | 56.3% | +123% |
| **FinSearchComp-T3** | 10.4% | 47.4% | +356% |
| **Frames** | 58.1% | 87.0% | +50% |

**분석:**
- BrowseComp 7.4% → 60.2%는 **8배 improvement**
- Agentic search = K2 Thinking의 killer app
- 200-300 tool calls의 위력

---

#### Coding

| Benchmark | K2 0905 | K2 Thinking | Improvement |
|-----------|---------|-------------|-------------|
| **SWE-bench Verified** | 69.2% | 71.3% | +3% |
| **SWE-Multilingual** | 55.9% | 61.1% | +9% |
| **Multi-SWE-bench** | 33.5% | 41.9% | +25% |
| **LiveCodeBench V6** | 56.1% | 83.1% | +48% |
| **OJ-Bench (cpp)** | 25.5% | 48.7% | +91% |
| **SciCode** | 30.7% | 44.8% | +46% |

**분석:**
- SWE-bench: Modest improvement (이미 높았음)
- Competitive programming: Major gains (83.1% LiveCodeBench!)
- Hard coding: Big jumps (OJ-Bench 25→48%)

---

### 전략적 분석

#### Technology Stack

**K2 0905: Standard LLM**
```
Architecture: Ultra-sparse MoE (1.04T/32B)
Training: MuonClip, PTX loss, Self-critique
Inference: Standard (fast responses)
Focus: General agentic capabilities
```

**K2 Thinking: Reasoning LLM**
```
Architecture: Same (1.04T/32B)
Training: Same + Thinking-specific post-training
Inference: Extended (thinking mode)
Focus: Test-time compute scaling
```

#### Market Positioning

**K2 0905:**
```
Position: "Best open-source agentic model"
Target: General agentic applications
Competitors: DeepSeek-V3, GLM-4.5/4.6
Strength: SWE-bench 65.8%, TAU 66.1%
```

**K2 Thinking:**
```
Position: "Best open-source thinking model"
Target: Complex reasoning + Agentic tasks
Competitors: GPT-5 (High), Claude 4.5, O1
Strength: HLE 44.9% (beats GPT-5), BrowseComp 60.2%
```

#### Competitive Advantage

**vs GPT-5 (High):**
```
HLE (with tools): 44.9% vs 41.7% (Win ✅)
BrowseComp: 60.2% vs 54.9% (Win ✅)
SWE-bench V: 71.3% vs 74.9% (Close)
AIME: 99.1% vs 99.6% (Tie)
Status: Open-source vs Proprietary (Win ✅)
```

**vs Claude Sonnet 4.5 (Thinking):**
```
HLE: 44.9% vs 32.0% (Win ✅, +40%)
BrowseComp: 60.2% vs 24.1% (Win ✅, +150%)
SWE-bench: 71.3% vs 77.2% (Lose)
Longform Writing: 73.8% vs 79.8% (Lose)
```

**Conclusion:**
- Dominates agentic reasoning/search
- Competitive or better in most areas
- Open-source advantage

---

### 배울 점

#### 1. "Inference-Time Compute Scaling Works"

**Evidence:**

**AIME 2025:**
```
K2 0905 (no thinking): 51.0%
K2 Thinking (thinking): 94.5%
→ 85% absolute improvement!
```

**HLE:**
```
K2 0905: 21.7%
K2 Thinking: 44.9%
K2 Thinking Heavy (8 parallel): 51.0%
→ Each step adds value
```

**Lesson:**
- More compute at inference = Better performance
- Not just for proprietary models (O1)
- Open-source can do it too
- Thinking + Tool use = Multiplicative effect

---

#### 2. "Tool Calling at Scale Changes Everything"

**BrowseComp:**
```
K2 0905: 7.4% (limited tool use)
K2 Thinking: 60.2% (200-300 tool calls)
→ 8x improvement!
```

**Pattern:**
```
More tool calls = More information = Better reasoning
Not linear: Exponential quality improvement
```

**Why:**
- Complex problems need multiple information sources
- Dead ends require pivoting (search different terms)
- Verification needs cross-referencing
- Synthesis needs comprehensive data

**Limit:**
```
Optimal: 200-300 calls for hard problems
Beyond: Diminishing returns (likely)
```

---

#### 3. "QAT Solves Quantization for Long Sequences"

**Breakthrough:**
```
Traditional: Thinking models + Quantization = Bad
K2 Thinking: QAT → INT4 with no penalty
```

**Impact:**
```
Speed: 2x faster
Memory: 4x less
Cost: ~4x cheaper to serve
Performance: Maintained or improved
```

**Lesson:**
- Quantization-Aware Training works
- INT4 for production thinking models feasible
- Efficiency + Performance not trade-off

---

#### 4. "Base Model Quality Matters"

**K2 0905 was Already Good:**
```
SWE-bench: 69.2% (before thinking)
TAU-Bench: 66.1%
```

**With Thinking Mode:**
```
Modest improvements: 69.2% → 71.3% (SWE-bench)
Huge on hard tasks: 21.7% → 44.9% (HLE)
```

**Lesson:**
```
Thinking mode amplifies base capabilities
Garbage in → Garbage out (even with thinking)
Good base → Great with thinking
```

**Application:**
- Focus on strong base model first
- Then add thinking mode
- Not a substitute for good training

---

#### 5. "Open-Source Can Lead on Thinking Models"

**K2 Thinking vs Proprietary:**

| Benchmark | K2 Thinking (Open) | GPT-5 (Closed) | Winner |
|-----------|-------------------|----------------|---------|
| **HLE** | 44.9% | 41.7% | Open ✅ |
| **BrowseComp** | 60.2% | 54.9% | Open ✅ |
| **AIME** | 99.1% | 99.6% | Tie |

**Implications:**
1. **Democratization:**
   - Advanced reasoning accessible
   - Community can build on top
   - No API rate limits

2. **Trust:**
   - Reproducible results
   - Transparent methods
   - Community auditable

3. **Customization:**
   - Fine-tune for specific domains
   - Modify tool suite
   - Control deployment

**Lesson:**
- Open-source thinking models viable
- Not just cost-reduced versions
- Can match or beat proprietary

---

## 4️⃣ 종합 비교: 세 시리즈의 진화

### Evolution Patterns

| Series | Evolution Type | Time Gap | Focus Shift |
|--------|---------------|----------|-------------|
| **GLM** | Incremental | ~10-12 months | Balanced → Agentic |
| **MiniMax** | Radical | ~6-10 months | Research → Production |
| **Kimi** | Extension | ~2-4 months | Agentic → Thinking Agentic |

---

### Architecture Changes

| Series | Old → New | Magnitude | Type |
|--------|-----------|-----------|------|
| **GLM** | Same arch, extended context (128K→200K) | Small | Incremental |
| **MiniMax** | Likely simplified (1M→256K?) | Medium | Redesign |
| **Kimi** | Exactly same (1.04T/32B) | None | Pure inference |

---

### Strategic Pivots

#### GLM: Capability Expansion
```
4.5: Good at everything (balanced ARC)
     ↓
4.6: Great at agentic + tool use in inference
     Strategy: Add capabilities without losing strengths
```

#### MiniMax: Market Pivot
```
M1: Technical excellence (1M context, CISPO)
     ↓
M2: Market adoption (8% price, 2x speed, integrations)
     Strategy: Sacrifice technical extremes for practical value
```

#### Kimi: Vertical Integration
```
K2: Best agentic (SWE 65.8%, TAU 66.1%)
     ↓
K2 Thinking: Best thinking agentic (HLE 44.9%, Browse 60.2%)
     Strategy: Deepen existing strength with test-time compute
```

---

### Performance Evolution

#### Math (AIME)

| Model | V1 | V2 | Improvement | Method |
|-------|-----|-----|-------------|--------|
| **GLM** | 91.0% | ? (likely same) | 0% | Already SOTA |
| **MiniMax** | 86.0% | ? | ? | Unknown |
| **Kimi** | 75.2% (w/ py) | 99.1% (w/ py) | +32% | **Thinking mode** |

---

#### Agentic (SWE-bench, TAU)

| Model | V1 (SWE) | V2 (SWE) | Improvement | Method |
|-------|----------|----------|-------------|--------|
| **GLM** | 64.2% | ? | ? | Tool in inference |
| **MiniMax** | 56.0% | ? | ? | Unknown |
| **Kimi** | 69.2% | 71.3% | +3% | Modest (already high) |

---

#### Agentic Search (BrowseComp, new benchmark)

| Model | V1 | V2 | Improvement | Method |
|-------|-----|-----|-------------|--------|
| **GLM** | 26.4% | ? | ? | Unknown |
| **MiniMax** | ? | ? | ? | Unknown |
| **Kimi** | 7.4% | **60.2%** | **+713%** | 🚀 **200-300 tool calls** |

---

### Cost-Performance Trade-offs

| Model | V1 Cost | V2 Cost | V1 Performance | V2 Performance | Value |
|-------|---------|---------|----------------|----------------|-------|
| **GLM** | ? | ? | High | Higher | ? |
| **MiniMax** | High (1M ctx) | **8% Claude** | Good | "Close to best" | ⭐⭐⭐⭐⭐ Best |
| **Kimi** | Moderate | Moderate | High | **SOTA thinking** | ⭐⭐⭐⭐ Excellent |

---

### Innovation Type

#### GLM-4.6: Architectural Add-on
```
Type: Feature addition
Core: Context extension (128K→200K)
      Tool use in inference (NEW!)
Risk: Low (same architecture)
Impact: Medium (useful features)
```

#### MiniMax-M2: Product Transformation
```
Type: Market repositioning
Core: Cost optimization (8% price)
      Speed (2x faster)
      Integration (IDE tools)
Risk: Medium (may lose technical edge)
Impact: High (market disruption)
```

#### Kimi K2 Thinking: Paradigm Extension
```
Type: Inference mode innovation
Core: Test-time compute scaling
      200-300 tool calls
      QAT INT4
Risk: Low (same base model)
Impact: Very High (new capability class)
```

---

## 5️⃣ 공통 트렌드

### 1. Agentic Capabilities 중심

**모든 시리즈:**
- GLM-4.6: Tool use in inference
- MiniMax-M2: "Born for Agents and code"
- Kimi K2 Thinking: 200-300 tool calls

**의미:**
- Agentic = Next frontier
- Not just answering questions
- Autonomous multi-step problem solving

---

### 2. Real-World Integration

**공통 강조:**
- Claude Code
- Cursor
- Cline
- Kilo Code / Roo Code

**의미:**
- IDE integration = Standard
- Developer productivity = Killer use case
- Standalone API → Embedded tools

---

### 3. Efficiency Focus

**각 시리즈:**
- GLM-4.6: 15% fewer tokens
- MiniMax-M2: 8% Claude price, 2x speed
- Kimi K2 Thinking: INT4 QAT, 2x speed

**의미:**
- Performance plateau reached
- Now competing on efficiency
- Cost-performance = New differentiator

---

### 4. Context Length Convergence

**Evolution:**
```
M1: 1M (extreme)
    ↓
M2: ? (likely 256K-512K, practical)
    ↓
GLM-4.6: 200K (moderate increase)
K2 Thinking: 256K (2x from 128K)

Converging to: 200K-512K sweet spot
```

**Lesson:**
- 1M was exploration
- 128K-512K is practical
- Most tasks fit in 200K-300K

---

### 5. Test-Time Compute Scaling

**Only K2 Thinking explicitly:**
- Thinking tokens: 96K-128K
- Tool calls: 200-300
- Heavy mode: 8 parallel

**Implicit in others:**
- GLM-4.6: Tool use in inference (adaptive compute)
- MiniMax-M2: ? (not mentioned, but possible)

**Trend:**
- Pre-training compute → Fixed capability
- Test-time compute → Adaptive capability
- O1 started, K2 Thinking scaled

---

## 6️⃣ 미래 예측

### Short-term (3-6 months)

**Expected:**
1. **MiniMax-M3?**
   - Even cheaper? (5% Claude?)
   - Multimodal?

2. **GLM-4.7 or 5.0**
   - Multimodal likely
   - Even longer context? (256K-512K)

3. **K2 Thinking variants**
   - K2 Thinking Lite (fewer tool calls, cheaper)
   - K2 Thinking Pro (more tool calls, better)

---

### Mid-term (6-12 months)

**Trends:**
1. **Agentic Standardization**
   - MCP protocol widespread
   - Native IDE support everywhere
   - Multi-agent collaboration

2. **Cost Continues Falling**
   - $0.10-$0.30 per M tokens standard
   - Open-source quality improves
   - Quantization advances (INT2?)

3. **Test-Time Compute Mainstream**
   - All major models have "thinking mode"
   - Variable compute budgets
   - User-selectable depth

---

### Long-term (12-24 months)

**Possibilities:**
1. **Multimodal Agentic AI**
   - Vision + Code + Reasoning + Tool use
   - Full-stack development from screenshots
   - UI/UX generation

2. **Self-Improving Agents**
   - Online learning from experience
   - Continual adaptation
   - Personal customization

3. **Extreme Tool Integration**
   - 1000+ tool calls for months-long projects
   - Persistent agent memory
   - Multi-session reasoning

---

## 7️⃣ 최종 교훈

### For Researchers

**1. "Incremental > Radical (Usually)"**
- GLM-4.6: Same arch, major impact
- K2 Thinking: Same base, SOTA thinking
- Lesson: Don't redesign, refine

**2. "Test-Time Scaling is Real"**
- K2 Thinking: 7.4% → 60.2% BrowseComp
- More compute at inference works
- Open problem: Optimal allocation

**3. "QAT Enables Efficient Thinking"**
- INT4 without loss
- 2x speed
- Deployment feasible

---

### For Practitioners

**1. "Cost Disruption Coming"**
- MiniMax-M2: 8% Claude price
- Quality staying high
- Prepare for commoditization

**2. "Integration > Capability"**
- All models emphasize IDE tools
- Standalone API not enough
- Embedded experience matters

**3. "200K-512K Context is Sweet Spot"**
- 128K: Good
- 200K-512K: Better
- 1M: Overkill

---

### For Builders

**1. "Agentic is the Play"**
- All models doubling down
- Tool use, multi-step, autonomous
- Build for agents, not chat

**2. "Thinking Mode When Needed"**
- K2 Thinking for hard problems
- K2 0905 for quick tasks
- User choice optimal

**3. "Open-Source Competitive"**
- K2 Thinking beats GPT-5 (some tasks)
- No compromise on quality
- Deploy yourself advantage

---

**작성일**: 2025-11-10
**분석 대상**: 6 models across 3 series
**방법론**: Evidence-based comparison, Critical analysis
**Total Length**: ~30KB
