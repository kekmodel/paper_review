# 월드클래스 모델의 이면: 숨겨진 고난과 실제 훈련의 현실

## 📋 개요

논문들은 "성공한 결과"만 보여줍니다:
- GLM-4.5: AIME 91.0%! ✨
- Kimi K2: 1.04T params, Ultra-sparse MoE! ✨
- MiniMax-M1: 1M context, 25% FLOPs! ✨

하지만 실제로는...
- 수개월의 디버깅 세션
- 예상치 못한 training restart
- 계속되는 loss spikes
- 알 수 없는 throughput degradation
- 50%+ ablation 비용

이 문서는 **Hugging Face의 Smol Training Playbook**에서 공유된 실제 훈련 고난들을 바탕으로, **GLM-4.5/4.6, MiniMax-M1/M2, Kimi K2/K2 Thinking** 같은 월드클래스 모델들이 겪었을 실제 문제들을 추론하고 분석합니다.

---

## 1️⃣ SmolLM3의 현실: 논문이 말하지 않는 것들

### SmolLM3 기본 스펙

```
Model: 3B parameters
Training: 11 trillion tokens
Infrastructure: 384 GPUs (H100)
Timeline: Several months
Team: Small (2-3 people focus)
```

### 실제로 겪은 문제들

#### Mystery 1: Vanishing Throughput
```
Initial: 400K tokens/sec (expected)
After hours: Dramatically degraded
Root cause: Tensor parallelism communication overhead
           Gradient accumulation interactions
Result: Hours of debugging
```

#### Mystery 2: Persistent Throughput Drops
```
Expected: 400K tokens/sec
Reality: 280K tokens/sec (30% loss!)
Root cause: Suboptimal GPU cluster interconnect config
           NVLink configuration issues
Result: Infrastructure tuning required
```

#### Mystery 3: Noisy Loss Curves
```
Symptom: Unexpected loss noise
Correlation: Checkpoint saving operations
Root cause: Asynchronous I/O interference
           Synchronization overhead
Solution: Batching checkpoint ops, faster storage
```

### 가장 충격적인 현실

**Unexpected Training Restart after 1 Trillion Tokens**

```
Progress: ~1T tokens trained (~10% of total)
Issue: Unresolved throughput/stability problems
Decision: RESTART from scratch
Lost compute: Months of GPU time
Lesson: Reserve 20-30% buffer for debugging
```

---

## 2️⃣ 월드클래스 모델들이 겪었을 고난 추론

### GLM-4.5: Deeper Layers + 96 Attention Heads

#### 논문이 말한 것
```
✅ 355B total params, 32B activated
✅ 96 attention heads (more than competitors)
✅ Deeper layers for better reasoning
✅ 23T tokens pre-training
```

#### 논문이 말하지 않은 것 (추론)

**Problem 1: Attention Head Communication Nightmare**

SmolLM3 (3B model):
```
Issue: Tensor parallelism communication overhead
Symptom: Throughput degradation over time
```

GLM-4.5 (355B, 96 heads):
```
Extrapolation:
- 100x larger model
- 96 attention heads (vs SmolLM's typical 32)
- → 3x more heads = 3x more communication

Likely problems:
- Massive all-to-all communication for attention
- NVLink bandwidth saturation
- Pipeline bubbles in 3D parallelism
- Gradient accumulation bugs at scale
```

**추정 impact:**
```
Throughput: Expected 500K tokens/sec
           Reality: Maybe 300K-350K tokens/sec (30-40% loss)

Debug time: Weeks to months finding optimal parallelism config
            Potentially multiple restarts
```

**Problem 2: Deeper Layers = Longer Pipeline**

```
Llama 3 70B: ~80 layers
GLM-4.5: "Deeper" → Likely 90-100+ layers

Pipeline Parallelism challenge:
- More layers = More pipeline stages
- More stages = More bubbles (idle time)
- Bubble overhead: 10-20% of compute wasted

Solution likely required:
- Interleaved pipeline scheduling (1F1B)
- Microbatching optimization
- Careful stage assignment
- → Weeks of tuning
```

**Problem 3: Memory Wall with 96 Heads**

```
SmolLM3 problem: KV-cache size
"MHA requires 20GB vs MQA 0.3GB (64x difference)"

GLM-4.5 with 96 heads:
- KV-cache per head: ~200MB per head at 128K context
- 96 heads = ~19GB KV-cache
- Activation memory: Additional 50-100GB
- Total memory: Easily exceeding single GPU (80GB H100)

Forced solutions:
- Tensor parallelism across GPUs (communication overhead)
- Activation checkpointing (recompute = slower)
- Mixed precision (BF16/FP8)
- → 20-30% throughput sacrifice
```

**Problem 4: 23T Tokens Training = Months of Debugging Opportunities**

```
SmolLM3: 11T tokens, multiple mysteries, 1 restart

GLM-4.5: 23T tokens (2x longer)
Expected issues:
- 2x more chances for loss spikes
- 2x more infrastructure failures
- Checkpoint corruption at scale
- Data pipeline bottlenecks
- Optimizer state explosion

Conservative estimate:
- 2-3 major incidents requiring restart/rollback
- Dozens of minor throughput degradations
- Hundreds of hours of monitoring/debugging
```

---

### MiniMax-M1: 1M Context + Hybrid Linear Attention

#### 논문이 말한 것
```
✅ 456B total, 45.9B activated
✅ 1M token context (8x DeepSeek R1)
✅ Hybrid linear + softmax attention
✅ CISPO RL algorithm
✅ 25% FLOPs of DeepSeek R1
```

#### 논문이 말하지 않은 것 (추론)

**Problem 1: 1M Context = Memory Apocalypse**

SmolLM3 context extension:
```
4K → 128K context (32x increase)
Required: Careful data curation, position interpolation
Issues: OOM errors, attention instability
```

MiniMax-M1:
```
128K → 1M context (8x SmolLM3's extension!)

Memory requirements (back-of-envelope):
- KV-cache at 1M: ~150GB per sample (even with GQA!)
- Activation memory: ~200GB
- Gradient checkpointing: Limited help (need full attention)

Forced architecture:
- Hybrid linear attention (essential, not optional)
- Multi-node tensor parallelism
- Ring attention or similar distributed attention
- → Development time: Months

Likely failures:
- Initial attempts with standard attention: OOM
- First linear attention impl: Numerical instability
- Communication overhead: Kills throughput
- Multiple architecture iterations before convergence
```

**Problem 2: Hybrid Attention = Custom CUDA Kernels**

```
Standard attention: Well-optimized (Flash Attention 2/3)

Linear attention:
❌ No standard implementations
❌ Need custom CUDA kernels
❌ Numerical stability issues
❌ Integration with existing frameworks difficult

Development reality:
- Months writing custom kernels
- Performance tuning (memory access patterns)
- Debugging numerical issues (NaN, Inf)
- Validating against standard attention
- Testing at multiple scales

Likely timeline:
- 3-6 months pure implementation
- 2-3 months debugging
- Multiple "back to drawing board" moments
```

**Problem 3: CISPO RL Algorithm = Novel Implementation**

SmolLM3 playbook warning:
> "No change escapes testing, even seemingly minor dependency updates"

MiniMax-M1:
```
CISPO: Novel RL algorithm (not in standard libraries)
vs PPO/GRPO: Well-tested, libraries available

Implementation challenges:
- Build from scratch (no reference impl)
- Importance sampling weights clipping (tricky math)
- Preventing token loss (preserving rare patterns)
- Hyperparameter tuning (no prior work)

Likely problems:
- Initial impl: Doesn't converge
- Clipping too aggressive: No learning
- Clipping too weak: Gradient explosion
- Reward signal design: Months of iteration

Timeline:
- Algorithm implementation: 1-2 months
- Debugging convergence: 2-3 months
- Hyperparameter search: 3-4 months
- Total: 6-9 months of RL tuning alone
```

**Problem 4: Data Curation for 1M Context**

SmolLM3 playbook insight:
> "Data obsession consistently outweighs architectural innovation"

MiniMax-M1:
```
1M context requires:
- Long documents (books, codebases, papers)
- Intra-document masking (prevent cross-doc attention)
- Careful mixture (single-doc vs multi-doc)

Challenges:
- Finding 1M-token documents (rare!)
- Cleaning long documents (more noise)
- Balancing with shorter data (most data <100K)
- Position interpolation artifacts

Data pipeline likely issues:
- Initial mixture: Models fail to use long context
- Too much long data: General performance degrades
- Masking bugs: Attention leaks across docs
- Position encoding extrapolation: Gibberish at 800K+

Iterations:
- 5-10 data mixture ablations
- Each: 100B tokens (days of training)
- Total: Months of experimentation
```

**Problem 5: 7.5T Tokens = Sustained Training Hell**

```
Training duration (estimated):
- 7.5T tokens
- 456B model (45.9B activated per token)
- GPUs: Likely 512-1024 H100s

Duration: 2-4 months continuous training

Inevitable issues over months:
- Hardware failures: 10-20 GPU failures expected
- Network issues: Multiple times per week
- Checkpoint corruption: At least once
- Data pipeline stalls: Regular occurrence
- Silent data corruption: Detected weeks later

Reality:
- 24/7 monitoring required
- On-call rotations
- Weekend debugging sessions
- Multiple "near-death" moments
- Probably 1-2 major rollbacks
```

---

### Kimi K2: 1.04T Parameters + 384 Experts

#### 논문이 말한 것
```
✅ 1.04 trillion parameters
✅ 32B activated (ultra-sparse, sparsity=48)
✅ 384 total experts (most ever!)
✅ MuonClip optimizer (novel)
✅ 15.5T tokens without loss spikes
```

#### 논문이 말하지 않은 것 (추론)

**Problem 1: 384 Experts = Routing Nightmare**

SmolLM3 doesn't use MoE, but we can extrapolate:

```
Standard MoE (e.g., Mixtral):
- 8 experts
- Simple routing (top-k)
- Load balancing challenges

Kimi K2:
- 384 experts (48x more!)
- Sparsity = 48 (activate 8 out of 384)
- Ultra-sparse routing

Routing challenges:
- Expert imbalance: Some experts unused, others overloaded
- Communication overhead: All-to-all expert routing
- Load balancing loss: Additional training objective
- Gradient synchronization: Nightmare at scale

Likely failures (early attempts):
- Routing collapses: All tokens route to same expert
- Training instability: Some experts never activated
- Communication bottleneck: Expert exchange kills throughput
- Loss auxiliary balancing: Too strong → Hurts performance
                            Too weak → Imbalance

Development timeline:
- Routing algorithm: 2-3 months
- Load balancing tuning: 2-3 months
- Infrastructure optimization: 3-4 months
- Total: 7-10 months before stable
```

**Problem 2: 1.04T Parameters = Storage & Checkpoint Hell**

```
Model size:
- 1.04T parameters × 2 bytes (BF16) = 2.08 TB
- Optimizer state (Adam): 2x = 4.16 TB
- Gradients: 2.08 TB
- Total: ~8-10 TB memory footprint

Checkpoint size:
- Model: 2.08 TB
- Optimizer: 4.16 TB
- Training state: ~500 GB
- Total per checkpoint: ~6.7 TB

Saving checkpoint (at 10 GB/s):
- Time: ~670 seconds = ~11 minutes
- Frequency: Every 100B tokens?
- Total saves: 155 times
- Total time waiting: 28 hours just saving!

Problems:
- I/O interference with training (SmolLM3's Mystery 3)
- Storage failures (TBs written per day)
- Checkpoint corruption (devastating at this scale)
- Recovery time: Hours to load 6.7 TB

Likely incidents:
- Corrupted checkpoint discovered 500B tokens later
- Rollback to previous checkpoint (lose days of training)
- Storage system crashes mid-save
- Network partition during distributed checkpoint
→ Multiple "heart attack" moments
```

**Problem 3: MuonClip Optimizer = Uncharted Territory**

SmolLM3 playbook:
> "Test everything systematically: No change escapes testing"

Kimi K2:
```
MuonClip: Novel optimizer (not in PyTorch/JAX)
vs Adam: Decade of battle-testing

Development challenges:
- Implementing QK-Clip mechanism
- Attention logit constraints
- Weight rescaling logic
- Integration with distributed training
- Memory overhead (additional state?)

Potential failures:
- Initial impl: Attention logits explode anyway
- Clipping threshold: Too tight → Poor performance
                     Too loose → Instability
- Numerical precision: BF16 vs FP32 for clipping
- Scale interaction: Works at 7B, breaks at 1T

Reality check:
- Paper says "15.5T tokens without loss spikes"
- But how many attempts before finding right config?
- How many spikes in early experiments?
- How many hyperparameters to tune?

Conservative estimate:
- 10-15 ablation runs (each 100B tokens)
- 5-10 "mystery spikes" requiring investigation
- 3-4 major hyperparameter sweeps
- → 6-12 months of optimizer tuning
```

**Problem 4: 15.5T Tokens = 1.5-2 Years Training Time**

```
Estimated training duration:
- 15.5T tokens
- 1.04T params (32B activated)
- GPUs: 1024-2048 H100s (guess)

Duration: 4-8 months continuous
         (or longer with restarts)

Over months/year:
- Hardware: 50+ GPU failures
- Software: 100+ library/driver bugs
- Data: Corruptions, pipeline stalls
- Infrastructure: Network partitions, storage issues
- Human: Burnout, on-call exhaustion

Invisible costs:
- 24/7 monitoring (team rotation)
- Weekend debugging (always something breaks)
- Alerting fatigue (false alarms)
- Opportunity cost (what else could be built?)

Paper: "15.5T tokens without loss spikes"
Reality: "15.5T tokens, countless spikes in dev,
         final run miraculously stable after
         months of tuning"
```

**Problem 5: Ultra-Sparse MoE = Debugging Blind Spots**

```
Problem: With 384 experts, debugging is hell

Token routes to Expert #247:
- Why that expert?
- What does Expert #247 specialize in?
- Is it working correctly?
- How to inspect 384 experts individually?

Debugging scenarios:
- Model outputs gibberish
  → Which expert failed?
  → Inspect all 384? (weeks of work)

- Certain languages degrade
  → Expert routing bias?
  → Load balancing issue?
  → Data mixture problem?

- Math performance drops
  → Math-expert never activated?
  → Routing policy broken?
  → Need per-expert analytics (infrastructure burden)

Tooling required:
- Expert activation heatmaps
- Per-expert loss tracking
- Routing pattern visualization
- Expert specialization analysis
→ Months building debugging infrastructure
```

---

### K2 Thinking: 200-300 Tool Calls + QAT INT4

#### 논문이 말한 것
```
✅ 200-300 sequential tool calls
✅ QAT INT4 (2x speed, no performance loss)
✅ Heavy Mode (8 parallel trajectories)
✅ HLE 44.9% (beats GPT-5)
```

#### 논문이 말하지 않은 것 (추론)

**Problem 1: 200-300 Tool Calls = Integration Hell**

SmolLM3 uses standard training loop.

K2 Thinking:
```
Tool ecosystem:
- Search API
- Python interpreter
- Web browsing
- File system (?)
- Database (?)

Integration challenges per tool:
- API rate limits
- Timeout handling
- Error recovery
- Output parsing
- State management

With 200-300 calls:
- Latency: 200 * 500ms = 100 seconds per task!
- Failures: 200 * 1% = 2 failures per task (expected)
- Error propagation: One bad call → Cascade failure
- Context overflow: Tool outputs fill 256K context

Early problems (likely):
- Tool calls hang indefinitely (timeout bugs)
- Parse errors from unexpected output format
- Context management: Outputs pushed out thinking tokens
- Rate limiting: API blocks after 50 calls
- Error cascades: Bad search → Bad code → Bad browse → Garbage

Development reality:
- 3-6 months building robust tool infrastructure
- Retry logic, fallback mechanisms
- Error handling for 200+ call sequences
- Context compression/summarization
- → Complex distributed system, not just LLM
```

**Problem 2: QAT INT4 at 1.04T Scale**

SmolLM3 playbook:
> "Quantization-Aware Training (QAT) during post-training"

K2 Thinking:
```
Challenge: QAT for trillion-parameter MoE

Standard QAT (small models):
- Inject quantization noise during training
- Learn to be robust to INT4
- Well-established for CNNs

At 1.04T scale:
- Quantization noise in 384 experts?
- Per-expert calibration needed?
- Routing affected by quantization?
- Memory overhead during QAT training?

Likely failures (initial attempts):
- INT4 routing: Completely broken
- Expert imbalance: Worse with quantization
- Some experts: Degrade 20-30%
- Overall: 5-10% performance loss

Fix attempts:
- Per-expert quantization ranges
- Mixed precision (some experts FP16)
- Quantization-aware routing
- Careful calibration data selection

Development timeline:
- QAT implementation: 2-3 months
- Debugging expert-specific issues: 2-3 months
- Performance recovery: 3-4 months
- Validation at scale: 1-2 months
- Total: 8-12 months

Paper claims: "2x speed, maintained performance"
Reality: After 8-12 months of hell,
         final run indeed achieves this
```

**Problem 3: Heavy Mode = 8x Compute Coordination**

```
Heavy Mode: 8 parallel trajectories

Implementation challenges:
- 8 separate inference streams
- Each: 200-300 tool calls
- Total: 1600-2400 tool calls per task!
- Synchronization at aggregation step

Problems:
- Load balancing: Some trajectories fast, others slow
- Resource contention: 8x GPU memory usage
- Tool API: 8x rate limit pressure
- Debugging: Which of 8 trajectories failed?

Aggregation complexity:
- Compare 8 different reasoning paths
- Different tools used, different intermediate results
- Some paths correct, some wrong, some gibberish
- How to synthesize?

Likely failures:
- Initial: Just averaging → Terrible results
- Attempt 2: Voting → Mediocre
- Attempt 3: LLM-based synthesis → Sometimes works
- Attempt 4: Rubric-based evaluation → Finally decent

Development:
- 2-3 months building parallel infrastructure
- 2-3 months tuning aggregation strategy
- 1-2 months optimization
- → 5-8 months for Heavy Mode alone
```

**Problem 4: Thinking Tokens = Data Creation Nightmare**

SmolLM3 playbook:
> "Data obsession consistently outweighs architectural innovation"

K2 Thinking:
```
Challenge: Create training data with thinking tokens

Where do thinking tokens come from?
- Can't use standard SFT data (no thinking)
- Can't manually annotate (96K tokens!)
- Must generate synthetically

Synthesis pipeline:
1. Use base model to generate thinking
   → But base model can't think well yet!

2. Use stronger model (GPT-4?) to generate
   → Expensive ($$$), quality varies

3. Use RL to learn thinking from rewards
   → Sparse rewards, hard exploration

Likely approach (multi-stage):
1. Bootstrap with GPT-4 examples (expensive)
2. Self-improvement: K2 generates, filter best
3. Iterative refinement: Multiple rounds
4. Quality control: Verify reasoning correctness

Data quality issues:
- Thinking too verbose: Wasted tokens
- Thinking too terse: Doesn't help
- Incorrect reasoning: Model learns bad habits
- Hallucinated steps: Looks good, actually wrong

Timeline:
- Pipeline development: 3-4 months
- Data generation: 2-3 months (compute-intensive)
- Quality filtering: 1-2 months
- Iteration: 2-3 rounds = 6-9 months total
- → Year-long data effort
```

---

## 3️⃣ 공통 고난: 모든 모델이 겪는 문제들

### Ablation Cost Reality

SmolLM3:
```
Main run: 276,480 GPU-hours
Ablations: 161,280 GPU-hours (58% of main!)
Total: 437,760 GPU-hours
```

**월드클래스 모델 extrapolation:**

GLM-4.5 (355B, 23T tokens):
```
Estimated main run: 2-3M GPU-hours
Ablations (58%): 1.2-1.7M GPU-hours
Total: 3.2-4.7M GPU-hours

At $2/GPU-hour (H100):
Main: $4-6M
Ablations: $2.4-3.4M
Total: $6.4-9.4M

But this assumes:
- No major failures (unrealistic)
- No restarts (unrealistic)
- Perfect efficiency (impossible)

Realistic total: $10-15M
```

MiniMax-M1 (456B, 7.5T, novel architecture):
```
Estimated main run: 1.5-2M GPU-hours
Ablations: Given novel architecture (linear attention, CISPO),
          likely 100-150% of main run
          → 1.5-3M GPU-hours

Total: 3-5M GPU-hours
Cost: $6-10M (they disclosed ~$534K for 3 weeks,
              full project much more)

Restarts/debugging: +50-100%
Realistic total: $10-20M
```

Kimi K2 (1.04T, 15.5T tokens, novel optimizer):
```
Estimated main run: 3-5M GPU-hours
Ablations: Ultra-sparse MoE (384 experts) + MuonClip
          Likely 150-200% of main run
          → 4.5-10M GPU-hours

Total: 7.5-15M GPU-hours
Cost: $15-30M

With debugging/restarts: $20-40M
```

**교훈:**
- 논문의 "final run" cost는 빙산의 일각
- Ablation + debugging + restarts = 2-3x main run cost
- 총 프로젝트 비용은 대부분 공개되지 않음

---

### Data Quality Paradoxes

SmolLM3 playbook insight:
> "Using what seems like highest quality data doesn't always yield stronger models. Training on arXiv underperforms diverse general text."

**월드클래스 모델 implications:**

**GLM-4.5 (AIME 91.0%):**
```
Intuition: More math papers → Better math
Reality: Likely tried pure arXiv/math corpus
Result: Probably worse than balanced mixture

Final mixture (guess):
- 60-70% General web (FineWeb-Edu style)
- 15-20% Math (curated, not raw arXiv)
- 10-15% Code
- 5-10% Science

Discovery process:
- Ablation 1: 50% arXiv → Worse general, no better math
- Ablation 2: 30% arXiv → Still suboptimal
- Ablation 3: 20% curated math → Better!
- Ablations 4-10: Fine-tuning proportions
→ 6-12 months finding right mixture
```

**Kimi K2 (Multilingual + Agentic):**
```
Challenge: Balance 6+ languages + code + reasoning

Problems likely encountered:
- Too much Chinese: English degrades
- Too much English: Chinese degrades
- Too much code: General language suffers
- Tool-use data: Hard to source, synthetic quality issues

Iterations:
- 10-15 mixture ablations
- Each: 100-200B tokens
- Total: 1-2T tokens just for ablations
- Timeline: 4-8 months of data experimentation

Final mixture (guess):
- English: 40%
- Chinese: 25%
- Other languages: 15%
- Code: 15%
- Tool-use: 5%
→ Months to arrive at this
```

---

### Training Instability: The 2am Debugging Sessions

SmolLM3 experienced:
- Throughput drops
- Loss noise
- Checkpoint I/O issues

**월드클래스 models at 100x scale:**

**Typical incident (extrapolated):**

```
3am, Week 8 of training (2T tokens done)

Alert: Loss spike detected
Engineer: Checks logs
          → Gradient norm: 1000x normal
          → Some layers: NaN gradients
          → Checkpoint: Corrupted

Investigation:
- Rollback to previous checkpoint (6 hours ago)
  → Lose 50B tokens ($10K wasted)
- Root cause:
  - Expert #247 (out of 384) had numerical issue
  - Specific token sequence triggered overflow
  - Cascaded through gradient accumulation
  - Corrupted optimizer state

Fix:
- Add gradient clipping (should have had from start!)
- Add per-expert gradient monitoring
- Add checkpoint validation
- Resume training

Time lost: 12 hours debugging + 6 hours re-training
Cost: $20K+ in compute
Morale: -50%

Similar incidents: 10-20 times during full training
```

**Real playbook quote:**
> "The playbook captures 'the 2am dataloader debugging sessions, the loss spikes, or the subtle tensor parallelism bug that quietly sabotages your training.'"

For 1.04T model over months:
- 50-100+ alerting incidents
- 10-20 major debugging sessions
- 3-5 "near-death" experiences
- 1-2 actual restarts
- Countless minor issues

---

## 4️⃣ Infrastructure: The Unsung Nightmare

### GPU Communication at Scale

SmolLM3 (384 GPUs):
```
Problem: Tensor parallelism communication overhead
Impact: 30% throughput loss
Fix: Infrastructure tuning (NVLink config)
```

**GLM-4.5 (355B, 96 attention heads):**

```
Estimated GPUs: 512-1024 H100s

Communication patterns:
- Attention: All-to-all across 96 heads
- MoE routing: (if used) All-to-all expert selection
- Gradient all-reduce: Synchronize across all GPUs
- Pipeline: Sequential layer dependencies

Bottlenecks:
- Intra-node: NVLink (900 GB/s) sufficient for 8 GPUs
- Inter-node: InfiniBand (400 Gbps = 50 GB/s)
  → 18x slower than NVLink!

When 96 heads split across nodes:
- Attention requires cross-node communication
- 50 GB/s shared across hundreds of operations
- Massive bottleneck

Likely reality:
- Initial config: 40% throughput of theoretical
- Tuning attempt 1: 50%
- Tuning attempt 2: 60%
- Tuning attempt 3: 70%
- Final config: 75% (best achievable)
- 25% of compute capability forever lost to communication

Timeline: 2-3 months finding "good enough" config
```

**Kimi K2 (1.04T, 384 experts):**

```
Estimated GPUs: 1024-2048 H100s

MoE communication nightmare:
- 384 experts distributed across GPUs
- Token routing: Need expert location lookup
- All-to-all expert exchange: Every token potentially
  needs different expert
- Load balancing: Some GPUs overloaded, others idle

Worst case:
- Token on GPU #1 needs Expert #247 on GPU #128
- Transfer: GPU #1 → Network → GPU #128
- Latency: 10-100 microseconds (vs 1 microsecond on-GPU)
- Throughput: 10-100x slower

At scale:
- Millions of tokens per batch
- Random expert routing
- Network saturation inevitable

Mitigations:
- Expert placement optimization (co-locate related experts)
- Batching expert calls (amortize communication)
- Hierarchical routing (local first, remote if needed)
- → 3-6 months developing + tuning
```

---

### Checkpoint Management Hell

SmolLM3:
```
Problem: Checkpoint I/O caused training interference
Solution: Asynchronous saving, faster storage
```

**Kimi K2 (6.7TB per checkpoint):**

```
Saving 6.7TB:
- Distributed across 1024 GPUs
- Each GPU saves ~6.5GB
- Parallel write to shared storage

Storage system requirements:
- Aggregate write: 500GB/s minimum
  (To save in ~13 seconds)
- IOPS: Millions (many small operations)
- Capacity: 100+ TB (keep 10-20 checkpoints)

Problems:
- Network filesystem: Slow (100 GB/s max)
  → 67 seconds per checkpoint
  → Training paused for 67 seconds!
  → 155 checkpoints = 2.9 hours wasted

- Distributed object storage: Complex
  → Coordination overhead
  → Consistency issues
  → Corruption risks

Incidents (likely):
- Checkpoint #47: Corrupted (network glitch)
- Checkpoint #89: Missing 200 experts (partial write)
- Checkpoint #120: Optimizer state wrong (race condition)

Each incident:
- Hours of debugging
- Rollback to previous checkpoint (lose tokens)
- Re-implement saving logic
- Add validation
→ Weeks of checkpoint engineering
```

---

## 5️⃣ Novel Algorithms: The Hidden Development Time

### SmolLM3 Baseline vs Novel Approaches

SmolLM3:
```
Architecture: Llama 3.2 (standard, proven)
Optimizer: AdamW (standard)
Training: Standard procedures

Development time: Mostly data + infrastructure
```

**월드클래스 models with novel components:**

### MiniMax-M1: CISPO RL

```
CISPO: Novel importance sampling weight clipping

Standard RL (PPO):
- PyTorch/TensorFlow implementations available
- Hyperparameters: Well-known ranges
- Debugging: Community knowledge

CISPO:
- No existing implementation
- Novel algorithm details (from paper only)
- Hyperparameters: Completely unknown
- Debugging: No community help

Development phases:

Phase 1: Implementation (2-3 months)
- Read paper 20+ times
- Implement basic algorithm
- Integrate with training loop
- Debug math errors

Phase 2: Making it work (3-4 months)
- Initial runs: Doesn't converge
- Tune clipping threshold: 10-20 attempts
- Fix numerical issues: Gradients explode/vanish
- Discover missing details from paper

Phase 3: Matching baselines (2-3 months)
- CISPO worse than PPO initially
- Find bugs in implementation
- Hyperparameter search: 50+ runs
- Finally competitive with PPO

Phase 4: Exceeding baselines (2-3 months)
- CISPO starts showing benefits
- Fine-tune for specific tasks
- Validate improvements real not noise

Total: 9-13 months for one novel algorithm
```

### Kimi K2: MuonClip Optimizer

```
MuonClip: Novel optimizer with QK-Clip

Standard (Adam):
- 1 week to integrate
- Well-known hyperparameters

MuonClip:
- Novel weight clipping mechanism
- Attention logit constraints
- Query/key projection rescaling

Development:

Phase 1: Understanding (1-2 months)
- Paper doesn't give full details
- Reverse-engineer from descriptions
- Math derivations
- Edge case analysis

Phase 2: Implementation (2-3 months)
- Build optimizer from scratch
- QK-Clip mechanism
- Integration with distributed training
- Checkpoint compatibility

Phase 3: Instability debugging (3-4 months)
- Loss spikes everywhere
- Attention logits: Still explode sometimes
- Clipping threshold: How to set?
- Per-layer? Per-head? Global?

Phase 4: Scaling (2-3 months)
- Works at 7B
- Breaks at 70B (different dynamics)
- Breaks at 1.04T (different again)
- Re-tune for each scale

Total: 8-12 months for optimizer alone

Paper claim: "15.5T tokens without loss spikes"
Reality: After year of optimizer engineering,
         yes, final run is stable
```

---

## 6️⃣ Post-Training: The Second Mountain

### SmolLM3 Post-Training

```
Phases:
1. SFT (supervised fine-tuning)
2. DPO (preference optimization)
3. Evaluation

Challenges (even with standard methods):
- Chat template design: Iteration required
- Dataset composition: Multiple attempts
- Hyperparameter tuning: Weeks
```

**월드클래스 models:**

### GLM-4.5: Expert Model Iteration

```
Paper: "Expert model iteration combining specialized
        experts in reasoning, agents, and general chat
        through self-distillation"

Reality (guessed):

Phase 1: Reasoning Expert (2-3 months)
- Curate reasoning data
- SFT on math/logic
- RL with verifiable rewards
- Test: Reasoning improved, but general degraded
- Fix: Add general data back
- Iterate: 5-10 attempts

Phase 2: Agentic Expert (2-3 months)
- Curate tool-use data (hard!)
- Synthetic data generation
- RL with execution feedback
- Test: Tool-use improved, reasoning degraded?
- Fix: Multi-task training
- Iterate: 5-10 attempts

Phase 3: General Expert (2-3 months)
- Balance all capabilities
- Prevent specialist collapse
- Test: Everything together
- Issue: Experts interfere
- Fix: Careful weighting
- Iterate: 5-10 attempts

Phase 4: Merging/Distillation (2-3 months)
- How to combine 3 experts?
- Ensemble? Merge? Distill?
- Test each approach: Months
- Final: Iterative distillation (not clear how!)

Total: 8-12 months post-training alone
```

### Kimi K2: Verifiable Rewards Gym

```
Paper: "Verifiable Rewards Gym" covering:
- Math/STEM (rule-based verification)
- Coding (sandbox testing)
- Instruction following (hybrid rules)
- Safety (adversarial prompts)

Building Verifiable Rewards Gym:

Math verification (2-3 months):
- Parse math expressions
- Symbolic verification
- Numerical verification
- Edge cases: Infinity, complex numbers, etc.

Code verification (3-4 months):
- Sandboxed execution (security!)
- Test case generation
- Timeout handling
- Resource limits (prevent DoS)

Instruction following (2-3 months):
- Rubric-based evaluation
- LLM-as-judge pipeline
- Human verification sampling
- Inter-annotator agreement

Safety (2-3 months):
- Adversarial prompt generation
- Red-teaming
- Attack-judge iteration
- Evolving threats

RL Training (3-4 months):
- Integrate all rewards
- Balance reward weights
- Prevent reward hacking
- Multi-objective optimization

Total: 12-17 months for comprehensive post-training

Paper: Shows final results
Reality: Year+ of infrastructure + training
```

---

## 7️⃣ 숨겨진 인적 비용

### Team Size & Burnout

SmolLM3:
```
Team: Small (2-3 people focus)
Duration: Several months
Benefit: Fast iteration

Cost: Burnout risk
```

**월드클래스 models (추정):**

```
GLM-4.5 (Zhipu AI & Tsinghua):
- Core team: 5-10 researchers
- Engineering: 10-20 engineers
- Infrastructure: 5-10 SRE/DevOps
- Data: 5-10 data engineers
- Total: 25-50 people-months

Work pattern:
- On-call rotations (24/7 monitoring)
- Weekend debugging (inevitable)
- Crunch before paper deadlines
- Pressure: Competing with DeepSeek, OpenAI

Human cost:
- Burnout: 2-3 people likely
- Turnover: 10-20% during project
- Relationships: Strained
- Health: Sleep deprivation, stress
```

**Kimi K2 (Moonshot AI):**
```
Project scope: 1.04T params, 15.5T tokens, novel optimizer

Estimated team:
- Research: 10-15
- Engineering: 20-30
- Infrastructure: 10-15
- Data: 10-15
- Total: 50-75 people over 12-18 months

Timeline: 12-18 months intensive

Human realities:
- 2am debugging: Weekly occurrence
- Failed experiments: Morale killer
- Restart decisions: Emotional toll
- Resource pressure: "Millions spent, where's result?"
- Competition: "DeepSeek just released X"

Invisible costs:
- Relationships: Families sacrificed
- Health: Chronic stress, burnout
- Career risk: "What if this fails?"
- Opportunity cost: Could have built 10 other things
```

---

### The Unwritten Rules

SmolLM3 playbook wisdom:
> "Reserve 20-30% buffer compute for unexpected debugging"

**월드클래스 팀들이 학습한 교훈 (추측):**

**Rule 1: Triple Your Timeline**
```
Initial estimate: 6 months
Realistic: 12-18 months

Why:
- Unexpected restarts
- Novel algorithm debugging
- Infrastructure issues
- Data quality iterations
- Post-training complexity
```

**Rule 2: Double Your Budget**
```
Initial budget: $10M
Realistic: $20-30M

Why:
- Ablations (50-100% of main cost)
- Debugging/restarts (30-50%)
- Human costs (often underestimated)
- Infrastructure overhead (20-30%)
```

**Rule 3: Plan for Failures**
```
Assumption: Smooth training
Reality:
- 10-20 major incidents
- 1-2 restarts
- 50-100 minor issues
- Countless surprises

Preparation:
- Checkpoint frequently
- Extensive monitoring
- Quick rollback procedures
- Post-mortem culture
```

**Rule 4: Data > Everything**
```
Temptation: Novel architecture!
Reality: "Data obsession outweighs architectural innovation"

Time allocation:
- 60% Data curation & ablations
- 20% Training & infrastructure
- 10% Algorithm development
- 10% Post-training
```

---

## 8️⃣ 결론: 논문 vs 현실

### 논문이 보여주는 것

```
✨ GLM-4.5: AIME 91.0%!
✨ MiniMax-M1: 1M context, 25% FLOPs!
✨ Kimi K2: 1.04T params, MuonClip optimizer!
✨ K2 Thinking: HLE 44.9%, beats GPT-5!

→ Clean, polished results
→ Novel contributions highlighted
→ Benchmarks impressive
```

### 현실

```
😓 GLM-4.5:
   - 6-12 months fighting 96 attention heads communication
   - Multiple restarts due to instability
   - $10-15M total spend (ablations + debugging)
   - Team burnout

😓 MiniMax-M1:
   - Year developing hybrid linear attention
   - 6-9 months getting CISPO to work
   - Countless OOM errors at 1M context
   - $10-20M total (much more than $534K disclosed)

😓 Kimi K2:
   - Year tuning 384-expert routing
   - 8-12 months making MuonClip stable
   - 6.7TB checkpoints causing constant headaches
   - $20-40M total investment
   - "15.5T tokens without spikes" after months of spikes

😓 K2 Thinking:
   - Year building tool integration infrastructure
   - 8-12 months QAT at trillion-parameter scale
   - Synthetic thinking data generation nightmare
   - Heavy Mode: 5-8 months parallel coordination

→ Messy, painful process
→ Failures hidden
→ True costs undisclosed
→ Human toll invisible
```

### 교훈

SmolLM3 playbook의 마지막 말:

> "This playbook doesn't give you a recipe. It gives you
> the messy reality behind training state-of-the-art
> language models."

**월드클래스 모델들에 적용하면:**

1. **성공은 빙산의 일각**
   - 보이는 것: 논문, 벤치마크, 코드
   - 보이지 않는 것: Failures, restarts, debugging, burnout

2. **진짜 비용은 3-5x 공식 추정치**
   - 논문: "3주, $534K"
   - 현실: 총 프로젝트 $10-20M

3. **Novel approaches = 6-12개월 추가 개발**
   - Standard: 빠른 시작
   - Novel (CISPO, MuonClip, 384 experts):
     Year+ of pain before working

4. **데이터가 90%, 알고리즘이 10%**
   - 대부분 시간: Data curation, mixture tuning
   - 논문 강조: Novel architecture/algorithm
   - 현실: "Data obsession" 승리

5. **성공한 모델 = 살아남은 모델**
   - 무수한 실패한 실험들
   - 운도 필요 (hardware failures, timing)
   - Perseverance가 핵심

---

## 9️⃣ 앞으로의 모델 개발자들에게

SmolLM3 playbook의 핵심 원칙들을 월드클래스 모델 개발에 적용하면:

### Principle 1: "Validate Before Investing"

```
Before training your 1T parameter model:
1. Can existing models do this? (Probably yes)
2. Will post-training work? (Likely yes)
3. Is training REALLY needed? (Probably no)

If still yes:
4. Start with 1B model ablations
5. Test every hypothesis systematically
6. Scale only when confident

Saves: 6-12 months, $5-10M
```

### Principle 2: "Data > Architecture"

```
Temptation: "Let's build 512-expert MoE!"
Reality: "Let's curate better data"

Impact:
- Novel architecture: +2-5% performance, 12 months
- Better data: +5-10% performance, 3 months

Winner: Data, every time
```

### Principle 3: "Reserve 50% Buffer"

```
Plan: 6 months, $10M
Reality: 12-18 months, $20-30M

Why 50% not 20-30%?
- Novel approaches always take longer
- Infrastructure issues unpredictable
- Data quality iterations underestimated
- Post-training more complex than expected

Better: Over-prepare, under-promise
```

### Principle 4: "Ablations Are Not Optional"

```
SmolLM3: 58% of compute on ablations
World-class: Should be 50-100%

Every decision needs ablation:
- Architecture choice
- Data mixture
- Hyperparameters
- Training schedule

Each ablation: 100B-500B tokens
Total: Months of compute

But: Only way to know what works
Alternative: Guessing (worse)
```

### Principle 5: "Monitor Everything"

```
Loss curve smooth ≠ Training healthy

Must monitor:
- Throughput (infrastructure health)
- Gradient norms (numerical stability)
- Per-layer metrics (localize issues)
- Checkpoint validity (corruption detection)
- Memory usage (leak detection)
- Communication overhead (bottleneck ID)

Tools needed:
- Custom dashboards
- Alerting systems
- Automatic anomaly detection
- Post-mortem automation

Development: 2-3 months
Value: Saves months of blind debugging
```

### Principle 6: "Team > Compute"

```
2-3 person team with:
- Fast iteration culture
- Systematic testing
- Data obsession
- 24/7 dedication

Beats:

20-person team with:
- Slow bureaucracy
- "Hero" runs without ablations
- Architecture obsession
- 9-to-5 mentality

Lesson: Iteration speed >> Team size
```

---

## 🔟 최종 메시지

논문을 읽을 때:
```
"GLM-4.5 achieves 91.0% on AIME"
```

실제로 생각해야 할 것:
```
"GLM-4.5 team spent 12-18 months,
 $10-15M, countless 2am debugging sessions,
 multiple restarts, team burnout,
 50+ ablations, 100+ incidents,
 to achieve 91.0% on AIME"
```

차이:
- 논문: 성공의 결과
- 현실: 성공의 과정

**가치:**
- 논문: 영감
- 현실: 교훈

**당신이 모델을 개발한다면:**
- 논문: 목표
- 현실: 지도

---

**작성일**: 2025-11-10
**기반**: SmolLM3 Training Playbook (Hugging Face)
**분석 대상**: GLM-4.5/4.6, MiniMax-M1/M2, Kimi K2/K2 Thinking
**방법**: Extrapolation from SmolLM3's documented reality to world-class model scale
**분량**: ~40KB

**핵심 메시지**: "Behind every impressive benchmark, there are months of debugging, millions of dollars, and countless sleepless nights. The papers show success; the reality is struggle."
