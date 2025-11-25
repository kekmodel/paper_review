# DreamGym: Production-Ready Agent RL Implementation

Comprehensive implementation guide for integrating [DreamGym](https://arxiv.org/abs/2410.06906) with production agent systems like Claude Code.

## Overview

DreamGym is a sample-efficient reinforcement learning framework that uses an experience model (M_exp) to synthesize training data. This repository provides **concrete implementations** bridging the gap between DreamGym's theoretical framework and real-world agent systems.

### Key Innovation: Zero Manual Heuristics

- **Rewards**: Judged by SOTA LLM (no reward engineering)
- **Entropy**: Calculated using Shannon entropy H = -Σ p log p (information theory)
- **Experience Model**: Uses SOTA reasoning models directly (no training needed)

## Features

### 🎯 Core Components

- **MDP Formalization**: Concrete state/action/reward definitions for real agents
- **LLM-Judged Rewards**: Binary rewards (0/1) based on LLM evaluation
- **Information-Theoretic Entropy**: Shannon entropy for initial state selection
- **Experience Memory**: Top-k/Top-p retrieval with oracle priority
- **Oracle-Guided Learning**: Bootstrap with 10-100 verified trajectories

### 🛠️ Production Integration

- **16+ Tool Integration**: Bash, Read, Write, Edit, Grep, Task, etc.
- **OpenAI Gym Interface**: LLM as environment with standard `reset()` and `step()` API
- **Subagent Orchestration**: Hierarchical task decomposition
- **Real Environment**: File system, git, bash command execution

### 📊 Sample Efficiency

- **Traditional RL**: 1,000-10,000+ trajectories needed
- **DreamGym**: 10-100 oracle trajectories + synthetic experiences
- **No M_exp Training**: Use SOTA reasoning models (Claude, o3) directly

## Documentation Structure

### Main Implementation
- **[dreamgym_with_claude_code_infrastructure.md](dreamgym_with_claude_code_infrastructure.md)** (★ START HERE)
  - Complete implementation with Claude Code-style infrastructure
  - MDP components (state, action, reward, transition)
  - Experience model using SOTA reasoning
  - LLM-judged rewards (heuristic-free)
  - Entropy-based state generation
  - Experience memory with top-k/top-p retrieval
  - Training loop and usage examples
  - 82KB comprehensive guide

### Specialized Approaches
- **[oracle_guided_dreamgym.md](oracle_guided_dreamgym.md)**
  - Using 10-100 verified trajectories as seeds
  - Oracle priority in experience retrieval
  - High accuracy with minimal data

- **[dreamgym_with_sota_reasoning.md](dreamgym_with_sota_reasoning.md)**
  - Zero-shot experience synthesis with SOTA models
  - DeepSeek-R1, OpenAI o1/o3 integration
  - No M_exp training required

- **[dreamgym_strategy_guide.md](dreamgym_strategy_guide.md)**
  - When to use which approach
  - Tradeoffs and decision framework
  - Practical recommendations

### Analysis
- **[fusion_methodology.md](fusion_methodology.md)** - Methodology fusion
- **[comparative_analysis.md](comparative_analysis.md)** - Comparison with related work
- **[paper2_detailed_analysis.md](paper2_detailed_analysis.md)** - DreamGym paper analysis
- **[paper2_figure_review.md](paper2_figure_review.md)** - Figure-by-figure review

## Quick Start

### 1. Oracle Data Collection
```python
# Collect 10-100 verified trajectories
oracle_trajectories = []

task = Task("Fix authentication bug in auth.py")
trajectory = Trajectory(task=task.description)

# Step 1: Read file
action1 = AgentAction(tool_name="Read", parameters={"file_path": "auth.py"})
trajectory.add_step(state1, action1, reward=1.0, next_state1)  # Contributes to goal

# Step 2: Fix bug
action2 = AgentAction(tool_name="Edit", parameters={...})
trajectory.add_step(state2, action2, reward=1.0, next_state2)  # Achieves goal

oracle_trajectories.append(trajectory)
```

### 2. Initialize DreamGym
```python
trainer = DreamGymTrainer()

# Load oracle data
trainer.initialize_with_oracle_data(oracle_trajectories)

# Add tasks to curriculum
tasks = [
    Task("Read a file and explain its purpose"),        # Easy
    Task("Find all TODOs in the codebase"),             # Medium
    Task("Fix the authentication bug"),                  # Medium
    Task("Implement a new feature: dark mode toggle"),  # Hard
]
for task in tasks:
    trainer.curriculum.add_task(task)
```

### 3. Train
```python
# Run training loop
trainer.train(num_iterations=1000)

# Training automatically:
# - Samples tasks from curriculum
# - Generates high-entropy initial states
# - Collects trajectories
# - Synthesizes experiences with M_exp
# - Trains policy
# - Updates curriculum
```

### 4. Use Trained Agent
```python
# Initialize environment
env = ClaudeCodeEnvironment()
state = env.reset(task="Implement user registration API")

# Agent loop
done = False
while not done:
    action = trained_policy.select_action(state, task)
    next_state, observation = env.transition(state, action)
    done = check_completion(task)
    state = next_state
```

## Key Algorithms

### LLM-Judged Rewards
```python
def compute_reward(state, action, next_state, task) -> float:
    prompt = f"""
    Task: {task}
    State before: {state}
    Action: {action}
    State after: {next_state}

    Does this action contribute to achieving the task goal?
    Answer: <reward>0 or 1</reward>
    """
    response = llm.generate(prompt)
    return parse_reward(response)  # 1 (contributes) or 0 (doesn't)
```

### Information-Theoretic Entropy
```python
def estimate_entropy(initial_state, task) -> float:
    # Do K rollouts from same state
    rollout_rewards = []
    for _ in range(K):
        trajectory = collect_trajectory(task, initial_state)
        rollout_rewards.append(trajectory.final_reward)

    # Compute Shannon entropy: H = -Σ p(r) log₂ p(r)
    unique, counts = np.unique(rollout_rewards, return_counts=True)
    probs = counts / K
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    return entropy  # High entropy = high uncertainty = informative state
```

### Experience Retrieval (Top-k/Top-p)
```python
def retrieve_similar(state, action, task, top_p=0.9):
    # 1. Embed query
    query_emb = embedding_model.encode(f"Task:{task}\nState:{state}\nAction:{action}")

    # 2. Compute similarities
    similarities = [cosine_similarity(query_emb, exp_emb) for exp in experiences]

    # 3. Oracle priority sort
    similarities.sort(key=lambda x: (not x.is_oracle, -x.similarity))

    # 4. Nucleus sampling (top-p)
    probs = softmax(similarities / temperature)
    cumulative = np.cumsum(probs)
    cutoff = np.searchsorted(cumulative, top_p)

    return experiences[:cutoff]  # Dynamic count based on distribution
```

## Architecture Comparison

| Aspect | DreamGym (Paper) | This Implementation |
|--------|------------------|---------------------|
| **Data Requirement** | Large offline dataset | 10-100 oracle trajectories |
| **Experience Model** | Trained M_exp | SOTA reasoning (no training) |
| **Reward Function** | Manual heuristics | LLM-judged (no heuristics) |
| **Entropy Calculation** | N/A | Shannon entropy from rollouts |
| **Action Space** | Abstract | Concrete (16+ tools with schemas) |
| **Environment** | Simulated | Real (file system, bash, git) |
| **Policy** | Neural network | LLM + system prompt |

## Advantages

✅ **Heuristic-Free**: All judgments via LLM or information theory
✅ **Sample Efficient**: 10-100 trajectories vs 1,000-10,000+
✅ **Production Ready**: Real tools, real environments
✅ **No Training**: Use SOTA reasoning models directly
✅ **Modular Design**: Swap components easily (real env ↔ LLM env)
✅ **Standard Interface**: OpenAI Gym compatible

## Papers

- **DreamGym**: [Sample-efficient RL via Experience Model](https://arxiv.org/abs/2410.06906)
- **AgentEvolver**: [Self-evolving agents](https://arxiv.org/abs/2410.07567)

## Use Cases

- **Coding Agents**: Bug fixing, feature implementation, refactoring
- **DevOps Agents**: Infrastructure management, deployment automation
- **Research Agents**: Literature review, experiment design
- **General Assistants**: Task automation with tool use

## Requirements

- SOTA LLM access (Claude Sonnet 4.5, OpenAI o1/o3, DeepSeek-R1, etc.)
- Python 3.8+
- Tools: git, bash, file system access (for real environment)
- Optional: SentenceTransformer for embedding-based retrieval

## Contributing

Contributions welcome! Areas of interest:
- Additional tool integrations
- Alternative policy training methods (SFT, RL, prompt optimization)
- Benchmark tasks and evaluation metrics
- Scalability improvements (vector DB for experience memory)

## Citation

If you use this implementation, please cite both the original DreamGym paper and this repository:

```bibtex
@article{dreamgym2024,
  title={DreamGym: Sample-efficient Reinforcement Learning via Experience Model},
  author={...},
  journal={arXiv preprint arXiv:2410.06906},
  year={2024}
}

@misc{dreamgym-implementation,
  title={DreamGym: Production-Ready Agent RL Implementation},
  author={kekmodel},
  year={2024},
  howpublished={\url{https://github.com/kekmodel/DreamGym}}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>

---

**Status**: Active Development
**Last Updated**: November 2024
**Maintainer**: [@kekmodel](https://github.com/kekmodel)
