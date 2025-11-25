# Paper Review

LLM 및 Agent RL 관련 논문 분석 저장소

## Structure

```
├── 01_make_llm/          # 중국 LLM 논문 기술 분석
│   ├── original_papers/  # GLM-4.5, MiniMax-M1, Kimi K2 분석
│   ├── technical_analysis/  # Post-training, Catastrophic forgetting, PTX Loss 등
│   └── new_versions/     # GLM-4.6, MiniMax-M2, Kimi K2 Thinking
│
└── 02_agent_rl/          # Agent RL 논문 리뷰
    ├── papers/           # 원본 PDF
    ├── 01_codex/         # Codex로 작성한 리뷰
    ├── 02_gemini/        # Gemini로 작성한 리뷰
    ├── 03_sonnet/        # Sonnet으로 작성한 리뷰 + DreamGym 구현 가이드
    └── 04_opus/          # Opus로 작성한 리뷰
```

## Papers

### LLM Architecture & Training
- **GLM-4.5/4.6** (Zhipu AI) - Multi-token prediction, Difficulty curriculum
- **MiniMax-M1/M2** - Hybrid linear attention, CISPO RL, 1M context
- **Kimi K2/K2 Thinking** (Moonshot AI) - Ultra-sparse MoE (384 experts), MuonClip

### Agent RL
- [DreamGym](https://arxiv.org/abs/2511.03773) - Scaling Agent Learning via Experience Synthesis
- [AgentEvolver](https://arxiv.org/abs/2511.10395) - Towards Efficient Self-Evolving Agent System

## Model Comparison

| Model | Total Params | Active Params | Context | Specialty |
|-------|--------------|---------------|---------|-----------|
| GLM-4.5 | 355B | 32B | 128K | Math (AIME 91.0%) |
| MiniMax-M1 | 456B | 45.9B | 1M | Long-context |
| Kimi K2 | 1.04T | 32B | 128K | Agentic (SWE-bench 65.8%) |

## Development

```bash
# Python 3.12+ with uv
source .venv/bin/activate
```
