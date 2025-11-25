# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a research analysis repository for LLM and Agent RL papers. It contains:

1. **`01_make_llm/`**: Technical analysis of Chinese LLM papers (GLM-4.5/4.6, MiniMax-M1/M2, Kimi K2/K2 Thinking) covering architecture, training methodologies, and critical evaluation of marketing claims
2. **`02_agent_rl/`**: Agent RL paper reviews (DreamGym, AgentEvolver) with implementation guides for production agent systems

## Development Environment

```bash
# Python 3.12+ with uv package manager
source .venv/bin/activate
```

This is primarily a documentation repository with markdown files—no build commands or tests.

## Content Guidelines

### Language
- Main text: Korean
- Technical terms: Keep in English (MoE, PPO, catastrophic forgetting, etc.)
- Benchmark names and scores: English without translation

### Document Structure
When creating analysis documents, include:
- Clear title and core question being answered
- Tables for model/method comparisons
- Specific citations to paper sections or quantitative evidence
- Actionable conclusions

### Analysis Approach
The repository distinguishes between:
- **True innovations** (e.g., ultra-sparse MoE with 384 experts)
- **Standard techniques with new names** (e.g., "self-distillation" that's actually multi-task training)
- **Omitted details** (compute requirements, failure cases)

Evaluate feasibility for individual developers/small teams considering resource requirements, risks, and cost-benefit trade-offs.

## Key Reference Models

| Model | Total Params | Active Params | Context | Specialty |
|-------|--------------|---------------|---------|-----------|
| GLM-4.5 | 355B | 32B | 128K | Pure math (AIME 91.0%) |
| MiniMax-M1 | 456B | 45.9B | 1M | Long-context reasoning |
| Kimi K2 | 1.04T | 32B | 128K | Agentic tasks (SWE-bench 65.8%) |
