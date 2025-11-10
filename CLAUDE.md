# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research analysis repository documenting and comparing three major Chinese LLM paper releases from 2024-2025:

- **GLM-4.5/4.6** (Zhipu AI & Tsinghua University)
- **MiniMax-M1/M2** (MiniMax)
- **Kimi K2/K2 Thinking** (Moonshot AI)

The repository contains deep technical analysis written primarily in Korean, focusing on architecture innovations, training methodologies, performance comparisons, and critical evaluation of claims made in the papers.

## Repository Structure

The repository is organized into three main analytical categories:

### `original_papers/`
Contains comprehensive analysis of the original paper releases (GLM-4.5, MiniMax-M1, Kimi K2). The primary document `paper_analysis.md` (~24KB) provides:
- Detailed architecture breakdowns
- Training methodology comparisons
- Benchmark performance analysis
- Cross-model technical innovation mapping
- Critical evaluation sections

### `technical_analysis/`
Deep-dive technical investigations questioning and analyzing specific claims:

- **`post_training_feasibility.md`**: Analyzes whether individual developers can realistically replicate the post-training approaches described in papers. Examines resource requirements, catastrophic forgetting risks, and practical constraints.

- **`catastrophic_forgetting_reanalysis.md`**: Critical examination of catastrophic forgetting claims in multi-stage RL training pipelines. Questions whether it's truly a significant problem or manageable with standard techniques.

- **`self_distillation_reality_check.md`**: Debunks common misconceptions about "self-distillation" in these papers. Clarifies that GLM-4.5 uses iterative data replacement (bootstrapping), Kimi K2 uses PTX loss (auxiliary task), and MiniMax-M1 uses curriculum learning—none are true self-distillation.

- **`ptx_loss_explanation.md`**: Detailed explanation of PTX Loss (Pretraining Mix) from OpenAI's InstructGPT paper, including mathematical formulation and how it differs from self-distillation and experience replay.

### `new_versions/`
Covers 2025 model updates:

- **`new_versions_update.md`**: Comparative analysis of second-generation models (MiniMax-M2, GLM-4.6, Kimi K2 Thinking), focusing on what changed, market positioning, and practical implications.

- **`kimi_k2_thinking_detailed.md`**: In-depth analysis of Kimi K2 Thinking's test-time scaling approach, including dual scaling (thinking tokens + 200-300 sequential tool calls), QAT INT4 optimization, and Heavy Mode with 8 parallel trajectories.

## Development Environment

This is a Python 3.12+ project using `uv` for dependency management:

```bash
# Python version
python --version  # Should be 3.12+

# Virtual environment is located at .venv/
source .venv/bin/activate  # On macOS/Linux
```

The project has minimal dependencies (see `pyproject.toml`) and is primarily focused on markdown documentation rather than code.

## Key Analysis Themes

When working with or extending this repository, understand these recurring analytical frameworks:

### 1. Evidence-Based Criticism
Documents systematically cite paper claims, benchmark results, and cross-reference technical details. When adding analysis, always ground observations in specific paper sections or quantitative evidence.

### 2. Critical Perspective on Marketing Claims
The analyses distinguish between:
- **True innovations** (e.g., ultra-sparse MoE with 384 experts in Kimi K2)
- **Standard techniques with new names** (e.g., "self-distillation" that's actually multi-task training)
- **Omitted details** (e.g., compute requirements, failure cases)

### 3. Practical Implementation Focus
Technical analyses evaluate feasibility for individual developers or small teams, considering:
- Resource requirements (compute, data, expertise)
- Risk factors (catastrophic forgetting, reward hacking)
- Cost-benefit trade-offs
- Use case appropriateness

### 4. Cross-Model Pattern Recognition
The repository identifies shared innovations across papers:
- **Multi-stage post-training**: Pre → Mid/Continual → Post
- **RL-centric approaches**: PPO, GRPO, CISPO with verifiable rewards
- **Synthetic data generation**: Using larger models to create training data
- **Hybrid architectures**: Combining dense and sparse (MoE) components

## Model Comparison Quick Reference

| Model | Total Params | Active Params | Context | Key Innovation |
|-------|--------------|---------------|---------|----------------|
| GLM-4.5 | 355B | 32B | 128K | Multi-token prediction, Difficulty curriculum |
| GLM-4.6 | 355B | 32B | 200K | Tool use in inference, +56% context |
| MiniMax-M1 | 456B | 45.9B | 1M | Hybrid linear attention, CISPO RL |
| MiniMax-M2 | ? | ? | ? | Cost-optimized (8% of Claude price) |
| Kimi K2 | 1.04T | 32B | 128K | Ultra-sparse MoE (384 experts), MuonClip |
| K2 Thinking | 1.04T | 32B | 256K | Test-time scaling, 200-300 tool calls, QAT INT4 |

## Performance Specializations

- **GLM-4.5**: Pure math (AIME 91.0%), balanced general performance
- **GLM-4.6**: Extended context, tool integration, coding
- **MiniMax-M1**: Long-context (1M tokens), extended reasoning
- **MiniMax-M2**: Cost efficiency, developer tooling integration
- **Kimi K2**: Agentic tasks (TAU 66.1%, SWE-bench 65.8%)
- **K2 Thinking**: Agentic reasoning (HLE 44.9%, BrowseComp 60.2%)

## Content Language

All analysis documents are written in Korean with embedded English technical terms. When adding new content:
- Follow the existing Korean prose style for main text
- Keep technical terms in English (e.g., "MoE", "PPO", "catastrophic forgetting")
- Use consistent terminology with existing documents
- Include English benchmark names and scores without translation

## Recommended Reading Paths

**For General Overview:**
1. README.md → `original_papers/paper_analysis.md` → `new_versions/kimi_k2_thinking_detailed.md`

**For Implementation Guidance:**
1. `original_papers/paper_analysis.md` → `technical_analysis/post_training_feasibility.md` → `new_versions/new_versions_update.md`

**For Deep Technical Understanding:**
1. Original papers first → `technical_analysis/catastrophic_forgetting_reanalysis.md` → `technical_analysis/self_distillation_reality_check.md` → `technical_analysis/ptx_loss_explanation.md`

## Document Metadata

When creating new analysis documents, include:
- **Title**: Clear topic description
- **Core Question**: What specific question does this answer?
- **Key Sections**: Major analytical themes covered
- **Conclusions**: Actionable insights or position statements
- **File Size Estimate**: For tracking in README

All documents use standard markdown formatting with heavy use of:
- Tables for comparisons
- Bullet lists for feature breakdowns
- Code blocks for formulas and technical specifications
- Emoji section markers (📋, 1️⃣, 2️⃣, etc.)
