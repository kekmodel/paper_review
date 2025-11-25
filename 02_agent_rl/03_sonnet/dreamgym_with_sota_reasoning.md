# DreamGym + SOTA 추론 모델: Zero-Shot Experience Model 분석

## 핵심 질문

> DreamGym의 경험 모델(M_exp)을 오프라인 데이터로 학습시키는 대신, **현재 SOTA 추론 모델**(o1, o3, DeepSeek-R1, QwQ 등)을 **학습 없이 직접 활용**한다면?

---

## 1. 현재 DreamGym의 M_exp 학습 방식

### 1.1 학습 프로세스 (paper2_detailed_analysis.md 참조)

**Phase 1: 오프라인 데이터 수집**
- 공개 리더보드 데이터 또는 수동 수집
- 크기: 2K-10K 스텝 (Figure 5 참조)
- 예시: WebArena 리더보드 궤적

**Phase 2: SFT (Supervised Fine-Tuning)**
```python
# 학습 목표 (paper2:218-221)
Loss = - log P_θ(R_t | s_t, a_t, history, demonstrations, task)
     - log P_θ(s_{t+1} | R_t, s_t, a_t, ...)
     - log P_θ(r_{t+1} | R_t, s_t, a_t, ...)
```

**Phase 3: 경험 모델 사용**
- 백본: Llama-3.1-8B-Instruct
- 입력: 현재 상태, 행동, 히스토리, 과거 경험, 태스크
- 출력: CoT 추론 → 다음 상태 + 보상

### 1.2 학습의 필요성 (DreamGym 관점)

**1) 도메인 적응**:
- 특정 환경(WebArena, WebShop)의 동역학 학습
- 상태 표현 형식 적응
- 행동-결과 패턴 학습

**2) 데이터 효율성**:
- Figure 5: 2K-10K 샘플로도 경쟁력 있는 성능
- 작은 백본(3B-8B)도 충분

**3) 비용**:
- 학습 비용: 한 번만 (몇 시간)
- 추론 비용: 매 스텝마다 (수천~수만 번)

---

## 2. SOTA 추론 모델로 대체 시나리오

### 2.1 후보 모델

| 모델 | 추론 능력 | 비용 | 가용성 |
|------|-----------|------|--------|
| **OpenAI o1** | ⭐⭐⭐⭐⭐ | 매우 높음 | API |
| **OpenAI o3** | ⭐⭐⭐⭐⭐ | 극도로 높음 | 제한적 |
| **DeepSeek-R1** | ⭐⭐⭐⭐⭐ | 중간 | 오픈소스 |
| **Alibaba QwQ-32B** | ⭐⭐⭐⭐ | 중간 | 오픈소스 |
| **Google Gemini 2.0 Flash Thinking** | ⭐⭐⭐⭐ | 낮음 | API |

### 2.2 Zero-Shot 경험 모델 설계

**프롬프트 템플릿**:
```python
ZERO_SHOT_EXPERIENCE_MODEL_PROMPT = """
You are an experience model for a reinforcement learning agent.
Your task is to predict the next state and reward after the agent
takes an action, using chain-of-thought reasoning.

## Current State
{current_state}

Note: In multi-turn interactions, the state already contains necessary context
from previous turns (observations and actions).

## Agent Action
{action}

## Task Goal
{task}

## Similar Past Experiences (from replay buffer)
{similar_experiences}

## Your Task
1. **Reasoning** (<think>...</think>):
   - Analyze what happens when the agent takes this action
   - Consider the task goal and current state
   - Reference similar past experiences if helpful
   - Reason about the causal relationship

2. **Next State Prediction**:
   - Predict the resulting state in the same format
   - Include only action-relevant elements
   - Be concrete and specific

3. **Reward Prediction**:
   - Outcome-based: r=1 if task completed, r=0 otherwise

Generate the experience:
"""
```

**Zero-Shot 사용 방식**:
```python
class ZeroShotExperienceModel:
    def __init__(self, sota_model="deepseek-r1"):
        self.model = sota_model  # o1, o3, DeepSeek-R1, QwQ 등

    def predict(self, state, action, task, similar_exp):
        """
        Zero-shot 경험 예측

        Args:
            state: 현재 상태 (멀티턴의 경우 이미 t-1까지의 맥락 포함)
            action: 선택할 행동
            task: 태스크 명령어
            similar_exp: 리플레이 버퍼에서 검색한 유사 경험

        Note: 멀티턴 LLM RL에서 state는 Markov property를 만족하도록
              이미 필요한 히스토리를 포함합니다.
        """
        prompt = ZERO_SHOT_EXPERIENCE_MODEL_PROMPT.format(
            current_state=state,
            action=action,
            task=task,
            similar_experiences=similar_exp
        )

        # SOTA 모델에 직접 질의 (학습 없음)
        response = self.model.generate(
            prompt,
            temperature=0.7,
            thinking_budget="medium"  # o1/o3/R1 전용
        )

        # 추론 + 다음 상태 + 보상 파싱
        reasoning, next_state, reward = parse_response(response)

        return next_state, reward, reasoning
```

---

## 3. 장점 분석 ✅

### 3.1 오프라인 데이터 의존성 완전 제거 ⭐⭐⭐

**DreamGym 현재**:
- 2K-10K 오프라인 샘플 필요 (Figure 5)
- 새로운 도메인마다 데이터 수집 필요
- 데이터 품질이 M_exp 성능에 직접 영향

**SOTA 모델 사용 시**:
```
오프라인 데이터: 0개 ✅
새로운 도메인: 즉시 적용 가능 ✅
Cold-start 문제: 완전 해결 ✅
```

**의미**:
> **AgentEvolver와의 시너지 불필요**: 이미 zero-shot으로 모든 도메인 처리 가능

### 3.2 Superior 추론 능력 ⭐⭐⭐⭐⭐

**SOTA 추론 모델의 강점**:

**1) 깊이 있는 추론**:
```
Llama-3.1-8B (학습됨):
<think>
The agent clicks "Total Commits" button.
This should show commit history.
</think>

DeepSeek-R1 (zero-shot):
<think>
Given the task "find commits in April 2023", the agent
clicks "Total Commits" button. Let me reason:

1. Current page shows overview with latest commit
2. "Total Commits" button likely navigates to full history
3. The resulting page should have:
   - Chronological commit list
   - Date filters or groupings
   - April 2023 entries must be present
4. The agent will then need to:
   - Identify April entries
   - Click specific commit for details

Therefore, I predict the next state contains:
- Commit history interface
- Visible April 2023 entries
- Interactive elements for each commit
</think>
```

**차이**: DeepSeek-R1이 **훨씬 더 구조화되고 논리적인 추론** 제공

**2) 인과 관계 이해**:
- o1/o3/R1은 multi-step reasoning에 특화
- 행동 → 결과의 인과 체인 명확히 추론
- "왜 이 상태로 전이하는가" 설명 능력 탁월

**3) 일반화 능력**:
- 학습 데이터 없이도 다양한 도메인 이해
- WebArena → WebShop → ALFWorld 모두 처리 가능
- 도메인 불가지적 추론

### 3.3 최신 지식 활용 ⭐⭐⭐

**학습된 M_exp**:
- 오프라인 데이터의 지식에 제한
- 새로운 웹사이트 UI 변경 시 적응 어려움

**SOTA 모델**:
- 최신 사전학습 지식 (2024-2025)
- 다양한 웹 인터페이스 패턴 이해
- API, 툴 사용 패턴 내장

### 3.4 개발 속도 ⭐⭐⭐⭐

**기존 DreamGym 개발 시간**:
```
1. 오프라인 데이터 수집: 1-2주
2. 데이터 정제 및 포맷팅: 3-5일
3. M_exp 학습: 4-8시간
4. 품질 검증: 2-3일
───────────────────────────
총: 약 3주
```

**SOTA 모델 사용 시**:
```
1. 프롬프트 설계: 1-2일
2. API 통합: 1일
3. 테스트: 1일
───────────────────────────
총: 약 1주 (70% 단축) ✅
```

### 3.5 유지보수 간소화 ⭐⭐

**학습 기반**:
- 환경 변경 시 재학습 필요
- 데이터 드리프트 관리
- 버전 관리 복잡

**Zero-Shot**:
- 모델 API 업그레이드만으로 개선
- 프롬프트 수정으로 빠른 조정
- 버전 관리 단순

---

## 4. 단점 및 도전 과제 ❌

### 4.1 추론 비용 폭증 ⚠️⚠️⚠️

**DreamGym의 롤아웃 규모** (paper2:323-336):
```python
# 각 반복마다
for task in curriculum_tasks:  # 10-100개 태스크
    for rollout in range(50):  # 태스크당 50 롤아웃
        for step in range(50):  # 롤아웃당 50 스텝
            next_state, reward = M_exp.predict(...)

# 총 M_exp 호출 횟수 (1 iteration):
10 tasks × 50 rollouts × 50 steps = 25,000 호출

# 전체 학습 (100 iterations):
25,000 × 100 = 2,500,000 호출 🚨
```

**비용 비교** (1 iteration = 25,000 호출 기준):

| 모델 | 비용/호출 | 총 비용 (1 iter) | 100 iter 비용 |
|------|-----------|------------------|---------------|
| **Llama-3.1-8B (학습)** | $0.0001 | **$2.50** | **$250** |
| DeepSeek-R1 (API) | $0.001 | $25 | $2,500 |
| OpenAI o1-preview | $0.015 | $375 | **$37,500** 🚨 |
| OpenAI o3 (high) | $0.10+ | $2,500+ | **$250,000+** 💥 |

**결론**: o1/o3는 **비현실적으로 비쌈**

**가능한 모델**:
- ✅ DeepSeek-R1: 10배 비용 증가 (감당 가능)
- ✅ QwQ-32B (오픈소스): 자체 호스팅 시 비용 감소
- ⚠️ Gemini 2.0 Flash Thinking: 저렴하지만 추론 능력 검증 필요

### 4.2 추론 속도 병목 ⚠️⚠️

**학습된 M_exp (Llama-3.1-8B)**:
- 추론 시간: ~0.5초/호출
- 병렬화: 배치 크기 32-64
- 총 시간 (25,000 호출): ~4시간

**SOTA 추론 모델**:
- **o1/o3**: 10-60초/호출 (thinking 시간)
- **DeepSeek-R1**: 5-15초/호출
- **병렬화**: API 제한 (RPM: 50-100)

**시간 비교** (25,000 호출):

| 모델 | 추론 시간 | 병렬화 | 총 시간 |
|------|-----------|--------|---------|
| Llama-3.1-8B | 0.5초 | 64 배치 | **4시간** |
| DeepSeek-R1 | 10초 | 50 RPM | **83시간** 🐌 |
| o1-preview | 30초 | 20 RPM | **~350시간** 💀 |

**해결책**:
1. **병렬 API 호출**: 여러 계정/키 사용
2. **자체 호스팅**: DeepSeek-R1, QwQ-32B (오픈소스)
3. **하이브리드**: 일부만 SOTA 모델 사용

### 4.3 도메인 특화 부족 ⚠️

**학습된 M_exp**:
- 특정 환경의 세부 동역학 학습
- 상태 포맷에 정확히 맞춤
- 빈번한 패턴 최적화

**Zero-Shot SOTA 모델**:
- 일반적 추론은 우수
- 세부 동역학 모를 수 있음
- 예: WebArena의 특정 버튼 ID 패턴

**예시**:
```python
# 학습된 M_exp (WebArena 특화):
# Input: Click([463])
# Output: [1144] listitem: Commit on Apr 2, 2023...
#         [1259] listitem: Commit on Apr 8, 2023...
# ✅ 정확한 element ID 생성

# Zero-Shot R1:
# Input: Click([463])
# Output: [001] listitem: First commit entry
#         [002] listitem: Second commit entry
# ⚠️ Element ID가 실제와 다를 수 있음
```

**완화 방안**:
- Few-shot 예제 제공 (프롬프트에 2-3개 샘플 포함)
- 리플레이 버퍼의 유사 경험 활용

### 4.4 일관성 vs 다양성 제어 어려움 ⚠️

**학습 기반**:
- Temperature, Top-p 등으로 섬세한 제어
- Figure 4: Consistency 1.3, Diversity 1.5 달성

**Zero-Shot**:
- SOTA 모델의 내부 샘플링 제어 제한적
- o1/o3는 temperature 조절 불가 (고정)
- Consistency-Diversity 균형 맞추기 어려움

### 4.5 Hallucination 위험 증가 가능 ⚠️

**Theorem 1의 오차 항** (paper2:368-380):
$$
\epsilon_R = \sup_{s,a} |R(s,a) - \hat{R}(s,a)|
$$
$$
\epsilon_P = \sup_{s,a} \| P(\cdot | s,a) - \hat{P}(\cdot | s,a) \|_1
$$

**Zero-Shot 모델**:
- 학습 데이터 없으면 $\epsilon_R$, $\epsilon_P$ 클 수 있음
- 특히 $\epsilon_P$ (도메인 일관성) 문제
- Hallucination → RL 학습 불안정

**검증 필요**:
- Figure 4 스타일의 Judge 평가
- 실제 환경과 비교하여 정확도 측정

---

## 5. 하이브리드 접근법 🌟

### 5.1 DreamGym + SOTA 추론 모델의 최적 결합

**전략**: **Selective High-Quality Experience Generation**

```python
class HybridExperienceModel:
    def __init__(self):
        # 빠르고 저렴한 기본 모델
        self.base_model = FineTunedLlama8B()  # 학습됨

        # 강력하지만 비싼 SOTA 모델
        self.sota_model = DeepSeekR1()  # Zero-shot

        # 품질 판정 모델
        self.judge = GPT4o()

    def predict(self, state, action, task, similar_exp):
        """
        Hybrid 경험 예측 (base + SOTA)

        Note: state는 이미 멀티턴 맥락을 포함
        """
        # Phase 1: 기본 모델로 빠르게 생성
        pred_base = self.base_model.predict(state, action, task, similar_exp)

        # Phase 2: 중요한 상황에서만 SOTA 모델 사용
        if self.is_critical_state(state, task):
            pred_sota = self.sota_model.predict(state, action, task, similar_exp)

            # Phase 3: 품질 비교 후 선택
            quality_base = self.judge.evaluate(pred_base)
            quality_sota = self.judge.evaluate(pred_sota)

            return pred_sota if quality_sota > quality_base else pred_base
        else:
            return pred_base

    def is_critical_state(self, state, task):
        """
        중요한 상황 판단:
        - 태스크 성공/실패의 분기점
        - 복잡한 의사결정 필요
        - Base 모델이 불확실성 높음
        """
        uncertainty = self.base_model.uncertainty(state)
        is_near_goal = self.estimate_distance_to_goal(state, task)

        return uncertainty > 0.8 or is_near_goal < 3  # 3스텝 이내
```

**비용 분석** (25,000 호출 기준):
```
Base model (90%): 22,500 × $0.0001 = $2.25
SOTA model (10%): 2,500 × $0.001 = $2.50
Judge (10%): 2,500 × $0.0005 = $1.25
────────────────────────────────────────
Total: $6.00 (vs Base only $2.50, vs SOTA only $25)
```

**성능 기대**:
- 중요한 10% 상황에서 SOTA 모델의 우수한 추론 활용
- 나머지 90%는 빠른 Base 모델 사용
- **비용 2.4배 증가로 핵심 품질 향상**

### 5.2 Curriculum-Aware SOTA Usage

```python
class CurriculumAwareHybrid:
    """
    커리큘럼 학습 단계에 따라 SOTA 모델 사용 비율 조절
    """
    def __init__(self):
        self.base_model = FineTunedLlama8B()
        self.sota_model = DeepSeekR1()
        self.curriculum_stage = 0  # 0: 초기, 1: 중기, 2: 후기

    def predict(self, state, action, task, similar_exp):
        """
        커리큘럼 인식 Hybrid 예측

        Note: state는 이미 멀티턴 맥락을 포함
        """
        # 초기 학습: 많은 SOTA 사용 (30%)
        # 중기 학습: 중간 (15%)
        # 후기 학습: 적게 (5%)

        sota_prob = {0: 0.30, 1: 0.15, 2: 0.05}[self.curriculum_stage]

        if random() < sota_prob:
            return self.sota_model.predict(...)
        else:
            return self.base_model.predict(...)

    def update_stage(self, success_rate):
        """성공률 기반 단계 업데이트"""
        if success_rate > 0.7:
            self.curriculum_stage = 2  # 후기
        elif success_rate > 0.4:
            self.curriculum_stage = 1  # 중기
        else:
            self.curriculum_stage = 0  # 초기
```

**논리**:
- **초기**: 어려운 태스크, SOTA 모델의 강력한 추론 필요
- **후기**: 에이전트가 이미 잘 학습됨, Base 모델로 충분

### 5.3 Self-Improving Experience Model

```python
class SelfImprovingExperienceModel:
    """
    SOTA 모델로 고품질 샘플 생성 → Base 모델 지속 학습
    """
    def __init__(self):
        self.base_model = FineTunedLlama8B()
        self.sota_model = DeepSeekR1()
        self.high_quality_buffer = []

    def collect_training_data(self, num_samples=1000):
        """SOTA 모델로 고품질 학습 데이터 생성"""
        for _ in range(num_samples):
            state, action, task = sample_from_curriculum()

            # SOTA 모델로 고품질 경험 생성
            next_state, reward, reasoning = self.sota_model.predict(
                state, action, task
            )

            # 품질 검증 (Judge)
            quality = GPT4o().evaluate(next_state, reasoning)

            if quality > 1.5:  # High quality
                self.high_quality_buffer.append({
                    'input': (state, action, task),
                    'output': (next_state, reward, reasoning)
                })

    def update_base_model(self):
        """고품질 데이터로 Base 모델 지속 학습"""
        self.base_model.finetune(
            data=self.high_quality_buffer,
            epochs=3,
            learning_rate=1e-5
        )

        # 버퍼 비우고 다음 라운드 준비
        self.high_quality_buffer = []
```

**워크플로우**:
```
1. 초기: Base 모델로 빠르게 학습 시작
2. 매 10 iterations마다:
   a) SOTA 모델로 1K 고품질 샘플 생성
   b) Base 모델 파인튜닝
3. Base 모델 점진적 개선 → SOTA 의존도 감소
```

**장점**:
- SOTA 모델을 "teacher"로 활용
- Base 모델이 점점 좋아짐
- 장기적으로 비용 감소

---

## 6. 실험 설계: 검증 방법

### 6.1 비교 실험

**3가지 변형 비교**:

| 변형 | M_exp | 오프라인 데이터 | 비용 (100 iter) |
|------|-------|----------------|-----------------|
| **Baseline** | Llama-3.1-8B (학습) | 10K | $250 + 데이터 비용 |
| **Zero-Shot** | DeepSeek-R1 | 0 | $2,500 |
| **Hybrid** | Base + R1 (10%) | 1K | $600 |

**평가 지표**:
1. **최종 성능**: WebArena/WebShop Success Rate
2. **학습 효율성**: Steps to 50% success rate
3. **경험 품질**: Figure 4 스타일 Judge 평가
4. **비용 효율성**: Cost per 1% performance gain

### 6.2 예상 결과

**가설 1: Zero-Shot이 초기에 우수**
```
Steps 0-30:
- Baseline: 50% (학습된 도메인 지식)
- Zero-Shot: 60% (강력한 일반 추론) ⭐
- Hybrid: 65% (둘의 장점 결합) ⭐⭐
```

**가설 2: 최종 성능은 비슷**
```
Steps 100-150:
- Baseline: 60%
- Zero-Shot: 62%
- Hybrid: 65% ⭐
```

**가설 3: 비용 효율성은 Hybrid 최고**
```
Cost per 1% gain (vs Baseline 60%):
- Zero-Shot: $2,500 / 2% = $1,250/% ❌
- Hybrid: $600 / 5% = $120/% ✅
```

### 6.3 핵심 검증 질문

**Q1: Theorem 1의 오차 항은?**
- Zero-Shot R1의 $\epsilon_R$, $\epsilon_P$ 측정
- 학습된 모델과 비교

**Q2: Figure 4 품질 지표는?**
- Consistency, Diversity, Informativeness, Hallucination
- Zero-Shot이 더 높은 Informativeness 보일까?

**Q3: 새로운 도메인 적응 속도는?**
- Baseline: 2-3주 (데이터 수집 + 학습)
- Zero-Shot: 즉시
- 이 차이가 성능을 상쇄할 만큼 가치 있는가?

---

## 7. 실용적 권장사항

### 7.1 언제 Zero-Shot SOTA 모델을 사용할까?

**✅ 추천 시나리오**:

**1) 새로운 도메인 빠른 프로토타이핑**
```
목표: 1주 내에 DreamGym 검증
방법: DeepSeek-R1 zero-shot
비용: ~$3,000 (감당 가능)
이점: 데이터 수집 없이 즉시 시작
```

**2) 오프라인 데이터 수집 불가능한 환경**
```
예시:
- 비공개 기업 내부 시스템
- 실시간 변화하는 환경
- 프라이버시 제약

해결책: Zero-shot만이 유일한 옵션
```

**3) 소규모 연구 프로젝트**
```
롤아웃 수: 1,000-5,000 (적음)
비용: $10-25 (DeepSeek-R1)
이점: 학습 인프라 불필요
```

**4) Hybrid 접근 (Best Practice)**
```
초기: DeepSeek-R1으로 1K 고품질 샘플 생성 ($1)
중기: 이 데이터로 Base 모델 학습 ($50)
후기: Base 모델 주로 사용, 필요시 R1 호출
총 비용: ~$300 (vs Baseline $250 + 데이터, vs Zero-Shot $2,500)
```

### 7.2 언제 학습 기반 M_exp가 나은가?

**✅ 기존 DreamGym 추천 시나리오**:

**1) 대규모 프로덕션 환경**
```
롤아웃 수: 100,000+
학습 비용: $100 (1회)
추론 비용: $50 (Base 모델, 100K 호출)
총: $150

vs Zero-Shot: $100,000 (DeepSeek-R1, 100K 호출) 💀
```

**2) 장기 운영 시스템**
```
사용 기간: 1년+
초기 투자: 데이터 수집 + 학습 ($500)
운영 비용: 매우 낮음 ($10/month)
총: $620

vs Zero-Shot: $30,000+ (지속적 API 비용)
```

**3) 특정 도메인 극한 최적화**
```
목표: WebArena 15% → 18% (한계 성능)
방법: 도메인 특화 M_exp + 대규모 데이터
결과: 최고 성능 달성

Zero-Shot: 일반화 능력은 좋으나 극한 최적화 어려움
```

### 7.3 최종 결정 트리

```
START: DreamGym M_exp 선택
│
├─ 새로운 도메인인가?
│  ├─ YES: 오프라인 데이터 있는가?
│  │  ├─ YES → 학습 기반 M_exp ✅
│  │  └─ NO → Zero-Shot 또는 Hybrid ✅
│  │
│  └─ NO (기존 도메인): 프로덕션인가?
│     ├─ YES → 학습 기반 M_exp ✅
│     └─ NO (연구) → Zero-Shot ✅
│
├─ 예산 제약은?
│  ├─ 매우 제한적 (<$500) → Hybrid ✅
│  ├─ 중간 ($500-$5K) → Zero-Shot 가능 ✅
│  └─ 충분 (>$5K) → 최적 방법 자유 선택
│
└─ 시간 제약은?
   ├─ 매우 급함 (<1주) → Zero-Shot ✅
   ├─ 중간 (1-4주) → Hybrid ✅
   └─ 여유 (>1달) → 학습 기반 (최고 성능) ✅
```

---

## 8. 미래 방향: SOTA 추론 + DreamGym 융합

### 8.1 이상적 시나리오

**"Progressive Experience Model"**:

```python
class ProgressiveExperienceModel:
    """
    Stage 1: DeepSeek-R1 zero-shot (즉시 시작)
    Stage 2: R1이 생성한 데이터로 Base 학습 (1주 후)
    Stage 3: Base + R1 hybrid (2주 후)
    Stage 4: Base 단독 (4주 후, 비용 최소화)
    """
    def __init__(self):
        self.stage = 1
        self.sota = DeepSeekR1()
        self.base = None
        self.collected_data = []

    def predict(self, *args):
        if self.stage == 1:
            # Pure SOTA
            result = self.sota.predict(*args)
            self.collected_data.append((*args, result))

            # 1K 샘플 수집 후 Stage 2로
            if len(self.collected_data) >= 1000:
                self.advance_to_stage_2()

            return result

        elif self.stage == 2:
            # Base 학습 중, SOTA 계속 사용
            return self.sota.predict(*args)

        elif self.stage == 3:
            # Hybrid
            if self.is_critical(*args):
                return self.sota.predict(*args)
            else:
                return self.base.predict(*args)

        else:  # stage == 4
            # Base 단독 (가끔 R1로 검증)
            return self.base.predict(*args)

    def advance_to_stage_2(self):
        """Stage 1 → 2: Base 모델 학습 시작"""
        self.base = FineTunedLlama8B()
        self.base.finetune_async(self.collected_data)
        # 비동기 학습하는 동안 Stage 1 계속

    def on_base_trained(self):
        """Base 학습 완료 → Stage 3로"""
        self.stage = 3

    def on_performance_plateau(self):
        """성능 안정화 → Stage 4로"""
        self.stage = 4
```

**비용 프로필**:
```
Week 1: $500 (SOTA, 빠른 진전)
Week 2: $300 (SOTA + Base 학습)
Week 3-4: $100 (Hybrid, 비용 감소)
Week 5+: $50 (Base, 최저 비용)
────────────────────────────────
Total (1 month): $950

vs Pure SOTA: $2,000+
vs Pure Baseline: $250 + 2-3주 지연
```

### 8.2 연구 방향

**1) SOTA 모델의 경험 합성 최적화**:
- DeepSeek-R1/o1에 특화된 프롬프트 엔지니어링
- Thinking budget 최적화
- In-context learning 전략

**2) Distillation 효율성**:
- SOTA → Base 지식 전이 최적화
- Active learning: 어떤 샘플을 SOTA로 생성할지

**3) 비용 최적화**:
- 배치 처리, 캐싱 전략
- SOTA 호출 최소화하는 정책 학습

---

## 9. 결론

### 9.1 핵심 답변

> **Q: M_exp를 학습 없이 SOTA 추론 모델로 대체한다면?**

**A: 상황에 따라 다름, 하지만 Hybrid가 최선**

### 9.2 장단점 요약

| 측면 | 학습 기반 M_exp | Zero-Shot SOTA | Hybrid |
|------|----------------|----------------|--------|
| **오프라인 데이터** | 필요 (2K-10K) | 불필요 ✅ | 최소 (1K) ✅ |
| **개발 시간** | 3주 | **1주** ✅ | 1.5주 ✅ |
| **추론 품질** | 중간 | **최고** ✅ | 높음 ✅ |
| **추론 비용** (100 iter) | **$250** ✅ | $2,500 ❌ | $600 ✅ |
| **추론 속도** | **빠름** ✅ | 느림 ❌ | 중간 ✅ |
| **도메인 적응** | 우수 | 중간 | 우수 ✅ |
| **확장성** | 우수 ✅ | 제한적 | 우수 ✅ |
| **새 도메인 적응** | 느림 | **즉시** ✅ | 빠름 ✅ |

**종합 점수**:
- 학습 기반: 7/9 ⭐⭐⭐
- Zero-Shot: 5/9 ⭐⭐
- **Hybrid: 9/9** ⭐⭐⭐⭐⭐

### 9.3 최종 권장

**🏆 Best Practice: Progressive Hybrid Approach**

```
Phase 0: Zero-Shot SOTA (DeepSeek-R1)
  - 즉시 시작, 빠른 검증
  - 1K 고품질 샘플 수집
  - 비용: $1, 시간: 1-2일

Phase 1: Base 모델 학습
  - Phase 0 데이터로 파인튜닝
  - 비용: $50, 시간: 4-8시간

Phase 2: Hybrid 운영
  - Base (90%) + SOTA (10%)
  - 비용: $300/100 iter
  - 최고 성능 + 합리적 비용

Phase 3: Base 중심 (Optional)
  - 성능 안정화 후
  - 비용 최소화
```

**이점**:
- ✅ 오프라인 데이터 의존성 90% 감소
- ✅ 개발 시간 50% 단축
- ✅ SOTA 추론 능력 활용
- ✅ 비용 증가 2-3배 (감당 가능)
- ✅ 최종 성능 향상 기대

**AgentEvolver와의 관계**:
> Hybrid 접근 시 **AgentEvolver 불필요**. SOTA 모델이 이미 zero-shot으로 초기 데이터 문제 해결.

---

## 10. 부록: 코드 스니펫

### 10.1 DeepSeek-R1 기반 Experience Model

```python
import anthropic  # or openai for DeepSeek API

class DeepSeekR1ExperienceModel:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def predict(self, state, action, task, similar_exp):
        """
        DeepSeek-R1 기반 경험 예측

        Note: state는 이미 멀티턴 맥락을 포함
        """
        prompt = self._build_prompt(state, action, task, similar_exp)

        response = self.client.messages.create(
            model="deepseek-reasoner",  # R1
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            # DeepSeek-R1 specific
            thinking_budget="medium"  # low, medium, high
        )

        # Parse <think>...</think> and prediction
        reasoning, next_state, reward = self._parse_response(
            response.content[0].text
        )

        return next_state, reward, reasoning

    def _build_prompt(self, state, action, task, similar_exp):
        return f"""
You are simulating an environment for RL agent training.

Task: {task}

Current State (includes context from previous turns in multi-turn interactions):
{state}

Agent Action:
{action}

Similar Past Experiences (for reference):
{similar_exp}

Generate the next state and reward with reasoning:
1. <think>Your step-by-step reasoning</think>
2. Next State: [formatted state]
3. Reward: 0 or 1
"""

    def _parse_response(self, text):
        # Extract reasoning
        reasoning = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        reasoning = reasoning.group(1) if reasoning else ""

        # Extract next state
        state_match = re.search(r'Next State:\s*(.*?)(?=Reward:|$)', text, re.DOTALL)
        next_state = state_match.group(1).strip() if state_match else ""

        # Extract reward
        reward_match = re.search(r'Reward:\s*([01])', text)
        reward = int(reward_match.group(1)) if reward_match else 0

        return reasoning, next_state, reward
```

### 10.2 비용 추적 래퍼

```python
class CostTrackedExperienceModel:
    def __init__(self, base_model, sota_model, cost_per_base=0.0001, cost_per_sota=0.001):
        self.base = base_model
        self.sota = sota_model
        self.cost_base = cost_per_base
        self.cost_sota = cost_per_sota

        self.total_calls = 0
        self.base_calls = 0
        self.sota_calls = 0
        self.total_cost = 0.0

    def predict(self, *args, use_sota=False):
        self.total_calls += 1

        if use_sota:
            result = self.sota.predict(*args)
            self.sota_calls += 1
            self.total_cost += self.cost_sota
        else:
            result = self.base.predict(*args)
            self.base_calls += 1
            self.total_cost += self.cost_base

        return result

    def get_stats(self):
        return {
            'total_calls': self.total_calls,
            'base_calls': self.base_calls,
            'sota_calls': self.sota_calls,
            'sota_ratio': self.sota_calls / self.total_calls,
            'total_cost': self.total_cost,
            'cost_per_call': self.total_cost / self.total_calls
        }
```

---

**문서 버전**: 1.0
**작성일**: 2025-11-22
**관련 문서**:
- paper2_detailed_analysis.md
- fusion_methodology.md
