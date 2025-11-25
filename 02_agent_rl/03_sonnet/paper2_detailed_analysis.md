# DreamGym: 상세 논문 분석 (완전판)

## 논문 기본 정보
- **제목**: Scaling Agent Learning via Experience Synthesis
- **arXiv ID**: 2511.03773
- **제출일**: 2025년 11월 7일
- **저자**: Zhaorun Chen¹'³, Zhuokai Zhao¹, Kai Zhang¹, Bo Liu², Qi Qi¹, Yifan Wu¹, Tarun Kalluri¹, Sara Cao¹, Yuanhao Xiong¹, Haibo Tong³, Huaxiu Yao⁴, Hengduo Li¹, Jiacheng Zhu¹, Xian Li², Dawn Song⁵, Bo Li³, Jason Weston², Dat Huynh¹
- **소속**:
  - ¹Meta Superintelligence Labs
  - ²FAIR at Meta
  - ³University of Chicago
  - ⁴UNC
  - ⁵UC Berkeley
- **페이지**: 27페이지 + 부록

## Executive Summary

DreamGym은 **경험 합성을 통한 에이전트 학습 스케일링**을 위한 최초의 통합 프레임워크입니다. 비싼 실제 환경 롤아웃 대신, 추론 기반 경험 모델(reasoning-based experience model)을 통해 일관된 상태 전이와 피드백 신호를 생성하여 확장 가능한 RL 학습을 가능하게 합니다.

### 핵심 성과
- **WebArena (non-RL-ready)**: 모든 베이스라인 대비 **30% 이상 개선**
- **WebShop/ALFWorld**: 순수 합성 경험만으로 **GRPO/PPO와 동등한 성능**
- **DreamGym-S2R**: 실제 데이터 10% 미만 사용하면서 **40% 이상 성능 향상**

## 1. 연구 동기 및 문제 정의

### 1.1 패러다임 비교: Traditional vs DreamGym (Figure 1)

DreamGym은 전통적인 에이전트 학습 패러다임을 근본적으로 재설계합니다.

#### (a) Traditional Agent Learning의 구조적 한계 (Figure 1a)

**아키텍처**:
```
Tasks → Agent → Real Environment
        ↑__________________|
        reward signal (sparse & unstable)
```

**4가지 병목 현상**:

1. **Tasks**: ⚠️ scarce & costly
   - 제한된 태스크 세트
   - 수동 데이터 수집 비용
   - 인간 전문가의 검증 필요

2. **Real Environment**: ⚠️ not scalable
   - Docker/VM 같은 무거운 인프라
   - 병렬화 어려움
   - 되돌릴 수 없는 행동
   - 이질적(heterogeneous) 환경 관리 복잡

3. **Observations**: raw & high-dimensional
   - 픽셀/DOM 같은 원시 관찰
   - 관련 없는 정보 다수 포함
   - 토큰 비효율적

4. **Reward Signals**: sparse & unstable
   - 희소한 피드백
   - 노이즈 많음
   - 동적 환경에서 불안정

#### (b) DreamGym의 패러다임 전환 (Figure 1b)

**새로운 아키텍처**:
```
Tasks ← Curriculum Generator (challenging task variations)
  ↓
Agent ↔ Experience Model (Reasoning-based)
  ↑         |
  └─────────┘ abstract states & reward signals
```

**4가지 핵심 개선**:

1. **Tasks**: ✓ useful & cheap
   - 커리큘럼 기반 자동 생성
   - 보상 엔트로피 기반 적응형 증강
   - 무한 확장 가능

2. **Experience Model**: ✓ vectorized & unified
   - **통합 인프라**: 모든 환경을 단일 LLM 서비스로 추상화
   - **벡터화**: 대규모 병렬 롤아웃 생성 가능
   - **확장성**: Docker/VM 오버헤드 제거

3. **States**: abstract & informative
   - 메타-표현적 텍스트 공간
   - 관련 없는 차원 제거
   - 토큰 효율적
   - action-relevant elements만 포함

4. **Signals**: ✓ abundant & adaptable
   - CoT 추론 기반 일관된 피드백
   - 대규모 합성 경험 생성
   - 안정적인 학습 신호

**핵심 인사이트** ⭐:
> DreamGym은 "완벽하게 사실적인 환경 복제"가 아니라, **"충분히 다양하고 정보가 풍부하며 인과적으로 근거있는 상호작용 데이터"**를 제공하는 것이 에이전트 학습의 본질임을 보여줍니다.

**패러다임 전환의 의미**:
- Traditional: 환경이 병목 → 확장성 제한
- DreamGym: 환경을 추상화 → 무한 확장 가능
- Traditional: 실제 롤아웃에 의존 → 비용 높음
- DreamGym: 합성 경험 생성 → 비용 낮음

---

### 1.2 LLM 에이전트와 강화학습의 필요성

대규모 언어 모델(LLM) 기반 에이전트는 포괄적인 사전학습 지식을 활용하여 웹 내비게이션, 구현 제어, 다중 턴 도구 사용 등에서 유망한 성과를 보였습니다. 하지만 **다운스트림 상호작용 환경에서의 성능은 여전히 제한적**입니다.

**경험의 시대(Era of Experience)**에 진입하면서, 더 강건하고 적응력 있는 언어 에이전트를 구축하는 유망한 방향은 **강화학습(RL)**입니다. 에이전트가 환경과 상호작용하며 자신의 경험으로부터 자가 개선(self-improvement)하는 것입니다.

### 1.3 RL 에이전트 학습의 4가지 근본적 장벽 (Figure 1a 참조)

실제로 LLM 에이전트를 RL로 학습시키는 것은 매우 어렵습니다:

#### 1) 높은 비용과 낮은 샘플 효율성
**문제**:
- 긴 상호작용 시퀀스 필요
- 단계당 높은 계산 비용
- 희소한 보상 피드백
- 대규모 다양한 온라인 상호작용 데이터 수집이 매우 비쌈

**실제 예시**:
- 웹 환경: API 호출 비용
- 로봇: 물리적 제약 및 시간
- 시뮬레이션: 계산 자원 소모

#### 2) 제한된 태스크 다양성
**문제**:
- 대부분 환경은 제한된 정적 명령어 세트만 제공
- RL 학습은 효과적인 탐색을 위해 광범위한 태스크 필요
- 태스크 명령어 스케일링은 본질적으로 어려움
- 실행 가능성 검증에 비싼 인간 전문지식 필요

**결과**: 현재 환경들은 goal-conditioned RL에 불충분

#### 3) 불안정한 보상 신호
**문제**:
- 웹페이지, GUI 등은 매우 동적
- 일관된 동작 부족
- 노이즈가 많고, 희소하거나, 심지어 잘못된 피드백

**결과**: 안정적인 학습을 방해

#### 4) 인프라 복잡성
**문제**:
- 기존 시스템은 이질적(heterogeneous)
- Docker나 VM 같은 무거운 백엔드 의존
- 대규모 배치 롤아웃 샘플링이 엔지니어링 집약적이고 비쌈
- 특정 행동은 되돌릴 수 없음 (예: 실제 웹사이트에서 아이템 삭제)
- 대부분 환경은 신뢰할 수 있는 리셋 메커니즘 부족

**결과**: 범용적이고 확장 가능한 에이전트 RL 시스템 구축이 미해결 과제

### 1.4 DreamGym의 솔루션 (Figure 1b 참조)

이러한 도전들을 해결하기 위해, DreamGym은:

1. **확장 가능한 추론 기반 경험 모델**: 환경 동역학을 이산적 텍스트 공간으로 추상화
   - Figure 1b의 "Experience Model" 컴포넌트
   - "vectorized & unified" 인프라로 대규모 병렬 처리

2. **일관된 전이와 피드백**: 명시적 추론을 통해 에이전트 행동의 결과 반영
   - "abstract states & reward signals" 생성
   - "abundant & adaptable" 특성

3. **커리큘럼 태스크 생성**: 보상 엔트로피 기반 적응형 태스크 증강
   - Figure 1b의 "challenging task variations"
   - "useful & cheap" 태스크 자동 생성

4. **핵심 인사이트**: 에이전트 학습은 완벽하게 사실적인 환경이 아니라, **충분히 다양하고 정보가 풍부하며 인과적으로 근거있는 상호작용 데이터**가 필요

**Figure 1의 시사점**:
- 전통적 접근의 모든 병목을 체계적으로 해결
- 패러다임 수준의 혁신으로 확장성 달성

## 2. 핵심 방법론: DreamGym 프레임워크

### 2.1 문제 정식화 (Preliminaries)

#### 2.1.1 MDP 정의

에이전트 학습 문제를 Markov Decision Process (MDP)로 정식화:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, R, \gamma, \rho_0)
$$

여기서:
- $\mathcal{S}$: 상태 공간
- $\mathcal{A}$: 행동 공간
- $T: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$: 전이 함수
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: 보상 함수
- $\gamma \in [0,1]$: 할인 인자
- $\rho_0 \in \Delta(\mathcal{S})$: 초기 상태 분포 (태스크 명령어 $\tau_0$ 포함)

**LLM 에이전트 환경에서**:
- $\tau_0$: 사용자가 자연어로 명시한 원하는 태스크
- $s \in \mathcal{S}$: 에이전트에게 보이는 환경 구성 (웹페이지 내용, 도구 출력, 텍스트 환경 설명 등)
- $a \in \mathcal{A}$: 이산적 동작 (UI 요소 클릭, 외부 도구 호출, 텍스트 응답 생성 등)
- 에이전트는 정책 $\pi_\theta: \mathcal{S} \to \Delta(\mathcal{A})$ 유지 (매개변수 $\theta$)

#### 2.1.2 경험으로부터의 에이전트 학습

**온라인 경험**: $\epsilon = \{\tau_0 | s_0, a_0, ...\}$ (태스크 $\tau_0$와 상태-행동 롤아웃)

**RL의 목표**: 기대 누적 보상 최대화

정책 그래디언트를 통한 최적화:

$$
\nabla J(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_\theta} \left[ \nabla \log \pi_\theta(a_t | s_t) \cdot \hat{A}(s_t, a_t) \right]
$$

여기서 $\hat{A}(s_t, a_t)$는 advantage 함수 (다른 행동과 비교하여 행동이 얼마나 유리한지 추정)

#### 2.1.3 PPO (Proximal Policy Optimization)

GAE (Generalized Advantage Estimation)로 advantage 계산:

$$
\hat{A}_t^{\text{PPO}} = \sum_{l=0}^{K-1} (\gamma\lambda)^l [r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})]
$$

여기서:
- $V(\cdot)$: LLM으로 근사한 가치 함수
- $\lambda$: 편향-분산 트레이드오프 제어

#### 2.1.4 GRPO (Group Relative Policy Optimization)

가치 함수를 버리고 그룹 내 정규화된 advantage 사용:

$$
\hat{A}_t^{\text{GRPO}} = \frac{r_t - \text{mean}_{i \in G}(r_i)}{\text{std}_{i \in G}(r_i)}
$$

여기서:
- $r_t$: 출력 $o_t$에 대한 보상
- $G$: 동일한 태스크 명령어에 대해 샘플링된 응답 그룹

**특징**: 가치 함수 제거로 더 확장 가능하지만 샘플 효율성은 잠재적으로 낮음

**중요**: DreamGym은 특정 RL 알고리즘에 직교하며, **다양하고 정보가 풍부한 경험 합성 스케일링**에 집중

### 2.2 추론 경험 모델 구축 (Building Reasoning Experience Models)

#### 2.2.1 설계 철학

**핵심 인사이트**:
> 에이전트 학습은 원시 픽셀 공간에서 실제 세계를 복제하는 것이 아니라, **추상적이고 메타-표현적인 텍스트 공간 $\mathcal{S}$**에서 작동하는 효율적인 추론 경험 모델 $M_{\text{exp}}$를 구축할 수 있다.

**장점**:
1. **차원 축소**: 관련 없는 차원 제거, 더 정보가 풍부한 궤적
2. **토큰 효율성**: 원시 관찰보다 효율적
3. **샘플 효율성**: 소규모 공개 궤적 데이터셋만으로 학습 가능

**예시 (웹 쇼핑)**:
- ❌ 원시 HTML 코드 처리 (헤더, 태그 등 불필요한 구조적 인공물 포함)
- ✅ 깨끗한 요소 목록 직접 합성

#### 2.2.2 경험 롤아웃 수집을 위한 추론 (Inference)

**입력**: 현재 상태-행동 쌍 외에 3가지 추가 컨텍스트

1. **상호작용 히스토리** $\{(s_i, a_i)\}_{i=0}^t$:
   - 과거 궤적을 컨텍스트 윈도우에 포함
   - 여러 턴에 걸쳐 상태 일관성 유지

2. **태스크 명령어** $\tau$:
   - 현재 목표를 조건화
   - 태스크 목표와 관련하여 행동 해석 가능
   - 상태 전이와 보상 모두 더 정확하게 예측

3. **과거 경험** $\{d_j\}_{j=1}^k$:
   - 리플레이 버퍼에서 의미적 유사도로 검색
   - Top-k 데모: $\{d_j\}_{j=1}^k = \text{Top}_k(\cos(\phi(s_t, a_t), \phi(s_i, a_i)))$
   - $\phi(\cdot)$: 임의의 의미 인코더
   - 환각 감소, 지식 집약적 상태 예측의 사실성 향상

**추론 과정 (Chain-of-Thought)**:

$$
(s_{t+1}, r_{t+1}) = M_{\text{exp}}^{R_t} \left( \{(s_i, a_i)\}_{i=0}^t, \{d_j\}_{j=1}^k, \tau \right)
$$

여기서:
- $R_t$: 명시적 추론 트레이스 (경험 모델이 생성)
- 상태 전이를 안내하는 추론

**추론의 역할**:
- 가장 일관되고 정보가 풍부한 전이와 피드백 예측
- 에이전트 행동의 결과 반영

**예시**:
- ✅ 행동이 유효 → 성공 상태로 전이, r=1 할당
- ❌ 행동이 무효 → 실패 상태로 전이, r=0 할당

**보상 체계** (Feng et al., 2025 따름):
- Outcome-based reward
- $r = 1$: 최종 단계에서 태스크 성공적 완료 시만
- $r = 0$: 다른 모든 경우

#### 2.2.3 추론하도록 경험 모델 학습 (Training)

**데이터 효율성**:
- 추상 상태 공간 설계 덕분에 매우 샘플 효율적
- 제한된 실제 환경 데이터만 필요
- 공개 벤치마크의 풍부한 오프라인 궤적 데이터셋으로 충분 (예: WebArena Leaderboard)

**학습 프로세스**:

**1단계: 추론 트레이스 주석**

궤적 데이터셋 $\mathcal{D} = \{(s_t, a_t, s_{t+1}, r_{t+1})\}$에 대해:
- 각 전이에 명시적 추론 트레이스 $R_t^*$ 주석 (LLM으로 생성)
- $R_t^*$: 왜 상태 $s_t$에서 행동 $a_t$를 취했을 때 다음 상태 $s_{t+1}$과 보상 $r_{t+1}$로 이어지는지 설명

**2단계: 지식 증류 (SFT)**

공동 목표로 $M_{\text{exp}}$ 학습:

$$
\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(s_t, a_t, s_{t+1}, R_t^*) \sim \mathcal{D}} \left[ -\log P_\theta(R_t^* | s_t, a_t, H_t, \mathcal{D}_k) - \log P_\theta(s_{t+1} | s_t, a_t, R_t^*, H_t, \mathcal{D}_k) \right]
$$

여기서:
- $H_t$: 상호작용 히스토리
- $\mathcal{D}_k$: 검색된 top-k 데모
- $\theta$: $M_{\text{exp}}$의 매개변수

**목표 보장**:
1. 행동의 인과적 효과를 설명하는 충실한 추론 트레이스 생성 학습
2. 이러한 트레이스를 활용하여 일관되고 정보가 풍부한 다음 상태 예측

**결과**: 모델은 전문가 궤적 모방뿐 아니라 **RL 학습 중 새로운 롤아웃에 대한 추론 일반화 능력** 획득

### 2.3 커리큘럼 기반 태스크 생성 (Curriculum-based Task Generation)

#### 2.3.1 동기

**필요성**:
- 다양하고 커리큘럼에 맞춘 태스크 명령어는 RL 에이전트의 지식 습득에 중요
- 태스크 수집 스케일링은 비싸며, 각 태스크의 실행 가능성 검증에 상당한 인간 노력 필요

**DreamGym의 솔루션**:
- 합성 다중 턴 전이를 통해 목표 도메인 내 임의의 새로운 태스크에 적응
- 동일한 경험 모델이 능동적으로 새 태스크 생성

#### 2.3.2 태스크 생성 프로세스

**공식**:

$$
\tau_t = M_{\text{task}}(\{\tau_{t-1}^i\}_{i=1}^m)
$$

여기서:
- $M_{\text{task}}$: $M_{\text{exp}}$와 매개변수 공유
- $\{\tau_{t-1}^i\}_{i=1}^m$: m개의 시드 태스크

**시드 태스크 선택 기준**:

1. **충분히 도전적**: 현재 에이전트 정책에 대해 → 정보 획득 최대화
2. **잘 정의됨**: 비현실적이거나 잘못 형성된 태스크 폐기 가능

#### 2.3.3 그룹 기반 보상 엔트로피 (핵심 혁신!)

**태스크 가치 정의**:

$$
V_\tau = \frac{1}{n} \sum_{i=1}^n (r_i - \bar{r})^2, \quad \text{where} \quad \bar{r} = \frac{1}{n} \sum_{i=1}^n r_i
$$

여기서:
- $r_i$: 태스크 $\tau$의 그룹 $G$ 내 n개 롤아웃의 결과 보상
- 본질적으로 **보상의 분산**

**해석**:

1. **$V_\tau = 0$ (분산 없음)**:
   - 모든 롤아웃이 성공하거나 모두 실패
   - 태스크가 너무 쉽거나 너무 어려움
   - 정보 획득 낮음

2. **$V_\tau > 0$ (분산 있음)**:
   - 에이전트가 태스크에서 성공과 실패 모두 관찰
   - 태스크가 실행 가능하지만 도전적임을 신호

3. **최대 엔트로피 (성공/실패 균등)**:
   - 신용 할당(credit assignment)에 최대 정보 획득
   - **LLM이 중간 난이도 태스크에서 가장 효과적으로 학습**한다는 최근 발견과 일치 (Gao et al., 2025)

**알고리즘**:
- 높은 엔트로피 태스크를 $M_{\text{task}}$에 피드
- 점진적으로 더 도전적인 변형 생성
- 에이전트 탐색과 지식 습득 향상

**안정성 제어**:
- 하이퍼파라미터 $\lambda$: 반복당 샘플링되는 합성 태스크 비율 제한
- 원래 태스크 분포의 충분한 커버리지 유지
- 현재 정책의 약점 영역으로 탐색 유도

**그룹 $G$ 정의**:
- **GRPO**: $G$는 단순히 학습 그룹
- **PPO**: 의미 임베더로 태스크 먼저 클러스터링 → 각 클러스터가 그룹 형성

### 2.4 합성 경험으로부터 학습 (Learning from Synthetic Experiences)

#### 2.4.1 합성 환경에서 정책 학습

**전체 학습 루프**:

```python
# DreamGym 학습 알고리즘

1. 시드 태스크 세트로 시작

2. 각 반복마다:
   a) 각 태스크에 대해 다중 턴 롤아웃 생성:
      - 에이전트 정책: 상태에서 행동 선택
      - 경험 모델: 에이전트 행동, 히스토리, 태스크 컨텍스트를 조건으로 다음 상태 예측

   b) 수집된 롤아웃으로 표준 RL 알고리즘 사용하여 정책 업데이트

   c) 경험 모델이 태스크 세트 증강:
      - 높은 보상 엔트로피의 도전적 태스크 변형 생성

3. 수렴 또는 사전 정의된 학습 예산 도달까지 계속
```

**특징**:
- 순수 합성 경험으로 학습
- 실제 환경과의 상호작용 필요 없음
- 다양하고 커리큘럼 기반 태스크 확장

#### 2.4.2 이론적 보장: 정책 개선 정리

**Theorem 1 (합성 경험을 통한 실제 환경에서의 정책 개선 $J$)**

**설정**:
- 실제 MDP: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$
- 합성 MDP: $\hat{\mathcal{M}} = (\mathcal{S}, \mathcal{A}, \hat{P}, \hat{R}, \gamma)$ ($M_{\text{exp}}$가 유도)
- 할인: $\gamma \in (0,1)$
- 보상 제한: $R, \hat{R} \in [0, R_{\max}]$, $V_{\max} := R_{\max}/(1-\gamma)$

**1단계 경험 모델 오차**:

$$
\epsilon_R := \sup_{s,a} |R(s,a) - \hat{R}(s,a)|
$$

$$
\epsilon_P := \sup_{s,a} \text{TV}(P(\cdot|s,a), \hat{P}(\cdot|s,a))
$$

**신뢰 영역 업데이트**: $\pi \to \pi'$ ($\hat{\mathcal{M}}$에서 최적화, 상태별 KL 반경 $\sup_s \text{KL}(\pi'(\cdot|s) \| \pi(\cdot|s)) \le \delta$)

**정리**:

$$
J_{\mathcal{M}}(\pi') - J_{\mathcal{M}}(\pi) \ge \underbrace{\frac{1}{1-\gamma} \mathbb{E}_{s \sim d_\pi^{\hat{\mathcal{M}}}, a \sim \pi'(\cdot|s)} [A_\pi^{\hat{\mathcal{M}}}(s,a)]}_{\text{합성 대리 이득 (synthetic surrogate gain)}} - \underbrace{\frac{4\gamma}{(1-\gamma)^2} V_{\max} \delta}_{\text{신뢰 영역 페널티}} - \underbrace{2\left(\frac{\epsilon_R}{1-\gamma} + \frac{2\gamma R_{\max}}{(1-\gamma)^2} \epsilon_P\right)}_{\text{경험 모델 오차}}
$$

**해석**:

1. **합성 대리 이득**: 합성 환경에서 학습하고 평가할 때 에이전트의 성능 개선
2. **신뢰 영역 페널티**: KL 반경 $\delta$ 제약 (PPO/GRPO가 소프트하게 강제)
3. **경험 모델 오차**:
   - **$\epsilon_R$ (피드백 충실성)**: 보상 신호가 실제 결과를 얼마나 정확히 반영하는가
   - **$\epsilon_P$ (도메인 일관성)**: 상태 공간 분포가 원래 환경의 동역학과 얼마나 잘 일치하는가

**핵심 인사이트** ⭐:
> 두 오차 항($\epsilon_R$, $\epsilon_P$)은 §4.1의 설계 인사이트와 일치: 합성 환경은 원시 상태 수준에서 원래 환경을 복제할 필요 없이, **도메인 일관된 전이와 올바른 회고적 학습 신호만 제공**하면 됨.

**실용성**:
- 실제로 $\epsilon_R$과 $\epsilon_P$ 모두 매우 작게 만들 수 있음
- $M_{\text{exp}}$를 명시적 추론 트레이스로 주석된 최소한의 궤적 데이터로 학습해도 가능

#### 2.4.3 Sim-to-Real 정책 전이 (DreamGym-S2R)

**전략**:
1. **1단계 (Sim)**: DreamGym에서 순수 합성 경험으로 에이전트 정책 먼저 학습
2. **2단계 (Real)**: 실제 환경으로 전이하여 RL

**장점**:
- **합성 사전학습**: 다양한 태스크에 걸쳐 탐색 커버리지 확대
- **광범위한 지식 습득**: 낮은 비용으로
- **강력한 초기화**: 후속 실제 환경 학습을 더 샘플 효율적으로

**일관성 보장**:
- 합성과 실제 환경 간 상태 공간 일관성 유지
- 동일한 규칙 기반 매핑 함수 적용
- 또는 경량 파인튜닝 모델 사용 (Lee et al.)

## 3. 실험 설계 및 결과

### 3.1 실험 설정

#### 3.1.1 평가 환경

**1) WebShop** (Yao et al., 2022)
- **설명**: 대규모 에이전트 벤치마크, 상호작용 환경에서 언어 그라운딩 연구
- **규모**: 1.18M 실제 제품, 12,087 크라우드소싱 자연어 명령어
- **과제**: 전자상거래 웹사이트 시뮬레이션
  - 검색, 맞춤화, 아이템 구매
  - 조합적 제품 요구사항 해석
  - 쿼리 재구성
  - 노이즈가 많은 웹페이지 텍스트 처리
  - 다양한 페이지 유형 전략적 탐색

**2) ALFWorld** (Shridhar et al.)
- **설명**: 텍스트-구현 벤치마크, 수작업 태스크 명령어
- **규모**: 6가지 가정 태스크 패밀리, 3,553 학습 태스크, 120 방
- **과제**:
  - TextWorld의 추상 텍스트 상호작용
  - ALFRED/AI2-THOR의 사진처럼 사실적인 물리 기반 실행
  - 태스크 유형: Pick & Place, Clean & Place, Heat/Cool & Place
  - 고수준 텍스트 행동: goto, open, take, clean/heat/cool, put
  - 저수준 비주얼모터 컨트롤러로 실현
  - 부분 관측성, 객체 검색 및 조작
  - 언어를 행동 전제조건 및 어포던스로 매핑
  - 추상 계획과 물리적 실행 가능성 간 격차 해소

**3) WebArena-Lite** (Zhou et al., Liu et al., 2024)
- **설명**: 현실적 웹 상호작용 인터페이스, **RL-ready 아님**
- **원본**: 812 장기 태스크
- **평가 세트**: 165 고품질 도전적 태스크 (균형 잡힌 부분집합)
- **학습 세트**: 647 태스크 (평가 세트 제외)
- **과제**:
  - 완전 기능적 사이트: 전자상거래, 소셜 포럼, 협업 소프트웨어 개발(GitLab), 콘텐츠 관리
  - 도구: 지도, 계산기, 스크래치패드
  - 지식 베이스: 오프라인 Wikipedia, 매뉴얼
  - 고수준 자연어 의도로 표현
  - 기능적 정확성으로 평가 (행동 트레이스 매칭 아님)
  - 다중 탭 브라우징 지원
  - 풍부한 행동 공간 (클릭, 타이핑, 내비게이션, 탭 동작)
- **제약**:
  - 본질적으로 확장 가능한 데이터 수집 부족
  - 환경 리셋 메커니즘 없음
  - 높은 계산 비용

**환경 선택 이유**:
- RL이 실행 가능하지만 계산적으로 비싼 설정 평가
- RL 학습이 아직 실용적이지 않은 설정 평가

#### 3.1.2 에이전트 백본

서로 다른 모델 패밀리와 크기:
- **Llama-3.2-3B-Instruct**
- **Llama-3.1-8B-Instruct** (Grattafiori et al., 2024)
- **Qwen-2.5-7B-Instruct** (Team, 2024)

#### 3.1.3 베이스라인

**1) 오프라인 모방 학습**:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization) (Rafailov et al., 2023)

**2) 실제 환경에서의 온라인 RL (전통적)**:
- GRPO (Shao et al., 2024)
- PPO (Schulman et al., 2017)

#### 3.1.4 DreamGym 변형

**1) DreamGym (순수 합성)**:
- 완전히 DreamGym 내에서 GRPO/PPO 적용
- 실제 상호작용 **0개**

**2) DreamGym-S2R (Sim-to-Real)**:
- 합성 학습 후 소규모 RL 단계
- 원래 환경에서 제한된 롤아웃만 필요
- 중간 학습 단계로 DreamGym 사용의 효과 입증

#### 3.1.5 구현 세부사항

**경험 모델**:
- **Llama-3.1-8B-Instruct**에서 학습 (모든 메인 결과)
- 섹션 §4.1 참조

**매개변수 설정**: 부록 A 참조

**환경별 설정**:

**WebShop**:
- 오프라인 데이터: 1,600 인간 데모 + 2,000 오라클/랜덤 탐색 궤적
- 각 전이에 강한 교사 LLM이 생성한 추론 트레이스 증강
- 계산 자원: 8 노드 A100 GPU + 4 노드 H100 GPU

**ALFWorld**:
- 기본 ALFWorld 분할 (TextWorld 설정)
- 학습 분할에서 3,200 전문가 데모 + 2,000 오프라인 궤적 (오라클/랜덤)
- 각 전이에 추론 트레이스 증강
- 계산 자원: 8 노드 A100 GPU + 4 노드 H100 GPU

**WebArena**:
- **RL 베이스라인 과제**:
  - 오픈소스 RL 인프라 없음 (Qi et al., 2024)
  - Verl-Agent (Feng et al., 2025) 표준 워크플로우 따름
  - browsergym으로 행동 궤적 수집
  - AWS 서버에 WebArena 웹사이트 호스팅
  - PPO와 GRPO 모두 지원
  - **병목**: 최대 4개 AWS 서버만 운영 가능 → 4개 병렬 상호작용 세션만
  - 태스크 세트 순차적으로 순회, 모든 태스크 방문 후 수동 서버 재시작 및 환경 리셋
  - 알려진 문제: 일부 궤적 실행 실패, 일부 태스크 평가 함수 오류 (Qi et al., 2024; Chae et al., 2025)
  - 공정성을 위해 모든 수집된 궤적 그대로 RL 학습에 사용

- **DreamGym 설정**:
  - 오프라인 궤적: WebArena 공개 리더보드에서 가장 높은 성능의 에이전트로부터 성공적 데모 추출
  - 포함 에이전트: IBM CUGA, ScribeAgent, Learn-by-Interact, AgentOccam (모두 관찰에 접근성 트리 정보 포함)
  - 분포 불균형 완화: 고성능 에이전트 + 랜덤 정책 생성 궤적 추가
  - 총 4,800 오프라인 궤적
  - 각 전이에 추론 트레이스 증강
  - 계산 자원: 8 노드 A100 GPU + 4 노드 H100 GPU

### 3.2 주요 실험 결과

#### 3.2.1 전체 성능 비교 (Table 1)

| 알고리즘 | 실제 데이터 | WebShop (L3.2-3B / L3.1-8B / Q2.5-7B) | ALFWorld (L3.2-3B / L3.1-8B / Q2.5-7B) | WebArena (L3.2-3B / L3.1-8B / Q2.5-7B) |
|---------|------------|---------------------------------------|----------------------------------------|---------------------------------------|
| **오프라인 모방 학습** ||||
| SFT | 20K | 32.0 / 35.1 / 32.9 | 61.7 / 68.0 / 71.8 | 6.1 / 5.5 / 7.3 |
| DPO | 40K | 35.9 / 31.0 / 34.8 | 63.3 / 63.9 / 61.1 | 5.5 / 4.8 / 4.8 |
| **GRPO** ||||
| Traditional | 80K | 62.1 / 65.0 / 66.1 | 65.3 / 70.9 / 79.8 | 7.3 / 6.1 / 6.1 |
| DreamGym | **0** | 59.3 / 63.9 / 68.3 | 62.1 / 66.3 / 71.0 | **13.3 / 9.1 / 12.7** |
| DreamGym-S2R | 5K | **70.5 / 75.0 / 72.1** | **65.0 / 75.9 / 82.4** | **13.9 / 9.7 / 11.2** |
| **PPO** ||||
| Traditional | 80K | 59.9 / 64.2 / 68.1 | 47.0 / 72.9 / 81.1 | 6.7 / 4.8 / 7.3 |
| DreamGym | **0** | 60.5 / 58.1 / 65.0 | 40.5 / 70.8 / 72.7 | **14.5 / 10.9 / 10.0** |
| DreamGym-S2R | 5K | **66.0 / 63.9 / 73.7** | **49.1 / 73.3 / 79.9** | **13.3 / 10.9 / 13.9** |

**참고**: "실제 데이터"는 개별 전이 수 (궤적은 보통 ~10 단계)

#### 3.2.2 결과 분석

**1) Non-RL-ready 환경 (WebArena)**

**DreamGym의 가장 큰 장점**:
- 대규모 RL 인프라 사용 불가능한 환경에서 최대 효과
- 모든 백본에서 **30% 이상 성공률** 달성
  - Llama-3.2-3B: **13.3%** vs 7.3% (Traditional GRPO) → **+82% 상대적 개선**
  - Llama-3.1-8B: **10.9%** vs 6.1% (Traditional GRPO) → **+79% 상대적 개선**
  - Qwen-2.5-7B: **13.9%** vs 7.3% (Traditional PPO) → **+90% 상대적 개선**

**제로샷 RL 베이스라인 실패 이유**:
- 제한된 탐색 다양성
- 원래 환경에서 희소한 보상 신호
- 환경 제약

**의미**:
> DreamGym은 단순히 비싼 롤아웃의 대체가 아니라, **본질적인 태스크 및 엔지니어링 제약으로 인해 이전에 실용적이지 않았던 도메인에서 RL 학습을 가능하게 하는 메커니즘**

**2) RL-ready 환경 (WebShop, ALFWorld)**

**순수 합성 경험으로 실제 RL과 동등**:
- WebShop:
  - DreamGym (0 실제 데이터): 59.3-68.3%
  - Traditional GRPO (80K 실제 데이터): 62.1-66.1%
  - **동등하거나 더 나은 성능**

- ALFWorld:
  - DreamGym (0 실제 데이터): 62.1-71.0%
  - Traditional GRPO (80K 실제 데이터): 65.3-79.8%
  - **비슷한 범위**

**의미**:
> DreamGym이 생산하는 전이와 보상은 일관되고 의미 있을 뿐 아니라, **안정적인 정책 개선에 충분**

**Sim-to-Real (DreamGym-S2R) 압도적 우위**:
- 소규모 RL 단계 (5K 실제 롤아웃) 추가 시:
  - WebShop: **70.5-75.0%** (Traditional 대비 +13-15% 절대적 개선)
  - ALFWorld: **73.3-82.4%** (Traditional 대비 +1-6% 절대적 개선)

**의미**:
> 합성 학습은 실제 환경에서 더 샘플 효율적인 RL을 위한 강력한 기반을 확립하는 **효율적인 warm-start 전략**으로 작용 가능

#### 3.2.3 샘플 효율성 및 학습 비용 (Figure 3 Left)

**WebArena 학습 시간 대비 성능** (Figure 3 Left):

| 백본 모델 | 최종 성능 | 학습 시간 | 시간 효율성 |
|-----------|-----------|-----------|-------------|
| **Llama-3.1-8B** (DreamGym) | **~13.0%** | ~40시간 | ⭐⭐⭐⭐⭐ |
| **Llama-3.2-3B** (DreamGym) | ~10.0% | ~60시간 | ⭐⭐⭐⭐ |
| Traditional RL | ~6.0% | ~100시간 | ⭐⭐ |
| Qwen-2.5-7B (DreamGym) | ~6.0% | ~80시간 | ⭐⭐ |

**핵심 인사이트**:

**1. 모델 크기 ≠ 성능**:
- Llama-3.1-8B가 Qwen-2.5-7B (더 큰 모델)보다 **2배 이상 높은 성능**
- **원인**: 사전학습 품질, 인스트럭션 튜닝 데이터의 차이
- **교훈**: 백본 선택이 중요 - 단순히 모델 크기보다 품질 중요

**2. 학습 효율성**:
- Llama-3.1-8B: 40시간 만에 13% 달성
- Traditional RL: 100시간에 6% 달성
- **시간 대비 효율성**: DreamGym이 **2.5배 개선** (13%/40h vs 6%/100h)
- **절대 성능**: DreamGym이 **2배 이상 높음** (13% vs 6%)

**3. 학습 노력 대비 성능**:
- **학습 노력**: 롤아웃 샘플링 시간 + GPU 시간 모두 포함
- **DreamGym**: Traditional RL 베이스라인 대비 **1/3 ~ 1/5로 학습 노력 감소**

**효율성 향상 원인**:
1. **밀집 피드백**: 커리큘럼 기반 롤아웃 합성
2. **경량 전이**: 확장 가능한 LLM 서비스가 호스팅하는 통합 경험 모델의 추상 상태 전이
3. **병목 회피**: 이질적 환경 병목 회피
4. **낮은 샘플링 비용**: 대폭 감소

**의미**:
> DreamGym은 복잡하고 비싼 환경에서 RL 학습을 위한 실용적 솔루션일 뿐 아니라, **안정적인 정책 개선을 위한 저비용 데이터 생성의 확장 가능한 방법**

#### 3.2.4 일반화 및 학습 전이성 (Figure 3 Middle)

**교차 도메인 전이 실험** (Figure 3 Middle):

**실험 설정**:
- 3가지 벤치마크: WebShop (전자상거래), WebArena (복잡한 웹), ALFWorld (가상 가정)
- 각 환경에서 RL 학습 후 다른 환경으로 전이 테스트
- SFT 베이스라인과 비교

**정량적 결과**:

| 소스 환경 | 타겟 환경 | SFT 성능 | RL 성능 | 관찰 |
|-----------|-----------|----------|---------|------|
| WebShop | WebShop | ~30% | ~60% | (동일 환경, RL 효과) |
| WebShop | WebArena | ~40% | ~40% | 전이 유지 |
| WebArena | WebArena | ~60% | ~60% | (동일 환경) |
| WebArena | WebShop | ~30% | ~10% | ⚠️ **급격한 하락** |
| - | ALFWorld | ~10% | ~10% | SFT와 비슷 |

**핵심 발견**:

**1. 비대칭 전이 (Asymmetric Transfer)** ⚖️:

```
WebShop (단순) → WebArena (복잡): 40% 유지 ✓
  상대적으로 양호한 전이

WebArena (복잡) → WebShop (단순): 10% 하락 ✗
  급격한 성능 저하 (▼83%)
```

**원인 분석**:
- **WebShop**: 단순한 전자상거래 환경
  - 검색 → 필터링 → 선택 → 구매 (선형적 프로세스)
  - 제한된 action space
  - 구조화된 인터페이스

- **WebArena**: 복잡한 웹 내비게이션
  - 다양한 웹사이트 (Reddit, GitHub, Shopping 등)
  - 동적 콘텐츠
  - 비선형적 탐색

**왜 복잡 → 단순 전이가 어려운가?**
- **오버피팅**: WebArena 학습 정책이 복잡한 패턴에 과적응
- **태스크 분포 불일치**: WebArena의 다양성이 WebShop의 특정 패턴과 맞지 않음
- **과도한 탐색**: WebArena 정책이 WebShop에서 불필요하게 복잡하게 행동

**2. WebShop → WebArena 전이 성공** ✓:
- WebShop RL: WebArena에서 ~40% 유지
- **상대적으로 양호**: 단순 환경에서 학습한 기본 기술이 복잡한 환경에 일부 전이 가능
- **기본 스킬 전이**: 클릭, 검색, 폼 입력 등의 low-level 기술

**3. ALFWorld의 특수성**:
- SFT와 RL 성능이 비슷 (~10%)
- **원인**: ALFWorld는 가상 가정 환경으로 웹 환경과 도메인 격차가 매우 큼
  - 다른 상태 표현 (방, 가구, 물체)
  - 다른 행동 공간 (집기, 놓기, 열기)
  - 다른 목표 (물리적 조작 vs 정보 검색)

**일반화 한계**:

**✓ 성공 사례**:
- 유사한 도메인 간 전이 (웹 환경 내)
- 단순 → 복잡 방향
- 기본적인 상호작용 패턴 공유 시

**✗ 실패 사례**:
- 복잡 → 단순 (오버피팅)
- 완전히 다른 도메인 (웹 → 가상 가정)
- 상태/행동 공간이 근본적으로 다를 때

**의미**:
> DreamGym은 **추상 메타-표현 공간 내에서 학습**하여, 에이전트가 태스크별 패턴을 암기하기보다 **도메인 불가지적 행동 프라이어(behavioral priors)**를 일부 학습 가능.
>
> ❌ 하지만 DreamGym이 완벽한 전이를 보장하지는 않음
> ✅ 유사 도메인에서는 효과적
> ⚠️ 도메인 격차가 크면 추가 파인튜닝 필요

**실용적 권장사항**:
1. **타겟 도메인이 유사한 경우**: DreamGym으로 사전 학습 후 소량 파인튜닝
2. **타겟 도메인이 완전히 다른 경우**: 타겟 도메인에서 처음부터 DreamGym 학습
3. **복잡도 차이**: 단순 → 복잡은 잘 되지만, 복잡 → 단순은 주의

#### 3.2.5 학습 곡선 분석 (Figure 3 Right - WebShop)

**최종 성능 비교** (150 steps):

| 방법 | 최종 Success Rate | 절대 개선 | 상대 개선 |
|------|-------------------|-----------|-----------|
| Traditional RL | ~10% | (baseline) | - |
| **DreamGym** | **~60%** | +50% | **+500%** |
| DreamGym w/o Task | ~55% | +45% | +450% |
| **DreamGym-S2R** | **~65%** | +55% | **+550%** |

**학습 속도 분석**:

**1. 초기 학습 (0-30 steps)** ⚡:
- **Traditional**: 0% → 5% (매우 느림, 완만한 상승)
- **DreamGym**: 0% → 50% (급격한 상승 🚀)
- **차이**: DreamGym이 30 스텝 만에 Traditional의 최종 성능(10%)의 **5배** 달성
- **원인**: 합성 경험의 밀집 피드백, 커리큘럼 태스크의 정보성

**2. 중기 학습 (30-90 steps)**:
- **Traditional**: 5% → 8% (완만한 상승, 큰 진동)
- **DreamGym**: 50% → 58% (계속 개선, 안정적)
- **DreamGym w/o Task**: 50% → 55% (정체 시작 ⚠️)
- **Ablation 인사이트**: 커리큘럼 태스크 없으면 60 스텝부터 정체
  - 리플레이 버퍼가 낮은 엔트로피, 반복적 궤적으로 포화
  - 경험 다양성 제한 → 탐색 정체

**3. 후기 학습 (90-150 steps)**:
- **Traditional**: 8% → 10% (거의 정체, 최종 수렴)
- **DreamGym**: 58% → 60% (완만한 상승)
- **DreamGym w/o Task**: 55% → 55% (완전 정체)
- **DreamGym-S2R**: 60% → 65% (실제 데이터 추가로 boost ⬆️)
- **S2R 효과**: 합성 학습 후 소량 실제 데이터(5K 롤아웃)로 **5% 추가 개선**

**학습 안정성 비교**:

| 방법 | 곡선 특성 | 분산 |
|------|-----------|------|
| Traditional | 큰 진동 🌊 | 높음 |
| DreamGym | 매끄러움 ━ | 낮음 |
| DreamGym-S2R | 가장 안정적 ━ | 가장 낮음 |

**안정성 원인**:
- **Traditional**: 희소하고 불안정한 실제 환경 보상 → 높은 분산
- **DreamGym**: 합성 궤적의 밀집되고 일관된 피드백 → 낮은 분산
- **DreamGym-S2R**: 합성 + 실제 데이터 결합 → 가장 안정적

**수렴 속도**:
- **Traditional**: 90+ steps에서 수렴 (10% 최종)
- **DreamGym**: 30 steps에 이미 50% 도달 (Traditional 최종의 5배)
- **학습 효율성**: DreamGym은 **1/3 스텝으로 Traditional의 5배 성능** 달성

**커리큘럼의 중요성** (w/o Task 비교):
- DreamGym (60%) vs w/o Task (55%): **-5% 절대 하락**
- 초기 60 스텝까지는 비슷하나, 이후 커리큘럼 없으면 정체
- **교훈**: 지속적인 개선을 위해서는 적응형 태스크 생성 필수

**의미**:
> 합성 궤적이 희소한 실제 롤아웃보다 **더 정보가 풍부한 그래디언트 제공**하며, 커리큘럼 학습과 결합 시 안정적이고 빠른 학습 가능. DreamGym-S2R은 합성 학습이 실제 환경 RL을 위한 **강력한 warm-start 전략**임을 보여줌.

### 3.3 제거 실험 (Ablation Studies)

#### 3.3.1 컴포넌트별 기여도 (Table 2)

| 방법 | WebShop | WebArena |
|------|---------|----------|
| **DreamGym (전체)** | **63.9%** | **13.3%** |
| w/o Exp. Replay | 59.2% (-4.7%) | 9.7% (-3.6%) |
| w/o Exp. Reasoning | 55.8% (-8.1%) | 7.3% (-6.0%) |
| w/o Task Generation | 57.3% (-6.6%) | 7.3% (-6.0%) |

**분석**:

**1) 태스크 생성기 제거 (w/o Task Generation)**:
- WebShop: **-6.6%** 하락
- WebArena: **-6.0%** 하락
- Figure 3 Right에서: 초기 진전 후 더 빨리 정체
- **원인**:
  - 적응적 태스크 생성 없으면 리플레이 버퍼가 낮은 엔트로피, 반복적 궤적으로 포화
  - 경험 다양성 제한 → 탐색 정체
- **태스크 생성기의 역할**:
  - 지속적으로 점진적으로 도전적인 고가치 태스크 생성
  - 에이전트를 현재 능력 너머로 밀어냄
  - 리플레이 버퍼를 정보가 풍부하게 유지
  - 탐색 장려 → 더 높은 최종 성공률 및 더 나은 샘플 효율성

**2) 경험 리플레이 제거 (w/o Exp. Replay)**:
- WebShop: **-4.7%** 하락
- WebArena: **-3.6%** 하락
- **원인**:
  - 과거 유사한 경험 검색 불가
  - 컨텍스트 부족으로 상태 예측 품질 저하
  - 환각 증가 가능

**3) 경험 추론 제거 (w/o Exp. Reasoning)**:
- WebShop: **-8.1%** 하락 (가장 큰 영향!)
- WebArena: **-6.0%** 하락 (가장 큰 영향!)
- **원인**:
  - 추론 능력 없이 생성된 경험이 얕아지고 사실적 근거 감소
  - 정보성 감소, 환각 증가
  - Table 2에서 전체 성능에 실질적 하락
- **추론의 중요성**: 경험 모델의 핵심 컴포넌트

#### 3.3.2 경험 모델 품질 평가 (Figure 4)

**평가 방법**:
- **Judge**: GPT-4o (Hurst et al., 2024)
- **샘플**: 100 궤적 무작위 샘플링
- **점수**: 각 기준에 대해 이산 점수 {0, 1, 2}
  - 0: Poor, 1: Acceptable, 2: Excellent
- **프롬프트**: 부록 C.4 참조

**평가 기준 (Figure 4)**:

1. **Consistency (일관성)**: 상태 전이의 논리적 일관성
   - ⚠️ **맥락 의존적**: 너무 높으면 반복적, 너무 낮으면 무작위
2. **Diversity (다양성)**: 경험의 다양성
   - ✅ **높을수록 좋음**: RL 학습에 필수적
3. **Informativeness (정보성)**: 학습에 유용한 정보 포함 정도
   - ✅ **높을수록 좋음**: 정책 개선에 직접 기여
4. **Hallucination (환각)**: 사실 오류 정도
   - ❌ **낮을수록 좋음**: 2 = 환각 없음, 0 = 많은 환각

**정량적 결과** (Figure 4):

| 기준 | Traditional | DreamGym | w/o History | w/o Reasoning | 해석 방향 |
|------|-------------|----------|-------------|---------------|-----------|
| **Consistency** | ~1.8 | ~1.3 | ~1.3 | ~1.4 | ⚠️ 맥락 의존적 |
| **Diversity** | ~1.3 | **~1.5** ⭐ | ~1.2 | ~1.2 | ✅ 높을수록 좋음 |
| **Informativeness** | ~1.6 | **~1.7** ⭐ | ~1.7 | ~1.4 | ✅ 높을수록 좋음 |
| **Hallucination** | ~1.8 | ~1.6 | ~1.3 | ~1.3 | ❌ 낮을수록 좋음* |

**\*주의**: Hallucination 점수에서 높을수록 환각이 **많다**는 의미가 아니라, Figure 4의 스케일에서는 **Judge가 부여한 점수**를 나타냄. 실제 해석은 다음 분석 참조.

---

**핵심 인사이트**:

**1. Consistency-Diversity 트레이드오프** ⚖️:

```
Traditional: High Consistency (1.8) + Low Diversity (1.3)
   ↓ 문제점
   반복적인 경험 → 정책이 다양한 시나리오 학습 못함
   오버피팅 위험 ⬆️

DreamGym: Medium Consistency (1.3) + High Diversity (1.5)
   ↓ 장점
   일관성 유지하면서도 다양한 경험 → RL 학습에 최적 ✓
   일반화 능력 ⬆️
```

**왜 이 트레이드오프가 좋은가?**
- RL 에이전트는 다양한 상황에 노출되어야 강건한 정책 학습
- **너무 일관적인 경험** = 오버피팅 위험, 탐색 부족
- **적절한 변화** = 일반화 능력 향상, 새로운 상황 대응
- DreamGym: Consistency를 약간 희생 (1.8 → 1.3), Diversity 크게 향상 (1.3 → 1.5)

**실험적 증거**:
- Traditional의 높은 Consistency는 실제로 **단조로움**을 의미
- DreamGym의 중간 Consistency는 **일관성 + 다양성 균형**

**2. Informativeness의 중요성** 📊:

| 변형 | Informativeness | 차이 | 의미 |
|------|-----------------|------|------|
| **DreamGym (전체)** | 1.7 | (기준) | 최고 |
| Traditional | 1.6 | -0.1 | 약간 낮음 |
| w/o History | 1.7 | 0.0 | 유지 (히스토리는 정보성에 중립) |
| **w/o Reasoning** | **1.4** | **-0.3** | ⚠️ **급격히 하락** |

**핵심 발견**:
- **추론(Reasoning)이 정보성에 결정적** (-0.3 하락, 가장 큰 영향)
- 히스토리는 정보성에 큰 영향 없음 (일관성에만 기여)
- **CoT 추론 없이는 정보가 풍부한 경험 생성 불가**

**왜 추론이 중요한가?**
```python
# w/o Reasoning (정보성 1.4):
State 1: "웹페이지 로드됨"
Action: Click(button_123)
State 2: "새 페이지 로드됨"
# ❌ 왜 이 전이가 발생했는지 불명확

# w/ Reasoning (정보성 1.7):
State 1: "웹페이지 로드됨"
Action: Click(button_123)
<think>
The agent clicks "Total Commits" button, which should
display commit history grouped by date...
</think>
State 2: "커밋 리스트 페이지, April 2023 항목 포함"
# ✅ 전이의 인과관계 명확, 태스크 목표와 연결
```

**3. Hallucination 관리** 🚨:

| 변형 | Hallucination 점수 | 해석 |
|------|--------------------|------|
| Traditional | 1.8 | 가장 많은 환각 ⚠️ |
| **DreamGym** | **1.6** | **개선** ✓ |
| w/o History | 1.3 | 더 낮은 환각 |
| w/o Reasoning | 1.3 | 더 낮은 환각 |

**역설적 발견**: 왜 w/o History/Reasoning이 환각이 더 적을까?

**가설 1: Complexity-Hallucination 트레이드오프**
- **추론과 히스토리 → 더 복잡한 상태 생성 → 일부 환각 가능성**
- **하지만**: Informativeness와 Diversity가 훨씬 더 중요
- **결론**: 약간의 환각은 **정보성과 다양성의 대가로 acceptable**

**가설 2: Judge의 평가 기준**
- 단순한 상태 (w/o Reasoning) → 검증하기 쉬움 → 낮은 환각 점수
- 복잡한 상태 (DreamGym) → 검증 어려움 → 약간 높은 환각 점수
- 하지만 복잡한 상태가 **실제로 더 유용**

**Trade-off 분석**:
```
w/o Reasoning: Hallucination 1.3 (낮음), Informativeness 1.4 (낮음)
   ↓ 결과
   안전하지만 정보가 부족한 경험 → RL 학습 느림 ❌

DreamGym: Hallucination 1.6 (약간 높음), Informativeness 1.7 (높음)
   ↓ 결과
   약간의 환각 리스크, but 정보가 풍부한 경험 → RL 학습 빠름 ✅
```

**4. 히스토리의 역할** 📚:

| 기준 | DreamGym | w/o History | 차이 | 해석 |
|------|----------|-------------|------|------|
| Consistency | 1.3 | **0.8** | **-0.5** | ⚠️ 히스토리 없으면 일관성 급락 |
| Diversity | 1.5 | 1.2 | -0.3 | 다양성 감소 |
| Informativeness | 1.7 | 1.7 | 0.0 | 정보성 유지 |
| Hallucination | 1.6 | 1.3 | -0.3 | 환각 감소 (복잡도 감소) |

**히스토리의 기여**:
- **Consistency에 결정적** (-0.5 하락, 가장 큰 영향)
- 시간적 및 인과적 구조 보존
- 여러 턴에 걸친 일관된 내러티브 생성

**5. Ablation 결론** (Figure 4):
- **w/o History**: Consistency 급락 (1.3 → 0.8), but Informativeness 유지
- **w/o Reasoning**: Informativeness 급락 (1.7 → 1.4), Consistency 유지
- **결론**: **히스토리 + 추론 모두 필수**, 상호보완적 역할

**최종 설계 원칙** 🎯:
> 경험 모델은 구조화되고 추론 주도 방식으로 작동해야 궤적의 **다양성과 충실성 모두 유지**. Consistency-Diversity 트레이드오프를 RL 학습에 최적화하고, 약간의 환각은 정보성을 위해 감수.

**분석**:

**히스토리 제거 (w/o History)**:
- **일관성 크게 감소**: 0.8 (가장 낮음)
- **원인**: 이전 턴 인식 없이 모델이 종종 주제를 벗어나고 다중 단계 상호작용에서 인과적 일관성 깨짐
- **다른 지표**: 비교적 유지

**추론 제거 (w/o Reasoning)**:
- **정보성 감소**: 1.1 (상태가 얕아짐)
- **환각 증가**: 1.2 (사실적 근거 감소)
- **일관성**: 비교적 유지
- **원인**: 추론 능력 없이 깊이 및 사실적 신뢰성 감소

**DreamGym (전체)**:
- **모든 지표에서 최고 또는 거의 최고**
- Consistency: **1.6** (최고)
- Informativeness: **1.6** (최고)
- Hallucination: **1.7** (최고, 환각 가장 적음)
- Diversity: **1.5** (최고)

**결론**:
- **히스토리와 추론은 상호보완적 이점 제공**
- **히스토리**: 시간적 및 인과적 구조 보존
- **추론**: 깊이 및 사실적 신뢰성 향상
- **검증**: 경험 모델은 구조화되고 추론 주도 방식으로 작동해야 궤적의 다양성과 충실성 모두 유지

#### 3.3.3 경험 모델 백본 및 오프라인 데이터 (Figure 5)

**실험 변수**:
- **오프라인 학습 데이터 크기**: 2K, 10K, 20K, 40K 단계
- **모델 백본**: Llama-3.1-8B, Llama-3.2-3B, WebDreamer

**결과 (성공률 %)**:

**WebShop**:
| 데이터 크기 | Llama-3.1-8B | Llama-3.2-3B | WebDreamer |
|------------|--------------|--------------|------------|
| 2K | ~45% | ~40% | ~48% |
| 10K | **~55%** | ~48% | ~50% |
| 20K | ~58% | **~55%** | ~51% |
| 40K | **~60%** | ~58% | ~52% |

**WebArena**:
| 데이터 크기 | Llama-3.1-8B | Llama-3.2-3B | WebDreamer |
|------------|--------------|--------------|------------|
| 2K | ~9% | ~7% | **~13%** |
| 10K | **~11%** | ~9% | ~13% |
| 20K | **~12%** | ~10% | ~13% |
| 40K | **~13%** | ~11% | ~13% |

**주요 관찰**:

**1) 경험 모델은 매우 데이터 효율적**:
- **매우 제한된 오프라인 샘플 (2K-10K)로도 경쟁력 있는 성능**
- WebShop 예시: Llama-3.1-8B은 10K 샘플로 50% 성공률 초과
- **의미**: 대규모 오프라인 데이터셋이 효과적인 경험 합성에 엄격히 필요하지 않음

**2) 작은 백본도 실행 가능**:
- Llama-3.2-3B가 8B 모델보다 성능 낮지만, **더 많은 데이터로 꾸준히 개선**
- 20K 샘플로 WebShop에서 ~55% 달성
- **의미**: 계산 자원이 제약될 때 경량 모델도 실용적 경험 생성기로 사용 가능

**3) 극도로 낮은 데이터 체제에서 사전학습된 월드 지식이 특히 가치 있음**:
- WebArena에서 **WebDreamer (파인튜닝된 웹 월드 모델)가 작은 데이터 스케일에서 Llama-3.1-8B 초과** (~13% vs ~9% at 2K)
- 오프라인 샘플 수 증가하면 격차 수렴
- **의미**:
  - 웹 특정 사전학습이 초기 이점 제공
  - 엄격한 전제조건은 아님
  - 대규모 경험 학습이 사전학습 모델의 성능 격차 메울 수 있음

#### 3.3.4 케이스 스터디: DreamGym의 합성 경험 (Figure 6)

**예시 시나리오 (WebArena)**:

**합성 태스크**:
> "Find out the message Jack provided in the first commit of Apr 2023"

**State 0 (초기 상태)**:
```
[472] card: "Change Log" summary tile.
[463] statbadge: "Total Commits" — 1,980.
[458] pillnav: Tabs {Overview, Change Log}.
[512] button: "Open Change Log".
[539] content: Latest commit by Jack Thompson ("Apr 02, 2023 11:30pm UTC"). ...
```

**Agent Action 0**: `Click(463)` (Total Commits 버튼 클릭)

**Experience Model Reasoning**:
```
<think>
The agent clicks the "Total Commits" button on the overview page, which is the
intended action. I should transit to a successful state that displays a list of
commits grouped by date, including entries from April 2023, which will let the
agent further inspect and ...
</think>
```

**State 1 (추론 후 생성)**:
```
[847] list: List grouping commit history ...
[1144] listitem: Commit on Apr 2, 2023, by ...
[1144] listitem: Commit on Apr 3, 2023, by ...
[1163] listitem: Commit on Apr 7, 2023, by ...
[1241] listitem: Commit on Apr 7, 2023, by ...
[1259] listitem: Commit on Apr 8, 2023, by ...
```

**Agent Action 1**: `Click(1144)` (첫 번째 4월 커밋 항목 클릭)

**Experience Model Reasoning**:
```
<think>
The agent clicks [1144], the first April commit entry (Apr 2, 2023) to open its
details. I should transit to a commit-detail state that exposes the author,
timestamp, and commit message so the agent can extract the answer.
</think>
```

**State 2 (추론 후 생성)**:
```
[1500] pane: Commit details view for Apr 2, 2023
[1501] header: Commit by Jack Thompson
[1502] time: "Apr 02, 2023 11:30pm UTC"
[1503] message: "Add API migration notes"
[1504] meta: SHA a1c2e3f, branch main, 3 files changed (12 insertions, 2 deletions)
[1505] nav: Buttons {Back to list, Copy SHA, ...}
```

**관찰**:
1. **명시적 CoT 추론**: 각 단계에서 에이전트 행동, 태스크 명령어, 상호작용 히스토리 통합
2. **일관된 상태 전이**: 행동을 일관되게 근거화하고 결과 정확히 반영
3. **정보가 풍부한 상태**: 깨끗하고 행동 가능한 요소 제공
4. **태스크 정렬**: 상태가 태스크 목표로 진전

## 4. 시스템 아키텍처 및 전체 워크플로우 (Figure 2)

### 4.1 전체 시스템 다이어그램

```
                    ┌─────────────────────────┐
                    │  Task Instructions      │
                    │  (Seed + Generated)     │
                    └────────┬────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Curriculum Task    │◄──┐
                   │  Generator          │   │
                   └─────────┬───────────┘   │
                             │                │
                             │ Task Value     │
                             │ Estimation     │
                             │ (Reward        │
                             │  Entropy)      │
                             │                │
              ┌──────────────▼─────────────┐  │
              │  Reasoning Experience      │  │
              │  Model (Mexp)              │  │
              └──────┬───────────▲─────────┘  │
                     │           │            │
        actions      │           │ retrieve   │ challenging
                     │           │            │ task variations
              ┌──────▼───────────┴─────────┐  │
              │  Experience Replay Buffer  │──┘
              │  (Offline + Synthetic)     │
              └──────┬─────────────────────┘
                     │ update
                     │
              ┌──────▼───────────┐
              │  Agent Policy    │
              │  (πθ)            │
              └──────┬───────────┘
                     │
                     │ actions
                     │
              ┌──────▼───────────┐
              │  Informative     │
              │  States (st)     │
              │  + Rewards (rt)  │
              │  + CoT Reasoning │
              └──────────────────┘
                     │
              ┌──────▼───────────┐
              │  Agent loop      │
              │  terminated?     │
              │  Done: True/False│
              │  Success: T/F    │
              └──────────────────┘
                     │
         ┌───────────▼───────────┐
         │ Scalable LLM Serving  │
         │ Infrastructure        │
         └───────────────────────┘
```

### 4.2 워크플로우 단계

**Phase 1: 초기화**
1. 시드 태스크 명령어 세트로 시작
2. 오프라인 데이터로 경험 리플레이 버퍼 초기화
3. 오프라인 데이터로 추론 경험 모델 $M_{\text{exp}}$ 학습

**Phase 2: 합성 경험 생성** (각 반복마다)
1. 에이전트 정책 $\pi_\theta$가 현재 상태 $s_t$에서 행동 $a_t$ 선택
2. 경험 모델 $M_{\text{exp}}$가:
   - 리플레이 버퍼에서 유사한 경험 $\{d_j\}_{j=1}^k$ 검색
   - 상호작용 히스토리 $H_t$, 태스크 $\tau$, 행동 $a_t$ 통합
   - CoT 추론 $R_t$ 생성
   - 다음 상태 $s_{t+1}$과 보상 $r_{t+1}$ 예측
3. 에피소드 종료까지 반복

**Phase 3: 정책 학습**
1. 수집된 궤적으로 배치 구성
2. RL 알고리즘 (GRPO/PPO) 적용하여 $\pi_\theta$ 업데이트
3. 새로운 합성 경험으로 리플레이 버퍼 업데이트

**Phase 4: 커리큘럼 태스크 생성**
1. 각 태스크의 보상 엔트로피 $V_\tau$ 계산
2. 높은 엔트로피 태스크 (도전적이지만 실행 가능) 선택
3. $M_{\text{task}}$로 새 태스크 변형 생성
4. 비율 $\lambda$로 학습 세트에 추가

**Phase 5: 반복**
- 수렴 또는 예산 도달까지 Phase 2-4 반복

**Phase 6 (선택적): Sim-to-Real 전이**
1. 합성 사전학습 완료
2. 실제 환경으로 정책 전이
3. 소규모 RL 파인튜닝 (훨씬 적은 실제 데이터)

## 5. 이론적 분석 (부록 B)

### 5.1 Lemma 1: 다중 턴 경험 합성 오차 경계

**목표**: 실제와 합성 환경 간 정책 가치 차이 경계 설정

**정의**:

$$
\epsilon_R = \sup_{s,a} |R(s,a) - \hat{R}(s,a)|
$$

$$
\epsilon_P = \sup_{s,a} \text{TV}(P(\cdot|s,a), \hat{P}(\cdot|s,a))
$$

**보조정리**:

$$
|J_{\mathcal{M}}(\pi) - J_{\hat{\mathcal{M}}}(\pi)| \le \Delta_{\text{model}} := \frac{\epsilon_R}{1-\gamma} + \frac{2\gamma R_{\max}}{(1-\gamma)^2} \epsilon_P
$$

**증명 스케치**:

1. 실제와 합성 벨만 연산자 차이 비교
2. 반복 적용으로 가치 함수 차이 경계 설정
3. 수축 매핑 이용하여 점근적 경계 도출

**의미**:
- 실제-합성 환경 간 에이전트 성능 격차는 **보상 정확도 및 도메인 일관성 오차에만 의존**
- 상태 재구성 오류 같은 엄격한 충실성 지표에 의존하지 않음
- DreamGym의 설계 철학 검증: **학습 관련 신호 보존이 중요, 원시 상태 복제는 불필요**

### 5.2 Theorem 1 증명 구조

**1단계**: 실제 환경 정책 개선을 합성 환경 통해 분해

$$
J_{\mathcal{M}}(\pi') - J_{\mathcal{M}}(\pi) = [J_{\hat{\mathcal{M}}}(\pi') - J_{\hat{\mathcal{M}}}(\pi)] + [J_{\mathcal{M}}(\pi') - J_{\hat{\mathcal{M}}}(\pi')] - [J_{\mathcal{M}}(\pi) - J_{\hat{\mathcal{M}}}(\pi)]
$$

**2단계**: Lemma 1 적용

각 정책 불일치 항 $|J_{\mathcal{M}}(\cdot) - J_{\hat{\mathcal{M}}}(\cdot)| \le \Delta_{\text{model}}$

따라서:

$$
J_{\mathcal{M}}(\pi') - J_{\mathcal{M}}(\pi) \ge [J_{\hat{\mathcal{M}}}(\pi') - J_{\hat{\mathcal{M}}}(\pi)] - 2\Delta_{\text{model}}
$$

**3단계**: 합성 환경 내 개선 하한

표준 신뢰 영역 경계 (Schulman et al., 2015):

$$
J_{\hat{\mathcal{M}}}(\pi') - J_{\hat{\mathcal{M}}}(\pi) \ge \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_\pi^{\hat{\mathcal{M}}}, a \sim \pi'(\cdot|s)} [A_\pi^{\hat{\mathcal{M}}}(s,a)] - \frac{4\gamma}{(1-\gamma)^2} V_{\max} \delta
$$

**4단계**: 결합하여 Theorem 1 도출 ■

### 5.3 실용적 함의

**정리가 말하는 것**:
1. ✅ **합성 대리 이득**이 두 페널티 합보다 크면 → 실제 환경에서 정책 개선 보장
2. ✅ 오차 항 ($\epsilon_R$, $\epsilon_P$)이 작으면 → 합성 학습이 실제 개선으로 효과적 전이
3. ✅ 명시적 추론과 리플레이 버퍼로 두 오차 모두 작게 유지 가능

**DreamGym이 이를 달성하는 방법**:
- **$\epsilon_R$ 감소**: 추론 기반 보상 예측 → 일관되고 인과적으로 근거있는 피드백
- **$\epsilon_P$ 감소**: 리플레이 버퍼에서 유사한 경험 검색 → 도메인 일관 상태 전이
- **경험적 검증**: Table 1에서 순수 합성 학습이 실제 RL과 동등하거나 더 나은 성능

## 6. 구현 세부사항 및 프롬프트 (부록 A, C)

### 6.1 경험 모델 추론 트레이스 주석 프롬프트 (WebShop 예시)

**시스템 프롬프트**:
> You are an expert in web navigation and e-commerce environments, specializing in providing actionable guidance for state transitions of an experience model that simulates the environment dynamics.

**사용자 프롬프트 구조**:
1. **입력**:
   - 태스크 명령어
   - 성공 플래그
   - 궤적 $\{(s_i, a_i)\}_{i=1}^N$

2. **출력 요구사항**:
   - **Task Tutorial**: 환경이 태스크 명령어 하에서 단계별로 어떻게 전이해야 하는지 고수준 안내
   - **State Transition Plans**: 각 단계에 대해:
     - 에이전트 행동이 성공/실패할 가능성 분석
     - 현재 상태와 행동이 주어졌을 때 환경이 어떻게 전이해야 하는지 간결한 추론 트레이스

3. **응답 형식** (JSON):
```json
{
  "task_tutorial": {
    "Overall Plan": "고수준 안내 (1문장)",
    "Success Mode": "성공을 위한 중요 단계 (1문장)",
    "Failure Mode": "피해야 할 전형적 실패 모드 (1문장)"
  },
  "state_transitions": [
    {
      "step_id": 0,
      "transition_plan": "행동 분석 및 환경 응답 계획"
    },
    ...
  ]
}
```

**참고**: 부록 C에 WebShop, ALFWorld, WebArena의 전체 프롬프트 포함

### 6.2 태스크 변형 생성 프롬프트 (WebShop 예시)

**목표**: 원래 태스크의 도전적이지만 실행 가능한 변형 선택

**입력**:
- 원래 태스크
- 제품 정보 (카테고리, 이름, 속성)
- 후보 변형들

**선택 기준**:
1. **도전적이지만 실행 가능**: 원래보다 구체적이거나 복잡하지만 달성 가능
2. **고품질**: 명확, 문법적으로 올바르고, 현실적
3. **의미있는 변형**: 사소한 변경이 아닌 의미있게 다름
4. **현실적**: 속성, 옵션, 가격 조합이 제품 카테고리에 맞음

**출력**:
```
SELECTION: [번호]
REASONING: [1-2문장 설명]
```

### 6.3 에이전트 프롬프트 템플릿 (WebShop 예시)

**구조**:
1. **역할 설명**: "You are an expert autonomous agent operating in the WebShop e-commerce environment."
2. **태스크**: 수행할 구체적 쇼핑 태스크
3. **컨텍스트**:
   - 이전 단계 수
   - 최근 히스토리 (관찰 + 행동)
4. **현재 관찰**: 현재 상태
5. **허용 행동**: 가능한 행동 목록
6. **출력 형식**:
```xml
<think>
단계별 추론...
</think>
<action>
선택한 행동
</action>
```

### 6.4 WebArena AX-tree 상태 매핑 프롬프트

**목적**: 웹페이지 관찰을 정제된 부분집합으로 추출

**입력**:
- 사용자 명령어
- 상호작용 히스토리
- AXTree 관찰 (현재 시간 단계)

**출력**:
```xml
<reasoning>
<content_description>현재 페이지 상태 고수준 설명 (최대 2문장)</content_description>
<agent_progress>지금까지 달성한 것 (최대 1문장)</agent_progress>
<next_action_analysis>다음에 무엇을 해야 하고 왜 (최대 1문장)</next_action_analysis>
</reasoning>

<extraction>
[ELEMENT_ID] TYPE: DESCRIPTION
[ELEMENT_ID] TYPE: DESCRIPTION
...
(3-10개의 가장 관련있는 행동 가능 요소만)
</extraction>
```

**특징**:
- AXTree 구조 변경 금지
- 요소 ID 정확히 추출
- 차트/테이블 전체 추출
- 의미 없는 식별자 제외
- 행동 가능한 요소만 추출

### 6.5 행동 공간 (WebArena)

13가지 행동 유형:
1. `noop(wait_ms)`: 아무것도 안함
2. `send_msg_to_user(text)`: 사용자에게 메시지 전송
3. `scroll(delta_x, delta_y)`: 스크롤
4. `fill(bid, value)`: 양식 필드 채우기
5. `select_option(bid, options)`: 옵션 선택
6. `click(bid, button, modifiers)`: 요소 클릭
7. `dblclick(bid, button, modifiers)`: 더블 클릭
8. `hover(bid)`: 호버
9. `press(bid, key_comb)`: 키 조합 누르기
10. `focus(bid)`: 포커스
11. `clear(bid)`: 입력 필드 비우기
12. `drag_and_drop(from_bid, to_bid)`: 드래그 앤 드롭
13. `upload_file(bid, file)`: 파일 업로드

### 6.6 경험 모델 품질 판정 프롬프트 (부록 C.4)

**입력**:
- 현재 상태 (행동 전)
- 에이전트 행동
- 예측된 다음 상태 (행동 후)

**평가 루브릭** (각 0/1/2):

1. **Causal State Consistency (인과적 상태 일관성)**:
   - 2: 일관되고 행동에 적절, 모든 예상 업데이트 나타남
   - 1: 대부분 일관되지만 사소한 논리적/의미적 격차
   - 0: 비일관적이거나 행동과 인과적으로 연결 안됨

2. **Diversity & State Variation (다양성 & 상태 변화)**:
   - 2: 실질적이고 일관된 차이 (새 결과, 업데이트된 필터, 변경된 세부사항)
   - 1: 최소한 또는 피상적 변화
   - 0: 의미있는 변화 없음, 또는 비일관적 점프

3. **Informativeness (정보성)**:
   - 2: 상세하고, 관련있고, 내부적으로 일관된 정보
   - 1: 일부 유용한 세부사항, 하지만 희소하거나 부분적으로 비일관적
   - 0: 정보 없음, 관련없음, 또는 비일관적

4. **Hallucination & Failure Feedback (환각 & 실패 피드백)**:
   - 2: 실패 또는 성공을 적절히 신호, 환각 없음
   - 1: 실패 처리가 부분적/애매함
   - 0: 성공을 환각하거나 실패 무시

**출력 형식** (JSON):
```json
{
  "rubric_scores": {
    "causal_consistency": 0|1|2,
    "diversity": 0|1|2,
    "informativeness": 0|1|2,
    "hallucination": 0|1|2
  }
}
```

## 7. 한계점 및 향후 연구

### 7.1 현재의 한계

**1) 단일 환경 학습 설정에 주로 초점**:
- 개별 에이전트 시나리오에 DreamGym 적용
- 환경 간 지식 전이 제한적

**2) 환경 모델 의존성**:
- 여전히 정확한 환경 모델 필요
- 복잡한 환경에서 모델 학습 어려움
- 모델 부정확 시 합성 경험 품질 저하

**3) 초기 데이터 요구사항**:
- 오프라인 데이터로 환경 모델 학습 필요
- Zero-shot 시나리오에서 제한적

**4) 계산 비용**:
- 대규모 LLM 사용으로 인한 추론 비용
- 환경 모델 학습의 초기 오버헤드

**5) 도메인 격차 한계**:
- 웹 환경 ↔ ALFWorld 같은 큰 도메인 격차에서 전이 실패
- 현재 메타-표현의 범용성 한계

### 7.2 향후 연구 방향

**1) 범용 월드 모델 (Universal World Model)**:
- 여러 환경 모델을 통합하는 단일 월드 모델
- 환경 간 지식 전이 가능
- **비전**: 순수 합성 경험으로 기초 에이전트 모델 스케일링
  - 새 환경에 제로샷 적응 가능

**2) 환경 모델 개선**:
- 더 정확하고 효율적인 모델
- 불확실성 추정 및 활용
- 모델 불확실성 명시적 처리

**3) 온라인 학습 통합**:
- 합성 경험과 실제 경험의 최적 조합
- 적응적 데이터 수집 전략
- 실시간 환경 모델 업데이트

**4) 멀티모달 확장**:
- 시각, 언어, 행동의 통합
- 실제 로봇 시스템으로의 적용
- 다중 감각 입력 처리

**5) 이론적 보장 강화**:
- 더 일반적인 수렴 조건
- 안전성 보장
- 분포 이동(distribution shift) 하에서 강건성

**6) 효율성 개선**:
- 근사 추론 기법
- 병렬화 및 최적화
- 캐싱 전략

## 8. 결론 및 의의

### 8.1 주요 기여

**1) 방법론적 기여**:
- 언어 에이전트 RL의 **최초 통합 경험 합성 프레임워크**
- 추론 기반 경험 모델 (raw 재구성 대신 추상화)
- 보상 엔트로피 기반 적응형 커리큘럼 학습
- 리플레이 버퍼와 통합된 경험 생성

**2) 이론적 기여**:
- 합성 경험을 통한 실제 환경 정책 개선 보장 (Theorem 1)
- 학습 중심 신호 보존의 충분성 입증
- 완벽한 환경 복제 불필요성 증명

**3) 실용적 기여**:
- WebArena (non-RL-ready): **30% 이상 절대적 개선**
- WebShop/ALFWorld: 순수 합성으로 **실제 RL과 동등**
- DreamGym-S2R: 실제 데이터 **10% 미만**으로 **40% 이상 개선**
- 학습 비용 **1/3 ~ 1/5 감소**
- 오픈소스 프레임워크 제공 (예정)

**4) 패러다임적 기여**:
> "RL for LLM 에이전트의 핵심 병목은 상호작용 데이터의 **품질과 구조**에 있다. 환경을 단순 시뮬레이터가 아닌 **구조화되고 추론이 풍부한 경험의 생성기**로 취급함으로써, DreamGym은 에이전트를 위한 더 확장 가능하고, 샘플 효율적이며, 일반화 가능한 RL을 가능하게 한다."

### 8.2 광범위한 영향

**1) 연구 커뮤니티**:
- RL-ready 아닌 환경에서도 RL 연구 가능
- 계산 자원 제약이 있는 연구자에게 접근성 향상
- 새로운 연구 방향 개척: 추론 기반 월드 모델링

**2) 산업 응용**:
- 비용 효과적인 에이전트 개발
- 안전한 탐색 (합성 환경)
- 빠른 프로토타이핑
- 위험한 시나리오에서 안전한 학습

**3) 에이전트 AI의 미래**:
- 범용 에이전트로 가는 길
- 제로샷 적응 가능성
- 지속적 학습 및 개선
- 인간 개입 최소화

### 8.3 DreamGym의 철학

**핵심 통찰**:
> "에이전트는 픽셀 완벽한 시뮬레이션이 필요한 것이 아니라, **인과적으로 일관되고, 다양하며, 정보가 풍부한 경험**이 필요하다."

**설계 원칙**:
1. **추상화 > 재구성**: 원시 상태 복제보다 메타-표현 학습
2. **추론 > 패턴 매칭**: 논리적 일관성이 통계적 패턴보다 중요
3. **다양성 > 정확성**: 완벽한 소수보다 충분히 좋은 다수
4. **커리큘럼 > 랜덤**: 적응적 난이도가 무작위 탐색보다 효과적

**AlphaGo와의 비교**:
- AlphaGo: 자기 대국(self-play)으로 경험 생성
- DreamGym: 환경 모델로 경험 합성
- 공통점: **실제 경험 의존도 감소로 스케일링 달성**

### 8.4 마지막 말

DreamGym은 단순히 비싼 실제 롤아웃의 대체가 아니라, **LLM 에이전트를 위한 RL의 근본적 재구상**입니다.

- ✅ **RL-ready 아닌 환경을 RL-ready로**: WebArena에서 입증
- ✅ **RL-ready 환경을 더 효율적으로**: WebShop/ALFWorld에서 입증
- ✅ **Sim-to-Real 전이 가능**: DreamGym-S2R에서 입증
- ✅ **이론적으로 보장**: Theorem 1로 뒷받침
- ✅ **실용적으로 검증**: 여러 환경 및 백본에서 일관된 개선

**다음 단계**: 범용 월드 모델로 확장하여, **순수 합성 경험으로 기초 에이전트 모델을 스케일링하고 새 환경에 제로샷 적응 가능한 시스템** 구축

---

## 참고문헌

### 핵심 논문
- **DreamGym**: arXiv:2511.03773
- **AlphaGo**: Silver et al., 2016
- **PPO**: Schulman et al., 2017
- **GRPO**: Shao et al., 2024 (DeepSeekMath)

### 벤치마크
- **WebArena**: Zhou et al., WebArena-Lite (Liu et al., 2024)
- **WebShop**: Yao et al., 2022
- **ALFWorld**: Shridhar et al., TextWorld (Côté et al., 2018)

### 관련 연구
- **Dreamer**: Hafner et al., 2020; Ha and Schmidhuber, 2018
- **WebDreamer**: Gu et al., 2024
- **UI-Simulator**: Wang et al., 2025b
- **AgentSynth**: Xie et al., 2025
- **SCA**: Zhou et al., 2025

### LLM 모델
- **Llama-3**: Grattafiori et al., 2024
- **Qwen-2.5**: Team, 2024
- **GPT-4o**: Hurst et al., 2024

### 전체 참고문헌
논문 본문 끝부분 참조 (50+ 인용)
