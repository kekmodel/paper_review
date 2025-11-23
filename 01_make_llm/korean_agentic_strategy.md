# 한국어 Agentic AI 구축 전략: 4명 팀의 실전 가이드

## 🎯 Executive Summary

**목표:** 한국어로 복잡한 multi-step tasks를 자율적으로 수행하는 agentic AI 모델 구축

**왜 한국어 Agentic인가?**
```
✅ 시장 기회
- 한국어 agentic AI는 거의 전무 (Blue Ocean)
- GPT-4, Claude도 한국어 tool-use는 약함
- 국내 기업/정부 수요 폭발적 증가

✅ 기술적 실현 가능성
- Kimi K2가 증명: Agentic은 post-training으로 가능
- 4명 팀 규모에 최적 (pre-training 불필요)
- Verifiable rewards로 자동 평가 가능

✅ 차별화 전략
- 한국 특화 tool ecosystem (법률, 행정, 금융)
- 한국어 multi-turn reasoning
- 국내 서비스 통합 (네이버, 카카오, 공공데이터)
```

**핵심 전략:**
```
Base Model: Polyglot-Ko-12.8B (또는 Llama-3-8B + Korean adaptation)
           ↓
Tool Calling 데이터 생성 (영어 번역 + 한국 특화)
           ↓
Multi-stage RL Training (Verifiable Rewards Gym)
           ↓
자체 벤치마크로 평가 및 iteration
           ↓
6개월 내 production-ready model
```

**예상 성과:**
- Korean Tool-use Benchmark: 70-80% (기존 모델 ~40%)
- Korean Multi-step Reasoning: 60-70% (기존 ~30%)
- 실사용 케이스: 법률 문서 분석, 행정 절차 자동화, 금융 계산

---

## 1️⃣ 한국어 Agentic의 기회

### 1.1 왜 Agentic인가?

**Agentic AI의 정의:**
```
단순 질의응답 (GPT-3 스타일):
User: "오늘 날씨 어때?"
AI: "서울은 맑고 기온 15도입니다."

Agentic AI (Kimi K2 스타일):
User: "다음 주 출장 일정 잡아줘. 부산이고 2박3일이야."
AI:
  1. [Search] 사용자 캘린더 확인
  2. [Think] 다음 주 월-금 중 2박3일 가능 구간 찾기
  3. [Search] 부산 날씨 예보
  4. [Search] KTX 예약 가능 시간
  5. [Python] 최적 일정 계산
  6. [Search] 부산 호텔 예약 (회사 출장 정책 고려)
  7. [Think] 일정 충돌 확인
  8. [API Call] 캘린더에 일정 등록
  9. [API Call] KTX 예약
  10. [API Call] 호텔 예약
  → "10월 15-17일 일정 잡았습니다. KTX는 오전 7시, 호텔은 해운대 ○○호텔입니다."
```

**핵심 차이:**
- **Tool Use**: 외부 API, 데이터베이스, 계산 도구 사용
- **Multi-step Planning**: 복잡한 작업을 여러 단계로 분해
- **Autonomous Execution**: 사람 개입 없이 자율적 실행
- **Verification**: 각 단계 결과 검증 후 다음 단계 진행

---

### 1.2 한국어 Agentic의 시장 가치

**현재 상황:**
```
영어 Agentic AI:
- Kimi K2: TAU-Bench 66.1%, SWE-bench 65.8%
- GPT-4: Tool-use 강력, 광범위한 API 지원
- Claude: Artifacts, multi-step reasoning

한국어 Agentic AI:
- GPT-4: 한국어 tool-use 40-50% 수준
- Claude: 한국어 multi-step reasoning 약함
- 국내 모델: Tool-use 거의 전무
→ 거대한 gap!
```

**시장 수요 (국내):**

1. **법률/행정**
   - 법률 문서 분석 + 판례 검색 + 보고서 생성
   - 행정 절차 자동화 (민원, 허가, 신고)
   - 수요: 법무법인, 로펌, 정부기관

2. **금융**
   - 재무제표 분석 + 리스크 계산 + 리포트 작성
   - 투자 포트폴리오 최적화
   - 수요: 은행, 증권사, 핀테크

3. **고객 지원**
   - 복잡한 문의 처리 (주문 조회 + 환불 처리 + 재주문)
   - Multi-turn 상담 자동화
   - 수요: 이커머스, 통신사, 게임사

4. **연구/개발**
   - 문헌 조사 + 데이터 분석 + 리포트 작성
   - 실험 설계 + 결과 해석
   - 수요: 연구소, 대학, 제약사

**시장 규모:**
```
국내 AI 시장: ~$2B (2025)
Agentic AI 잠재 시장: ~$500M (25%)
→ 한국어 특화 agentic AI가 10% 점유해도 $50M
```

---

### 1.3 왜 4명 팀이 가능한가?

**Kimi K2가 증명한 것:**
```
Agentic capability ≠ Pre-training
Agentic capability = Post-training (RL + Data)
```

**근거:**
1. **Kimi K2의 Base Model**은 일반 MoE
2. **Agentic Data Synthesis Pipeline**으로 데이터 생성
3. **Verifiable Rewards Gym**으로 RL 훈련
4. → TAU-Bench 66.1% (SOTA)

**4명 팀의 강점:**
```
✅ Tool Ecosystem을 특정 도메인에 집중 (법률/금융/행정)
✅ 한국어 데이터만 생성 (영어보다 10배 적은 양)
✅ RL 훈련은 verification 기반 (human feedback 불필요)
✅ 빠른 iteration (1-2주 단위)
```

**비교:**
| 항목 | 대기업 (범용 agentic) | 4명 팀 (한국어 특화) |
|------|---------------------|---------------------|
| Tool 수 | 20,000+ | 500-1,000 (특화) |
| 훈련 데이터 | 1M+ samples | 50K-100K (고품질) |
| 도메인 | 범용 | 법률/금융/행정 |
| 언어 | 영어 중심 | 한국어 네이티브 |
| 시장 | 글로벌 경쟁 | Blue Ocean |

---

## 2️⃣ 현실과 도전 과제

### 2.1 한국어 Agentic의 어려움

**Challenge 1: 데이터 부족**
```
영어 Tool-calling 데이터:
- ToolBench: 16,000+ APIs
- Gorilla: 1,600+ API calls
- Berkeley Function Calling: 2,000+ samples

한국어 Tool-calling 데이터:
- 공개 데이터셋: 거의 없음
- 기존 모델: Tool-use capability 약함
→ Ground truth가 없음!
```

**Challenge 2: 벤치마크 부족**
```
영어 Agentic Benchmarks:
- TAU-Bench: Tool-augmented user tasks
- SWE-bench: Software engineering
- HLE: Human-like evaluation
- WebArena: Web navigation

한국어 Agentic Benchmarks:
- 존재하지 않음
→ 평가 기준 자체를 만들어야 함!
```

**Challenge 3: Tool Ecosystem 미흡**
```
영어 Tools:
- 3,000+ MCP tools (Kimi K2)
- OpenAI function calling
- LangChain 방대한 tool library

한국어 Tools:
- 네이버/카카오 API (제한적)
- 공공데이터 API (복잡한 인증)
- 금융/법률 API (접근 어려움)
→ Tool을 직접 구축해야 함!
```

**Challenge 4: Multi-turn Complexity**
```
한국어 특성:
- 높임말/반말 혼용
- 맥락 의존적 대화
- 생략이 많음 ("그거", "저기", "그래서")
→ Multi-turn에서 context 추적 어려움
```

---

### 2.2 우리가 극복 가능한 이유

**극복 전략 1: Synthetic Data Generation**
```
영어 tool-calling 데이터를 기반으로:
1. GPT-4로 한국어 번역 (고품질)
2. 한국 특화 시나리오 생성 (법률, 금융)
3. Self-improving cycle로 품질 향상

Cost: $5K-$10K
Time: 2-3 weeks
→ 50K high-quality samples 생성 가능
```

**극복 전략 2: 자체 벤치마크 제작**
```
Korean Agentic Benchmark (KAB):
- 100-200 hand-crafted tasks
- Verifiable outcomes (정답 확인 가능)
- Domain-specific (법률, 금융, 행정)

Cost: $2K-$5K (크라우드소싱)
Time: 2-3 weeks
→ 평가 기준 확립
```

**극복 전략 3: 최소 Tool Ecosystem**
```
Phase 1 Tools (20-30개):
- 계산: 수학, 금융 계산기
- 검색: 네이버 검색, 위키피디아
- 데이터: 공공데이터 API (날씨, 주가)
- 실행: Python sandbox

Phase 2 Tools (50-100개):
- 법률: 판례 검색, 법령 조회
- 금융: 재무제표, 주가 분석
- 행정: 민원, 신고

Cost: $10K-$20K (API 통합)
Time: 1-2 months
→ 핵심 도메인 커버
```

**극복 전략 4: Curriculum Learning**
```
Stage 1: Single-turn tool-use
"오늘 날씨 알려줘" → [API Call] → "서울 15도"

Stage 2: 2-3 step tasks
"서울 날씨 좋으면 카페 추천해줘"
→ [Weather API] → [Think] → [Search Cafes] → 추천

Stage 3: Complex multi-step (5-10 steps)
"다음 주 부산 출장 일정 잡아줘"
→ 10+ steps (캘린더, KTX, 호텔, 예산 확인)

→ 점진적 난이도 증가로 학습 안정성 확보
```

---

## 3️⃣ Kimi K2 스타일 접근법

### 3.1 Kimi K2의 Agentic 성공 요인

**Kimi K2가 SOTA 달성한 방법:**

**1. Agentic Data Synthesis Pipeline**
```
Step 1: Domain Evolution
- 20,000+ synthetic tools 생성
- 다양한 도메인 (coding, math, search, API)

Step 2: Agent Diversification
- 다양한 system prompts
- 다양한 tool combinations
- 다양한 user personas

Step 3: Rubric-based Task Generation
- 명확한 success criteria
- Verifiable outcomes
- Multi-turn trajectories

Step 4: Hybrid Execution
- Simulated + Real sandboxes
- 실제 API 호출 + 시뮬레이션
```

**2. Verifiable Rewards Gym**
```
자동 검증 가능한 작업들:
✅ Math/STEM: 정답 확인
✅ Coding: Unit test 실행
✅ Tool-calling: API 결과 검증
✅ Search: Information retrieval accuracy

→ Human feedback 불필요!
```

**3. Self-Critique Mechanism**
```python
def self_critique(model, task, output_a, output_b):
    """
    모델이 자신의 두 출력을 비교하여 더 나은 것 선택
    """
    prompt = f"""
    Task: {task}

    Output A: {output_a}
    Output B: {output_b}

    Which output better completes the task? Consider:
    1. Correctness of tool calls
    2. Efficiency (fewer steps better)
    3. Completeness (all requirements met)
    4. Reasoning coherence

    Answer: [A/B], Reason: [...]
    """

    critique = model.generate(prompt)
    winner = parse_winner(critique)  # 'A' or 'B'

    return winner
```

**4. Multi-stage RL Training**
```
Stage 1: Simple Tool-use (1-2 steps)
- Single API calls
- Basic calculations
→ RL with verifiable rewards

Stage 2: Multi-step Tasks (3-5 steps)
- Chained tool calls
- Conditional logic
→ RL with process rewards

Stage 3: Complex Agentic Tasks (5-10+ steps)
- Planning + Execution
- Error recovery
→ RL with outcome + process rewards
```

---

### 3.2 한국어 적용 전략

**우리의 Adaptation:**

**Phase 1: Tool Ecosystem 구축 (Week 1-4)**

```python
"""
Korean Tool Ecosystem
최소 20개 핵심 tools로 시작
"""

# Category 1: Computation (5 tools)
tools = {
    "calculator": "수학 계산",
    "financial_calculator": "금융 계산 (이자, 세금)",
    "unit_converter": "단위 변환",
    "date_calculator": "날짜 계산",
    "statistics": "기초 통계",
}

# Category 2: Information Retrieval (5 tools)
tools.update({
    "naver_search": "네이버 검색",
    "wikipedia_kr": "한국어 위키피디아",
    "news_search": "뉴스 검색",
    "law_search": "법령/판례 검색",
    "company_info": "기업 정보 조회",
})

# Category 3: Data APIs (5 tools)
tools.update({
    "weather_api": "날씨 정보",
    "stock_api": "주가 정보",
    "exchange_rate": "환율 정보",
    "public_data": "공공데이터",
    "map_api": "지도/위치 정보",
})

# Category 4: Execution (5 tools)
tools.update({
    "python_sandbox": "Python 코드 실행",
    "sql_query": "SQL 쿼리",
    "text_analysis": "텍스트 분석",
    "pdf_parser": "PDF 파싱",
    "json_processor": "JSON 처리",
})

# 각 tool은 다음 구조:
class Tool:
    def __init__(self, name, description, parameters, returns):
        self.name = name
        self.description = description  # 한국어 설명
        self.parameters = parameters    # JSON schema
        self.returns = returns          # Return type

    def execute(self, **kwargs):
        # 실제 실행 로직
        pass

    def verify(self, input, output):
        # 출력 검증 로직
        pass

# Example: Calculator tool
calculator = Tool(
    name="calculator",
    description="수학 계산을 수행합니다. 사칙연산, 제곱, 제곱근, 삼각함수 등을 지원합니다.",
    parameters={
        "expression": {
            "type": "string",
            "description": "계산할 수식 (예: '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')"
        }
    },
    returns={"type": "number", "description": "계산 결과"},
)

def calculator_execute(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, safe_math_functions)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculator_verify(expression, output):
    # Re-compute and compare
    expected = eval(expression, {"__builtins__": {}}, safe_math_functions)
    return abs(output["result"] - expected) < 1e-6
```

**Phase 2: 한국어 Tool-calling 데이터 생성 (Week 5-8)**

```python
"""
Synthetic Data Generation Pipeline
영어 → 한국어 adaptation + 한국 특화 생성
"""

# Method 1: Translation (10K samples)
def translate_english_toolcalling():
    """
    영어 tool-calling datasets 번역
    - ToolBench, Gorilla, Berkeley Function Calling
    """
    english_samples = load_toolbench()  # 16K samples

    korean_samples = []
    for sample in english_samples[:10000]:  # 10K subset
        # GPT-4로 고품질 번역
        korean_task = translate_with_gpt4(
            sample["task"],
            style="natural_korean"  # 자연스러운 한국어
        )

        # Tool calls는 그대로, 설명만 한국어화
        korean_sample = {
            "task": korean_task,
            "tools": sample["tools"],  # Tool definitions
            "trajectory": translate_trajectory(sample["trajectory"]),
            "final_answer": translate_with_gpt4(sample["final_answer"]),
        }

        korean_samples.append(korean_sample)

    # Cost: 10K * $0.01 = $100
    return korean_samples


# Method 2: Korean-specific Generation (20K samples)
def generate_korean_specific_data():
    """
    한국 특화 시나리오 생성
    """
    domains = {
        "법률": [
            "판례 검색 후 유사 사례 찾기",
            "법령 조회 후 적용 가능성 분석",
            "소송 비용 계산",
        ],
        "금융": [
            "재무제표 분석 후 투자 의견",
            "대출 이자 계산 및 상환 계획",
            "주식 포트폴리오 최적화",
        ],
        "행정": [
            "민원 신청 절차 안내",
            "세금 계산 및 납부 기한 확인",
            "정부 지원금 신청 자격 확인",
        ],
    }

    synthetic_data = []
    for domain, task_templates in domains.items():
        for template in task_templates:
            # GPT-4로 다양한 variation 생성
            variations = generate_task_variations(
                domain=domain,
                template=template,
                num_variations=100,  # 각 template당 100개
                difficulty_levels=["easy", "medium", "hard"]
            )

            for variation in variations:
                # Tool trajectory 생성
                trajectory = generate_tool_trajectory(
                    task=variation,
                    available_tools=get_domain_tools(domain),
                    model="gpt-4"
                )

                # Verification
                is_valid = verify_trajectory(trajectory)
                if is_valid:
                    synthetic_data.append({
                        "domain": domain,
                        "task": variation,
                        "trajectory": trajectory,
                        "difficulty": variation["difficulty"],
                    })

    # Cost: 20K * $0.015 = $300
    return synthetic_data  # ~20K samples


# Method 3: Self-improvement Cycle (10K samples)
def self_improvement_generation(base_model, initial_data):
    """
    모델이 생성한 데이터로 재학습 (GLM-4.5 스타일)
    """
    current_model = base_model

    for iteration in range(3):  # 3 cycles
        # Generate new trajectories
        new_samples = []
        for task in initial_data:
            # Current model generates trajectory
            trajectory = current_model.generate_trajectory(
                task=task["task"],
                tools=task["tools"],
                num_candidates=5  # 5 candidates
            )

            # Verify and select best
            verified = [t for t in trajectory if verify_trajectory(t)]
            if verified:
                best = max(verified, key=lambda t: score_quality(t))
                new_samples.append({
                    "task": task["task"],
                    "trajectory": best,
                    "source": f"self_gen_iter{iteration}"
                })

        # Combine with original
        combined = initial_data + new_samples

        # Train next iteration
        next_model = finetune(base_model, combined)

        # Evaluate improvement
        improvement = evaluate(next_model) - evaluate(current_model)
        print(f"Iteration {iteration}: +{improvement:.1%}")

        if improvement < 0.02:  # <2% improvement
            break

        current_model = next_model

    return combined  # ~10K additional samples

# Total: 10K (translation) + 20K (korean-specific) + 10K (self-improvement)
#      = 40K high-quality samples
# Cost: $100 + $300 + $500 (training) = $900
# Time: 3-4 weeks
```

**Phase 3: Multi-turn Trajectory Generation (Week 9-12)**

```python
"""
복잡한 multi-turn agentic tasks
Kimi K2의 200-300 sequential calls를 목표로
"""

def generate_complex_trajectories():
    """
    Multi-step agentic tasks (5-15 steps)
    """

    # Template: 부산 출장 일정 잡기
    complex_task = {
        "task": "다음 주 부산 출장 일정을 잡아주세요. 2박3일이고 예산은 100만원입니다.",
        "required_steps": [
            "캘린더 확인",
            "부산 날씨 확인",
            "KTX 예약 가능 시간 검색",
            "호텔 검색 및 가격 비교",
            "예산 계산",
            "최적 일정 선택",
            "KTX 예약",
            "호텔 예약",
            "캘린더 등록",
            "확인 메일 발송",
        ],
        "expected_steps": 10-15,  # Tool calls
    }

    # User simulation
    user_simulator = UserSimulator(
        persona="30대 직장인, 출장 많음, 효율성 중시"
    )

    # Generate trajectory with user interaction
    trajectory = []
    state = {"calendar": [], "budget": 1000000, "preferences": {}}

    step = 0
    while not task_completed(state) and step < 20:
        # Model generates next action
        action = model.generate_action(
            task=complex_task["task"],
            state=state,
            history=trajectory
        )

        # Execute action
        result = execute_tool(action["tool"], action["parameters"])

        # Update state
        state = update_state(state, action, result)

        # Log trajectory
        trajectory.append({
            "step": step,
            "thought": action["thought"],  # Internal reasoning
            "tool": action["tool"],
            "parameters": action["parameters"],
            "result": result,
        })

        # User feedback (simulated)
        if action["requires_confirmation"]:
            user_feedback = user_simulator.respond(action, result)
            trajectory.append({
                "step": step,
                "type": "user_feedback",
                "content": user_feedback
            })

        step += 1

    # Verify final outcome
    success = verify_task_completion(complex_task, state)

    return {
        "task": complex_task,
        "trajectory": trajectory,
        "steps": len(trajectory),
        "success": success,
    }

# Generate 5K complex multi-turn tasks
complex_data = []
for i in range(5000):
    task_type = random.choice(["출장", "법률_분석", "금융_계획", "행정_절차"])
    task = generate_task_of_type(task_type)
    trajectory = generate_complex_trajectories(task)

    if trajectory["success"]:
        complex_data.append(trajectory)

# Cost: 5K * $0.03 = $150
# Time: 1-2 weeks
```

---

## 4️⃣ Verifiable Rewards Gym (한국어 특화)

### 4.1 자동 검증 가능한 작업 설계

**Kimi K2의 Verifiable Rewards Gym:**
```
핵심 아이디어: Human feedback 없이 자동으로 보상 계산
→ RL 훈련 비용 대폭 감소
```

**우리의 Korean Verifiable Rewards:**

```python
"""
한국어 Agentic Tasks의 자동 검증
"""

class KoreanVerifiableRewardsGym:
    def __init__(self):
        self.task_categories = {
            "computation": ComputationVerifier(),
            "information_retrieval": InformationVerifier(),
            "tool_calling": ToolCallingVerifier(),
            "multi_step_reasoning": ReasoningVerifier(),
        }

    def compute_reward(self, task, trajectory, outcome):
        """
        Task 타입에 따라 적절한 verifier 선택
        """
        category = classify_task(task)
        verifier = self.task_categories[category]

        reward = verifier.verify(task, trajectory, outcome)
        return reward


# Verifier 1: Computation Tasks
class ComputationVerifier:
    """
    수학, 금융 계산 등 검증 가능한 작업
    """
    def verify(self, task, trajectory, outcome):
        # Extract expected answer
        expected = extract_ground_truth(task)

        # Parse model output
        predicted = parse_answer(outcome)

        # Numerical comparison
        if isinstance(expected, (int, float)):
            error = abs(expected - predicted) / (abs(expected) + 1e-6)
            reward = max(0, 1.0 - error)
        else:
            reward = 1.0 if expected == predicted else 0.0

        # Process reward: Efficiency bonus
        optimal_steps = estimate_optimal_steps(task)
        actual_steps = len(trajectory)

        if actual_steps <= optimal_steps:
            efficiency_bonus = 0.2
        elif actual_steps <= optimal_steps * 1.5:
            efficiency_bonus = 0.1
        else:
            efficiency_bonus = 0.0

        total_reward = reward + efficiency_bonus

        return total_reward

    # Example tasks:
    # "2024년 1월부터 12월까지 월 50만원씩 적금하면 연 3% 이자로 얼마?"
    # → Expected: 6,089,713원
    # → Verifiable: 정확한 수치 비교


# Verifier 2: Information Retrieval
class InformationVerifier:
    """
    검색, 데이터 조회 작업
    """
    def verify(self, task, trajectory, outcome):
        # Extract key facts from outcome
        extracted_facts = extract_entities_and_facts(outcome)

        # Ground truth from reliable source
        ground_truth = query_reliable_source(task)

        # Fact-level comparison
        precision = count_correct_facts(extracted_facts, ground_truth) / len(extracted_facts)
        recall = count_correct_facts(extracted_facts, ground_truth) / len(ground_truth)

        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Tool use correctness
        tool_reward = 0.0
        for step in trajectory:
            if step["tool"] in ["naver_search", "wikipedia_kr", "law_search"]:
                # Did model use appropriate tool?
                if is_appropriate_tool(step["tool"], task):
                    tool_reward += 0.1

        total_reward = 0.7 * f1 + 0.3 * min(tool_reward, 1.0)

        return total_reward

    # Example tasks:
    # "대한민국 헌법 제1조 내용을 찾아서 요약해줘"
    # → Expected: "대한민국은 민주공화국이다"
    # → Verifiable: 핵심 키워드 포함 여부


# Verifier 3: Tool Calling
class ToolCallingVerifier:
    """
    API 호출, 프로그램 실행 작업
    """
    def verify(self, task, trajectory, outcome):
        reward = 0.0

        # 1. Syntactic correctness (30%)
        for step in trajectory:
            if step["type"] == "tool_call":
                is_valid_call = validate_tool_call_syntax(
                    step["tool"],
                    step["parameters"]
                )
                if is_valid_call:
                    reward += 0.3 / len([s for s in trajectory if s["type"] == "tool_call"])

        # 2. Semantic correctness (40%)
        # Execute tools in sandbox
        sandbox_result = execute_trajectory_in_sandbox(trajectory)

        if sandbox_result["success"]:
            reward += 0.4

        # 3. Outcome correctness (30%)
        if verify_final_outcome(task, outcome, sandbox_result):
            reward += 0.3

        return reward

    # Example tasks:
    # "오늘 코스피 지수 조회하고, 어제보다 얼마나 올랐는지 계산해줘"
    # → Trajectory: [stock_api("KOSPI", "today"), calculator("today - yesterday")]
    # → Verifiable: API 호출 성공 + 계산 정확도


# Verifier 4: Multi-step Reasoning
class ReasoningVerifier:
    """
    복잡한 추론 작업
    """
    def verify(self, task, trajectory, outcome):
        # 1. Intermediate step verification (50%)
        step_rewards = []
        for i, step in enumerate(trajectory):
            if "expected_steps" in task:
                expected_step = task["expected_steps"][i] if i < len(task["expected_steps"]) else None

                if expected_step:
                    similarity = compute_semantic_similarity(step["action"], expected_step)
                    step_rewards.append(similarity)

        avg_step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

        # 2. Final outcome verification (50%)
        outcome_reward = verify_complex_outcome(task, outcome)

        total_reward = 0.5 * avg_step_reward + 0.5 * outcome_reward

        return total_reward

    # Example tasks:
    # "민법 제750조 불법행위 책임이 성립하려면 어떤 요건이 필요한지 판례를 찾아서 설명해줘"
    # → Expected steps: [law_search("민법 제750조"), case_search("불법행위"), summarize()]
    # → Verifiable: 각 step의 적절성 + 최종 설명의 정확성


# Hybrid Verifier: LLM-as-Judge (for subjective tasks)
class LLMJudgeVerifier:
    """
    자동 검증 어려운 작업은 LLM으로 평가
    """
    def verify(self, task, trajectory, outcome):
        # Use GPT-4 as judge
        prompt = f"""
        Task: {task}

        Model's trajectory:
        {format_trajectory(trajectory)}

        Model's outcome:
        {outcome}

        Evaluate the quality on a scale of 0-1:
        1. Correctness (0-0.4): Is the outcome correct?
        2. Efficiency (0-0.3): Is the solution efficient?
        3. Completeness (0-0.3): Are all requirements met?

        Output format: {{"correctness": 0.0-1.0, "efficiency": 0.0-1.0, "completeness": 0.0-1.0, "total": 0.0-1.0}}
        """

        judgment = gpt4_judge(prompt)
        reward = judgment["total"]

        return reward

    # Use for: 법률 분석, 창의적 작업, 주관적 품질 평가
    # Cost: $0.01 per evaluation
    # → Only use for 10-20% of training data
```

---

### 4.2 RL Training with Verifiable Rewards

```python
"""
GRPO (Group Relative Policy Optimization) with Verifiable Rewards
Kimi K2 스타일 RL 훈련
"""

def train_korean_agentic_with_rl(
    base_model,
    verifiable_gym,
    training_data,
    num_epochs=3
):
    """
    Multi-stage RL training
    """

    # Stage 1: Simple tasks (1-2 steps)
    simple_tasks = [t for t in training_data if t["complexity"] == "simple"]

    model_v1 = rl_train(
        model=base_model,
        tasks=simple_tasks,
        reward_fn=verifiable_gym.compute_reward,
        algorithm="GRPO",
        epochs=num_epochs,
        batch_size=64,
    )

    print(f"Stage 1 완료: Simple task accuracy: {evaluate(model_v1, simple_tasks):.1%}")

    # Stage 2: Medium tasks (3-5 steps)
    medium_tasks = [t for t in training_data if t["complexity"] == "medium"]

    # Curriculum: 70% medium + 30% simple (forgetting 방지)
    curriculum_data = medium_tasks + random.sample(simple_tasks, len(medium_tasks) // 3)

    model_v2 = rl_train(
        model=model_v1,
        tasks=curriculum_data,
        reward_fn=verifiable_gym.compute_reward,
        algorithm="GRPO",
        epochs=num_epochs,
        batch_size=64,
    )

    print(f"Stage 2 완료: Medium task accuracy: {evaluate(model_v2, medium_tasks):.1%}")

    # Stage 3: Complex tasks (5-15 steps)
    complex_tasks = [t for t in training_data if t["complexity"] == "complex"]

    # Curriculum: 60% complex + 30% medium + 10% simple
    curriculum_data = (
        complex_tasks +
        random.sample(medium_tasks, int(len(complex_tasks) * 0.5)) +
        random.sample(simple_tasks, int(len(complex_tasks) * 0.17))
    )

    model_v3 = rl_train(
        model=model_v2,
        tasks=curriculum_data,
        reward_fn=verifiable_gym.compute_reward,
        algorithm="GRPO",
        epochs=num_epochs,
        batch_size=32,  # Larger context
    )

    print(f"Stage 3 완료: Complex task accuracy: {evaluate(model_v3, complex_tasks):.1%}")

    return model_v3


# GRPO Implementation
def rl_train(model, tasks, reward_fn, algorithm, epochs, batch_size):
    """
    Group Relative Policy Optimization
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    for epoch in range(epochs):
        for batch in batch_iterator(tasks, batch_size):
            # Generate K responses per prompt (K=4)
            responses = []
            for task in batch:
                task_responses = []
                for k in range(4):
                    response = model.generate(
                        task["task"],
                        temperature=0.8,
                        max_length=2048
                    )
                    task_responses.append(response)
                responses.append(task_responses)

            # Compute rewards
            rewards = []
            for task, task_responses in zip(batch, responses):
                task_rewards = []
                for response in task_responses:
                    trajectory = parse_trajectory(response)
                    outcome = extract_outcome(response)
                    reward = reward_fn(task, trajectory, outcome)
                    task_rewards.append(reward)
                rewards.append(task_rewards)

            # Group normalization
            advantages = []
            for task_rewards in rewards:
                mean_reward = np.mean(task_rewards)
                std_reward = np.std(task_rewards) + 1e-6
                task_advantages = [(r - mean_reward) / std_reward for r in task_rewards]
                advantages.append(task_advantages)

            # Policy gradient loss
            loss = 0
            for task, task_responses, task_advantages in zip(batch, responses, advantages):
                for response, advantage in zip(task_responses, task_advantages):
                    log_prob = model.compute_log_prob(task["task"], response)
                    loss -= log_prob * advantage

            loss = loss / (batch_size * 4)

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate
        accuracy = evaluate(model, tasks)
        print(f"Epoch {epoch+1}/{epochs}: Accuracy: {accuracy:.1%}")

    return model


# Cost estimation
def estimate_rl_cost():
    """
    RL training cost
    """
    config = {
        "model_size": "7B",
        "num_tasks": 40000,
        "responses_per_task": 4,  # GRPO K=4
        "epochs": 3,
        "gpus": 16,
        "gpu_type": "A100",
        "hours_per_epoch": 48,
    }

    total_hours = config["epochs"] * config["hours_per_epoch"] * config["gpus"]
    cost_per_hour = 2.0  # A100 $2/hour

    total_cost = total_hours * cost_per_hour

    print(f"""
    RL Training Cost:
    - GPUs: {config['gpus']} x {config['gpu_type']}
    - Duration: {config['epochs']} epochs x {config['hours_per_epoch']} hours = {config['epochs'] * config['hours_per_epoch']} hours
    - GPU hours: {total_hours}
    - Total cost: ${total_cost:,}
    """)

    return total_cost

# RL Training: $4,608 (16 A100 x 144 hours x $2/hr)
```

---

## 5️⃣ 평가 및 벤치마크

### 5.1 Korean Agentic Benchmark (KAB) 설계

**문제: 한국어 agentic 벤치마크 없음**

**해결: 자체 제작**

```python
"""
Korean Agentic Benchmark (KAB)
3가지 카테고리, 총 200 tasks
"""

class KoreanAgenticBenchmark:
    def __init__(self):
        self.categories = {
            "tool_use": ToolUseBenchmark(),
            "multi_step_reasoning": MultiStepReasoningBenchmark(),
            "domain_specific": DomainSpecificBenchmark(),
        }

    def evaluate(self, model):
        results = {}
        for category, benchmark in self.categories.items():
            score = benchmark.evaluate(model)
            results[category] = score

        # Overall score
        results["overall"] = np.mean(list(results.values()))

        return results


# Category 1: Tool Use (80 tasks)
class ToolUseBenchmark:
    """
    기본 tool-calling 능력 평가
    """
    def __init__(self):
        self.tasks = self.create_tasks()

    def create_tasks(self):
        tasks = []

        # Level 1: Single tool call (30 tasks)
        tasks.extend([
            {
                "task": "오늘 서울 날씨 알려줘",
                "expected_tools": ["weather_api"],
                "verifiable": True,
                "difficulty": "easy",
            },
            {
                "task": "삼성전자 현재 주가 조회해줘",
                "expected_tools": ["stock_api"],
                "verifiable": True,
                "difficulty": "easy",
            },
            {
                "task": "1달러는 한국 돈으로 얼마야?",
                "expected_tools": ["exchange_rate"],
                "verifiable": True,
                "difficulty": "easy",
            },
            # ... 30 tasks total
        ])

        # Level 2: Sequential tool calls (30 tasks)
        tasks.extend([
            {
                "task": "서울 날씨 확인하고, 비 오면 우산 챙기라고 알려줘",
                "expected_tools": ["weather_api", "conditional_logic"],
                "expected_steps": 2-3,
                "verifiable": True,
                "difficulty": "medium",
            },
            {
                "task": "삼성전자 주가 조회하고, 어제 대비 등락률 계산해줘",
                "expected_tools": ["stock_api", "calculator"],
                "expected_steps": 3,
                "verifiable": True,
                "difficulty": "medium",
            },
            # ... 30 tasks total
        ])

        # Level 3: Complex tool orchestration (20 tasks)
        tasks.extend([
            {
                "task": "코스피 상위 10개 종목의 시가총액을 조회하고, 엑셀 파일로 만들어줘",
                "expected_tools": ["stock_api", "data_processor", "file_generator"],
                "expected_steps": 5-7,
                "verifiable": True,
                "difficulty": "hard",
            },
            # ... 20 tasks total
        ])

        return tasks

    def evaluate(self, model):
        correct = 0
        for task in self.tasks:
            response = model.generate_with_tools(
                task["task"],
                available_tools=get_all_tools()
            )

            # Parse trajectory
            trajectory = parse_trajectory(response)

            # Verify
            is_correct = self.verify_task(task, trajectory)
            if is_correct:
                correct += 1

        accuracy = correct / len(self.tasks)
        return accuracy


# Category 2: Multi-step Reasoning (80 tasks)
class MultiStepReasoningBenchmark:
    """
    복잡한 multi-step reasoning 평가
    """
    def create_tasks(self):
        tasks = []

        # Subcategory 1: 법률 (20 tasks)
        tasks.extend([
            {
                "task": "민법 제750조에 따른 불법행위 책임이 성립하려면 어떤 요건이 필요한지 판례를 찾아서 설명해줘",
                "domain": "법률",
                "expected_steps": ["law_search", "case_search", "analysis", "summary"],
                "expected_length": "medium",  # 200-400 words
                "verifiable": "partial",  # LLM judge
                "difficulty": "hard",
            },
            # ... 20 tasks
        ])

        # Subcategory 2: 금융 (20 tasks)
        tasks.extend([
            {
                "task": "삼성전자 최근 3년 재무제표를 분석하고, 투자 의견을 제시해줘",
                "domain": "금융",
                "expected_steps": ["company_info", "financial_data", "ratio_analysis", "comparison", "recommendation"],
                "expected_length": "long",  # 400+ words
                "verifiable": "partial",
                "difficulty": "hard",
            },
            # ... 20 tasks
        ])

        # Subcategory 3: 행정 (20 tasks)
        tasks.extend([
            {
                "task": "1인 기업을 설립하려고 하는데 필요한 절차와 비용을 알려줘",
                "domain": "행정",
                "expected_steps": ["procedure_search", "cost_calculation", "document_list", "timeline"],
                "expected_length": "medium",
                "verifiable": "partial",
                "difficulty": "medium",
            },
            # ... 20 tasks
        ])

        # Subcategory 4: 일반 (20 tasks)
        tasks.extend([
            {
                "task": "다음 주 제주도 여행 계획을 짜줘. 2박3일이고 예산은 50만원이야.",
                "domain": "일반",
                "expected_steps": ["weather", "flights", "hotels", "attractions", "budget", "itinerary"],
                "expected_length": "long",
                "verifiable": "partial",
                "difficulty": "medium",
            },
            # ... 20 tasks
        ])

        return tasks


# Category 3: Domain-specific (40 tasks)
class DomainSpecificBenchmark:
    """
    특정 도메인 전문 능력 평가
    """
    def create_tasks(self):
        # 법률 전문 (20 tasks)
        legal_tasks = [
            {
                "task": "음주운전으로 사고를 낸 경우 형사처벌과 행정처분은 어떻게 되나요?",
                "domain": "법률",
                "expertise": "형법",
                "expected_citations": ["도로교통법", "판례"],
                "verifiable": "expert_review",
            },
            # ... 20 tasks
        ]

        # 금융 전문 (20 tasks)
        finance_tasks = [
            {
                "task": "주택담보대출 DSR 규제가 완화되면 부동산 시장에 어떤 영향이 있을까요?",
                "domain": "금융",
                "expertise": "부동산 금융",
                "expected_analysis": ["DSR 계산", "시장 영향", "정책 효과"],
                "verifiable": "expert_review",
            },
            # ... 20 tasks
        ]

        return legal_tasks + finance_tasks


# Benchmark creation process
def create_korean_agentic_benchmark():
    """
    KAB 제작 프로세스
    """

    # Step 1: Task collection (Week 1-2)
    # - Domain experts 섭외 (법률, 금융, 행정)
    # - 각 도메인당 50-100 candidate tasks 수집
    # - Difficulty labeling

    # Step 2: Task refinement (Week 3)
    # - Duplicates 제거
    # - Clarity 향상
    # - Verifiability 확인
    # - Expected steps/tools 정의

    # Step 3: Ground truth creation (Week 4)
    # - Expert solutions (20% of tasks)
    # - GPT-4 solutions (80% of tasks)
    # - Human verification (100% of tasks)

    # Step 4: Pilot evaluation (Week 5)
    # - Test with baseline models (GPT-4, Polyglot-Ko)
    # - Identify too easy/hard tasks
    # - Adjust difficulty distribution

    # Step 5: Final benchmark release (Week 6)
    # - Public release on Hugging Face
    # - Leaderboard setup
    # - Documentation

    cost = {
        "domain_experts": 3 * 2000 * 2,  # 3 experts, $2K each, 2 weeks
        "crowdsourcing": 200 * 20,  # 200 tasks, $20 each for ground truth
        "gpt4_solutions": 160 * 1,  # 160 tasks (80%), $1 each
        "infrastructure": 1000,  # Leaderboard, hosting
    }

    total_cost = sum(cost.values())
    print(f"KAB Creation Cost: ${total_cost:,}")

    return total_cost

# Cost: $17,000
# Time: 6 weeks
# Output: Korean Agentic Benchmark with 200 high-quality tasks
```

---

### 5.2 평가 프로세스

```python
"""
Iterative evaluation and improvement
"""

def evaluation_cycle(model, benchmark, iteration):
    """
    매 iteration마다 평가
    """

    # 1. Benchmark evaluation
    benchmark_scores = benchmark.evaluate(model)

    print(f"""
    Iteration {iteration} Results:
    - Tool Use: {benchmark_scores['tool_use']:.1%}
    - Multi-step Reasoning: {benchmark_scores['multi_step_reasoning']:.1%}
    - Domain-specific: {benchmark_scores['domain_specific']:.1%}
    - Overall: {benchmark_scores['overall']:.1%}
    """)

    # 2. Error analysis
    errors = benchmark.get_errors(model)
    error_patterns = analyze_error_patterns(errors)

    print(f"Top 3 error patterns:")
    for pattern in error_patterns[:3]:
        print(f"  - {pattern['type']}: {pattern['frequency']} occurrences")

    # 3. Targeted improvement
    improvement_tasks = generate_improvement_tasks(error_patterns)

    print(f"Generated {len(improvement_tasks)} targeted improvement tasks")

    # 4. Decision: Continue or stop?
    if benchmark_scores['overall'] >= 0.75:  # 75% target
        print("✅ Target achieved! Ready for production.")
        return "DEPLOY"

    elif improvement_rate(iteration) < 0.02:  # <2% improvement
        print("⚠️  Diminishing returns. Consider pivot.")
        return "PIVOT"

    elif budget_remaining() < 0.2:  # <20% budget left
        print("⚠️  Budget low. Wrapping up.")
        return "WRAP_UP"

    else:
        print("➡️  Continuing to next iteration.")
        return "CONTINUE", improvement_tasks


# Real-world evaluation
def real_world_pilot(model, domain="법률"):
    """
    실제 사용자와 파일럿 테스트
    """

    # Recruit 10-20 users from target domain
    users = recruit_pilot_users(domain, count=15)

    # Each user tests model for 1 week
    feedback = []
    for user in users:
        tasks = user.generate_real_tasks(count=20)  # 20 real tasks per user

        for task in tasks:
            response = model.generate_with_tools(task)

            rating = user.rate_response(
                response,
                criteria=["correctness", "efficiency", "usability"]
            )

            feedback.append({
                "user": user.id,
                "task": task,
                "response": response,
                "rating": rating,
                "comments": user.comments,
            })

    # Aggregate feedback
    avg_rating = np.mean([f["rating"]["overall"] for f in feedback])

    print(f"""
    Pilot Test Results ({domain}):
    - Users: {len(users)}
    - Tasks: {len(feedback)}
    - Average rating: {avg_rating:.1f}/5.0
    - Satisfaction: {avg_rating/5.0:.1%}
    """)

    # Analyze failure cases
    failures = [f for f in feedback if f["rating"]["overall"] < 3.0]
    print(f"Failure cases: {len(failures)} ({len(failures)/len(feedback):.1%})")

    for failure in failures[:5]:  # Top 5 failures
        print(f"  - Task: {failure['task']}")
        print(f"    Issue: {failure['comments']}")

    return feedback


# Continuous improvement loop
def continuous_improvement(model, benchmark):
    """
    Production 배포 후에도 계속 개선
    """

    # Collect production feedback
    production_data = collect_production_data(days=7)  # 1 week

    # Filter high-quality interactions
    high_quality = [
        d for d in production_data
        if d["user_rating"] >= 4.0 and d["task_complexity"] >= "medium"
    ]

    print(f"Collected {len(high_quality)} high-quality production samples")

    # Add to training data
    augmented_data = training_data + high_quality

    # Retrain (LoRA fine-tuning for efficiency)
    updated_model = lora_finetune(
        base_model=model,
        data=high_quality,
        epochs=1,
        rank=16,
    )

    # A/B test
    improvement = ab_test(model, updated_model, duration_days=3)

    if improvement["win_rate"] > 0.55:  # >55% users prefer new model
        print(f"✅ Model updated! Win rate: {improvement['win_rate']:.1%}")
        return updated_model
    else:
        print(f"❌ No significant improvement. Keeping current model.")
        return model
```

---

## 6️⃣ 6개월 실행 로드맵

### Month 1-2: 기반 구축

**Week 1-2: Tool Ecosystem**
```
[ ] Day 1-3: Core tools 설계 (20개)
    - Computation: calculator, financial_calc, date_calc
    - Retrieval: naver_search, wikipedia_kr, law_search
    - Data: weather, stock, exchange_rate
    - Execution: python_sandbox, sql_query

[ ] Day 4-7: Tool implementation
    - API 통합
    - Sandbox 설정
    - Verification logic

[ ] Day 8-14: Tool testing
    - Unit tests
    - Integration tests
    - Documentation

Cost: $5K (API access, infrastructure)
Deliverable: 20 production-ready tools
```

**Week 3-4: Base Model Selection**
```
[ ] Day 1-2: Candidate evaluation
    - Polyglot-Ko-12.8B
    - Llama-3-8B + Korean SFT
    - Qwen2-7B + Korean adaptation

[ ] Day 3-5: Quick tests
    - Korean comprehension
    - Tool-use capability (with prompting)
    - Instruction following

[ ] Day 6-7: Decision
    → Polyglot-Ko-12.8B recommended
    (Best Korean performance, 12.8B manageable for 4-person team)

Cost: $500 (GPU hours for testing)
Deliverable: Base model selected
```

**Week 5-8: 데이터 생성**
```
[ ] Week 5: Translation (10K samples)
    - ToolBench → Korean
    - Berkeley Function Calling → Korean
    - Quality check

[ ] Week 6-7: Korean-specific generation (20K samples)
    - 법률: 5K
    - 금융: 5K
    - 행정: 5K
    - 일반: 5K

[ ] Week 8: Multi-turn trajectories (5K samples)
    - Complex tasks (5-15 steps)
    - User simulation
    - Verification

Cost: $2K (GPT-4 generation)
Deliverable: 35K Korean agentic training samples
```

---

### Month 3-4: RL Training

**Week 9-10: Stage 1 - Simple Tasks**
```
[ ] RL training on simple tasks (10K samples)
    - 1-2 step tool calls
    - High verification accuracy
    - Build foundation

GPUs: 16 A100
Duration: 2 weeks
Cost: $8K

Target: 80% accuracy on simple tool-use
```

**Week 11-12: Stage 2 - Medium Tasks**
```
[ ] RL training on medium tasks (15K samples)
    - 3-5 step reasoning
    - Curriculum: 70% medium + 30% simple
    - Forgetting prevention

GPUs: 16 A100
Duration: 2 weeks
Cost: $8K

Target: 70% accuracy on medium multi-step
```

**Week 13-16: Stage 3 - Complex Tasks**
```
[ ] RL training on complex tasks (10K samples)
    - 5-15 step agentic tasks
    - Curriculum: 60% complex + 30% medium + 10% simple
    - Full agentic capability

GPUs: 16 A100
Duration: 4 weeks
Cost: $16K

Target: 60% accuracy on complex agentic tasks
```

---

### Month 5: 평가 및 개선

**Week 17-18: Benchmark Evaluation**
```
[ ] Week 17: KAB creation
    - 200 tasks across 3 categories
    - Ground truth from experts
    - Pilot testing

[ ] Week 18: Full evaluation
    - Tool Use: Target 75%
    - Multi-step Reasoning: Target 65%
    - Domain-specific: Target 60%

Cost: $5K (benchmark creation)
```

**Week 19-20: Gap Analysis & Iteration**
```
[ ] Error analysis
    - Identify failure patterns
    - Generate targeted improvement data

[ ] Targeted RL iteration
    - 5K additional samples for weak areas
    - 1 epoch fine-tuning

GPUs: 16 A100
Duration: 1 week
Cost: $4K

Target: +5-10% improvement
```

---

### Month 6: Production & Launch

**Week 21-22: 최적화**
```
[ ] Quantization
    - QAT INT4 (Kimi K2 style)
    - 2x speedup, minimal degradation

[ ] Inference optimization
    - vLLM integration
    - Batching strategies
    - Latency testing

Cost: $2K
Target: <2s per tool call
```

**Week 23: Real-world Pilot**
```
[ ] Recruit 15 users
    - 법률: 5 users
    - 금융: 5 users
    - 행정: 5 users

[ ] 1-week pilot test
    - Real tasks (20 per user)
    - Feedback collection
    - Satisfaction measurement

Cost: $3K (user incentives)
Target: 4.0/5.0 satisfaction
```

**Week 24: Launch**
```
[ ] Public beta release
    - API endpoint
    - Documentation
    - Demo website

[ ] Marketing
    - Blog post
    - Korean AI community
    - Industry contacts

Cost: $2K (hosting, marketing)
```

---

## 7️⃣ 비용 및 리소스 요약

### 총 비용

| Phase | Duration | Cost |
|-------|----------|------|
| **Month 1-2: 기반 구축** | 8 weeks | $7.5K |
| - Tool Ecosystem | 2 weeks | $5K |
| - Base Model Selection | 1 week | $0.5K |
| - 데이터 생성 | 4 weeks | $2K |
| **Month 3-4: RL Training** | 8 weeks | $32K |
| - Stage 1: Simple | 2 weeks | $8K |
| - Stage 2: Medium | 2 weeks | $8K |
| - Stage 3: Complex | 4 weeks | $16K |
| **Month 5: 평가 및 개선** | 4 weeks | $9K |
| - Benchmark | 2 weeks | $5K |
| - Iteration | 2 weeks | $4K |
| **Month 6: Production** | 4 weeks | $7K |
| - Optimization | 2 weeks | $2K |
| - Pilot | 1 week | $3K |
| - Launch | 1 week | $2K |
| **Total** | **6 months** | **$55.5K** |

**Total Budget: $55-60K**

---

### GPU 요구사항

**Training:**
- 16 A100 GPUs (40GB)
- 3 months (Month 3-5)
- Cloud: AWS, GCP, Lambda Labs

**Inference (Production):**
- 2-4 A100 GPUs
- vLLM + INT4 quantization
- Expected: 100-200 requests/min

---

### 팀 역할 분담

**4명 팀:**

1. **Data Lead (Person 1)**
   - Tool ecosystem 구축
   - 데이터 생성 파이프라인
   - Quality control

2. **Training Lead (Person 2)**
   - RL 훈련 세팅
   - Hyperparameter tuning
   - Forgetting monitoring

3. **Evaluation Lead (Person 3)**
   - Benchmark 제작
   - Error analysis
   - Iteration planning

4. **Domain Expert (Person 4)**
   - 법률/금융/행정 도메인 지식
   - Task 검증
   - Real-world pilot 운영

---

## 8️⃣ 성공 지표

### 기술 지표

**Benchmark Performance:**
```
Korean Agentic Benchmark (KAB):
✅ Tool Use: 75%+ (baseline: 40%)
✅ Multi-step Reasoning: 65%+ (baseline: 30%)
✅ Domain-specific: 60%+ (baseline: 25%)
✅ Overall: 70%+
```

**Efficiency:**
```
✅ Average steps: <1.5x optimal
✅ Success rate: >80% (within 3 tries)
✅ Latency: <2s per tool call
```

**Robustness:**
```
✅ Error recovery: 70%+ (can fix mistakes)
✅ Edge cases: 60%+ (handles unusual inputs)
✅ Multi-turn coherence: 75%+
```

---

### 비즈니스 지표

**Pilot Test (Month 6):**
```
✅ User satisfaction: 4.0/5.0+
✅ Task completion rate: 75%+
✅ Repeat usage: 60%+ (users return)
```

**Production (Month 7-12):**
```
✅ Active users: 100+ (Month 7)
✅ Daily tasks: 1,000+ (Month 8)
✅ Revenue: $10K+/month (Month 10)
✅ Churn: <15%
```

---

## 9️⃣ 리스크 및 대응

### 주요 리스크

**Risk 1: 데이터 품질 부족**
```
문제: Synthetic data가 실제 사용과 gap
대응:
1. Real-world pilot 빠르게 실시 (Month 4)
2. Production feedback loop 구축
3. Active learning으로 효율적 labeling
```

**Risk 2: RL 훈련 불안정**
```
문제: Reward hacking, catastrophic forgetting
대응:
1. Curriculum learning (점진적 난이도)
2. Multi-task training (40% 이전 데이터 유지)
3. Forgetting monitor (자동 감지)
```

**Risk 3: Tool Ecosystem 한계**
```
문제: 20-30개 tools로는 부족할 수 있음
대응:
1. Phase 1: 핵심 20개로 MVP
2. Phase 2: User feedback 기반 확장
3. Community contributions (open-source tools)
```

**Risk 4: 시장 수용성**
```
문제: 기업들이 도입 주저
대응:
1. Free tier로 진입장벽 낮춤
2. Success cases 빠르게 확보
3. ROI 명확히 제시 (자동화 시간 절감)
```

---

## 🚀 시작하기

### Week 1 Action Plan

```
[ ] Day 1: Team Kickoff
    [ ] 목표 재확인: 한국어 agentic AI
    [ ] 역할 분담: Data/Training/Eval/Domain
    [ ] Success metrics 합의
    [ ] Budget 확정: $55-60K

[ ] Day 2-3: Tool Ecosystem 설계
    [ ] 핵심 20개 tools 리스트
    [ ] API 접근 확인 (네이버, 공공데이터)
    [ ] Sandbox 환경 설정

[ ] Day 4-5: Base Model
    [ ] Polyglot-Ko-12.8B download
    [ ] Quick capability test
    [ ] Infrastructure setup (16 A100 예약)

[ ] Day 6-7: 데이터 파이프라인
    [ ] GPT-4 API 설정
    [ ] Translation pipeline 구축
    [ ] 첫 100 samples 생성 및 검증
```

---

## 📚 참고 자료

**Papers:**
- Kimi K2: "Open Agentic Intelligence" - Agentic data synthesis pipeline
- K2 Thinking: Test-time scaling with 200-300 tool calls
- GLM-4.5: Multi-stage RL training
- SmolLM3 Playbook: Training reality

**Korean Resources:**
- Polyglot-Ko models: EleutherAI Korean LLM
- Korean NLP community
- 공공데이터 포털 API
- 네이버/카카오 Open API

**Tools & Frameworks:**
- Training: TRL, Axolotl, OpenRLHF
- Inference: vLLM, TGI
- Evaluation: lm-evaluation-harness
- Monitoring: Weights & Biases

---

## 💪 Closing Thoughts

**한국어 Agentic AI는 Blue Ocean입니다:**

✅ **시장**: 국내 기업/정부 수요 폭발, 경쟁자 전무
✅ **기술**: Post-training으로 실현 가능, 4명 팀 규모 적합
✅ **차별화**: 한국 특화 tools + 도메인 전문성
✅ **수익성**: B2B SaaS, API, 컨설팅 등 다양한 수익 모델

**6개월 로드맵:**
```
Month 1-2: Tools + Data (기반)
Month 3-4: RL Training (핵심)
Month 5: Evaluation + Iteration (개선)
Month 6: Production + Launch (출시)
```

**예상 성과:**
- Korean Agentic Benchmark: 70%+ (기존 40%)
- User satisfaction: 4.0/5.0+
- Production-ready specialized model

**핵심 전략:**
> "Kimi K2의 agentic 방법론 + 한국어 특화 = Blue Ocean 선점"

**시작은 Week 1부터. Good luck! 🚀**

---

**작성일**: 2025-11-12
**기반**: Kimi K2, K2 Thinking, GLM-4.5, MiniMax-M1 전체 분석
**대상**: 4명 팀, 한국어 agentic AI 목표
**예산**: $55-60K, 6개월
