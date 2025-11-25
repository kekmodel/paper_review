# DreamGym Banking Tasks

뱅킹 에이전트 학습을 위한 태스크 정의 및 커리큘럼

## Overview

은행 업무 자동화를 위한 에이전트 태스크. 난이도별로 구성되어 커리큘럼 학습에 활용.

```
Difficulty Distribution:
├─ Easy (0.1-0.3): 단순 조회, 기본 이체
├─ Medium (0.4-0.6): 복합 거래, 상품 가입
├─ Hard (0.7-0.9): 대출 심사, 자산 관리, 다중 조건 처리
└─ Expert (0.9+): 복합 금융 상품, 규정 준수 판단
```

---

## 0. 공통 State 정의 (Base Context)

모든 태스크에서 에이전트가 접근 가능한 기본 컨텍스트 정보.

### 0.1 고객 정보 (Customer Profile)
```python
customer_profile = {
    # === 기본 정보 ===
    "user_id": "U12345",
    "name": "홍길동",
    "phone": "010-1234-5678",
    "email": "hong@email.com",
    "birth_date": "1990-05-15",
    "age": 34,
    "gender": "M",

    # === 인증 상태 ===
    "authenticated": True,
    "auth_level": "full",  # "basic", "full", "enhanced"
    "last_login": "2024-11-22 09:30:00",

    # === 고객 등급 ===
    "customer_grade": "우수",  # "일반", "우수", "VIP", "VVIP"
    "membership_years": 5,
    "total_assets": 150000000,  # 총 자산

    # === 직업/소득 정보 ===
    "occupation": "회사원",
    "company": "ABC주식회사",
    "annual_income": 60000000,
    "employment_years": 8,

    # === 신용 정보 ===
    "credit_score": 850,
    "credit_grade": "1등급",

    # === 투자 성향 ===
    "investment_profile": {
        "risk_tolerance": "중립형",  # "안정형", "안정추구형", "중립형", "적극투자형", "공격투자형"
        "investment_experience": 3,   # 투자 경력 (년)
        "survey_date": "2024-06-15"
    },

    # === 마케팅 동의 ===
    "marketing_consent": {
        "sms": True,
        "email": True,
        "push": True,
        "phone": False
    }
}
```

### 0.2 계좌 정보 (Accounts)
```python
accounts = [
    {
        "account_no": "110-XXX-XXXXX",
        "type": "입출금",
        "product_name": "자유입출금통장",
        "balance": 15000000,
        "available_balance": 14500000,  # 출금 가능 금액
        "holder": "홍길동",
        "opened_date": "2019-03-15",
        "status": "active",
        "daily_transfer_limit": 5000000,
        "today_transferred": 500000
    },
    {
        "account_no": "111-XXX-XXXXX",
        "type": "적금",
        "product_name": "자유적금",
        "balance": 5000000,
        "monthly_payment": 500000,
        "interest_rate": 4.0,
        "maturity_date": "2025-06-15",
        "holder": "홍길동",
        "opened_date": "2023-06-15",
        "status": "active"
    },
    {
        "account_no": "112-XXX-XXXXX",
        "type": "예금",
        "product_name": "정기예금",
        "balance": 50000000,
        "interest_rate": 3.8,
        "maturity_date": "2024-12-20",
        "holder": "홍길동",
        "opened_date": "2023-12-20",
        "status": "active"
    }
]
```

### 0.3 거래 내역 (Transaction History)
```python
transaction_history = [
    # === 최근 거래 내역 (최근 3개월) ===
    {
        "tx_id": "TX001",
        "date": "2024-11-22 09:15:00",
        "type": "이체",
        "direction": "출금",
        "amount": 500000,
        "balance_after": 14500000,
        "account_no": "110-XXX-XXXXX",
        "counterparty": {
            "name": "엄마",
            "account_no": "220-YYY-YYYYY",
            "bank": "국민은행",
            "holder": "김순자"
        },
        "memo": "용돈",
        "channel": "모바일"
    },
    {
        "tx_id": "TX002",
        "date": "2024-11-20 14:30:00",
        "type": "입금",
        "direction": "입금",
        "amount": 3500000,
        "balance_after": 15000000,
        "account_no": "110-XXX-XXXXX",
        "counterparty": {
            "name": "ABC주식회사",
            "account_no": "999-ZZZ-ZZZZZ",
            "bank": "기업은행",
            "holder": "ABC주식회사"
        },
        "memo": "11월급여",
        "channel": "자동이체"
    },
    {
        "tx_id": "TX003",
        "date": "2024-11-15 18:20:00",
        "type": "이체",
        "direction": "출금",
        "amount": 300000,
        "balance_after": 11500000,
        "account_no": "110-XXX-XXXXX",
        "counterparty": {
            "name": "동생",
            "account_no": "330-ZZZ-ZZZZZ",
            "bank": "신한은행",
            "holder": "홍길순"
        },
        "memo": "생일선물",
        "channel": "모바일"
    },
    {
        "tx_id": "TX004",
        "date": "2024-11-10 12:00:00",
        "type": "카드결제",
        "direction": "출금",
        "amount": 45000,
        "balance_after": 11800000,
        "account_no": "110-XXX-XXXXX",
        "merchant": "스타벅스 강남점",
        "category": "식음료",
        "channel": "카드"
    },
    {
        "tx_id": "TX005",
        "date": "2024-11-01 09:00:00",
        "type": "자동이체",
        "direction": "출금",
        "amount": 850000,
        "balance_after": 11845000,
        "account_no": "110-XXX-XXXXX",
        "counterparty": {
            "name": "임대인",
            "account_no": "444-AAA-AAAAA",
            "bank": "하나은행",
            "holder": "박건물"
        },
        "memo": "11월 월세",
        "channel": "자동이체"
    },
    # ... 더 많은 거래내역
]
```

### 0.4 연락처 (Contacts)
```python
contacts = [
    {
        "id": "C001",
        "alias": "엄마",           # 사용자가 설정한 별칭
        "name": "김순자",          # 예금주명
        "account_no": "220-YYY-YYYYY",
        "bank": "국민은행",
        "bank_code": "004",
        "registered_date": "2020-01-15",
        "last_transfer_date": "2024-11-22",
        "transfer_count": 45       # 이체 횟수
    },
    {
        "id": "C002",
        "alias": "동생",
        "name": "홍길순",
        "account_no": "330-ZZZ-ZZZZZ",
        "bank": "신한은행",
        "bank_code": "088",
        "registered_date": "2020-03-20",
        "last_transfer_date": "2024-11-15",
        "transfer_count": 23
    },
    {
        "id": "C003",
        "alias": "월세",
        "name": "박건물",
        "account_no": "444-AAA-AAAAA",
        "bank": "하나은행",
        "bank_code": "081",
        "registered_date": "2023-01-01",
        "last_transfer_date": "2024-11-01",
        "transfer_count": 11
    }
]
```

### 0.5 대출 정보 (Loans)
```python
loans = [
    {
        "loan_id": "L001",
        "type": "신용대출",
        "product_name": "직장인신용대출",
        "principal": 30000000,       # 대출원금
        "balance": 25000000,         # 대출잔액
        "interest_rate": 5.5,
        "monthly_payment": 950000,
        "start_date": "2023-06-01",
        "end_date": "2026-06-01",
        "repayment_type": "원리금균등",
        "status": "정상"
    }
]
```

### 0.6 카드 정보 (Cards)
```python
cards = [
    {
        "card_no": "1234-XXXX-XXXX-5678",
        "type": "신용",
        "product_name": "올리브카드",
        "credit_limit": 10000000,
        "used_amount": 1500000,
        "available_amount": 8500000,
        "payment_date": 15,          # 결제일
        "linked_account": "110-XXX-XXXXX",
        "status": "active"
    },
    {
        "card_no": "9876-XXXX-XXXX-4321",
        "type": "체크",
        "product_name": "체크카드",
        "linked_account": "110-XXX-XXXXX",
        "status": "active"
    }
]
```

### 0.7 State 구조
```python
# 모든 태스크에서 사용하는 공통 State
base_state = {
    # 고객 정보
    "customer": customer_profile,

    # 금융 정보
    "accounts": accounts,
    "cards": cards,
    "loans": loans,

    # 거래 관련
    "transaction_history": transaction_history,
    "contacts": contacts,

    # 현재 대화
    "conversation": [
        {"role": "user", "content": "<사용자 요청>"}
    ],

    # 세션 정보
    "session": {
        "channel": "mobile",        # "mobile", "web", "atm", "branch"
        "timestamp": "2024-11-22 10:00:00",
        "device_id": "device_abc123"
    }
}
```

### 0.8 정보 접근 흐름
```
┌─────────────────────────────────────────────────────────────────┐
│                    뱅킹 에이전트 컨텍스트                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  고객 정보    │    │  금융 정보    │    │  거래 정보    │      │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤      │
│  │ • 기본정보    │    │ • 계좌 목록   │    │ • 거래내역    │      │
│  │ • 인증상태    │    │ • 카드 목록   │    │ • 연락처     │      │
│  │ • 등급/자산   │    │ • 대출 현황   │    │ • 예약이체   │      │
│  │ • 직업/소득   │    │ • 예적금     │    │ • 자동이체   │      │
│  │ • 신용정보    │    │ • 펀드/투자   │    │              │      │
│  │ • 투자성향    │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│          │                  │                  │               │
│          └──────────────────┼──────────────────┘               │
│                             ▼                                  │
│                    ┌──────────────┐                            │
│                    │   에이전트    │                            │
│                    │  (LLM 기반)  │                            │
│                    └──────────────┘                            │
│                             │                                  │
│          ┌──────────────────┼──────────────────┐               │
│          ▼                  ▼                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  조회 기능    │    │  거래 기능    │    │  상품 기능    │      │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤      │
│  │ • 잔액조회    │    │ • 이체       │    │ • 상품추천    │      │
│  │ • 내역조회    │    │ • 예약이체   │    │ • 가입신청    │      │
│  │ • 한도조회    │    │ • 자동이체   │    │ • 해지신청    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 계좌 조회 (Easy)

### Task 1.1: 잔액 조회
```python
Task(
    id="balance_inquiry",
    description="내 입출금 계좌 잔액 조회해줘",
    difficulty=0.1,
    category="inquiry"
)

# Expected trajectory
# Step 1: 계좌 목록 조회 → reward=1 (필수 단계)
# Step 2: 잔액 정보 반환 → reward=1 (목표 달성)
# Final: 1.0 (모든 스텝 기여)
```

**State 정의:**
```python
state = {
    "user_id": "U12345",
    "authenticated": True,
    "accounts": [
        {"account_no": "110-XXX-XXXXX", "type": "입출금", "balance": 1500000},
        {"account_no": "111-XXX-XXXXX", "type": "적금", "balance": 5000000}
    ],
    "conversation": [
        {"role": "user", "content": "내 입출금 계좌 잔액 조회해줘"}
    ]
}
```

**Actions:**
```python
available_actions = [
    {"tool": "get_account_list", "params": {"user_id": str}},
    {"tool": "get_balance", "params": {"account_no": str}},
    {"tool": "respond", "params": {"message": str}}
]
```

### Task 1.2: 거래내역 조회
```python
Task(
    id="transaction_history",
    description="이번 달 카드 사용 내역 보여줘",
    difficulty=0.2,
    category="inquiry"
)

# Expected trajectory
# Step 1: 카드 목록 조회 → reward=1
# Step 2: 기간 설정 (이번 달) → reward=1
# Step 3: 거래내역 조회 → reward=1
# Step 4: 결과 포맷팅 및 응답 → reward=1
# Final: 1.0
```

### Task 1.3: 예금/적금 만기일 확인
```python
Task(
    id="maturity_check",
    description="내 적금 만기일이 언제야?",
    difficulty=0.15,
    category="inquiry"
)
```

---

## 2. 이체 (Easy-Medium)

### Task 2.1: 단순 이체
```python
Task(
    id="simple_transfer",
    description="엄마한테 50만원 보내줘",
    difficulty=0.3,
    category="transfer"
)

# State
state = {
    "user_id": "U12345",
    "authenticated": True,
    "accounts": [
        {"account_no": "110-XXX-XXXXX", "type": "입출금", "balance": 1500000, "holder": "홍길동"}
    ],
    "contacts": [
        {"name": "엄마", "account_no": "220-YYY-YYYYY", "bank": "국민은행", "holder": "김순자"}
    ],
    "transaction_history": [
        # 과거 이체 내역 (수취인 검색 소스)
        {"date": "2024-11-15", "type": "이체", "to_name": "엄마", "to_account": "220-YYY-YYYYY", "to_bank": "국민은행", "amount": 300000},
        {"date": "2024-11-10", "type": "이체", "to_name": "동생", "to_account": "330-ZZZ-ZZZZZ", "to_bank": "신한은행", "amount": 200000},
        {"date": "2024-10-25", "type": "이체", "to_name": "김철수", "to_account": "440-AAA-AAAAA", "to_bank": "하나은행", "amount": 500000}
    ],
    "daily_limit": 5000000,
    "today_transferred": 0
}

# Expected trajectory
# === 송금인(보내는 사람) 정보 파악 ===
# Step 1: 송금인 계좌 목록 조회 → reward=1
# Step 2: 출금 계좌 선택 (입출금 계좌) → reward=1
# Step 3: 출금 계좌 잔액 확인 (150만원) → reward=1
# Step 4: 일일 이체 한도 확인 → reward=1

# === 수취인(받는 사람) 정보 파악 ===
# 방법 1: 연락처에서 검색
# Step 5a: 연락처에서 수취인 검색 ("엄마") → reward=1
#
# 방법 2: 거래내역에서 검색 (연락처에 없는 경우 또는 "지난번에 보낸 사람" 요청 시)
# Step 5b: 거래내역 조회 → reward=1
# Step 5c: 거래내역에서 수취인 검색 ("엄마" 또는 "최근 이체") → reward=1
#
# Step 6: 수취인 계좌번호 확인 (220-YYY-YYYYY) → reward=1
# Step 7: 수취인 은행 확인 (국민은행) → reward=1
# Step 8: 수취인 예금주명 확인 (김순자) → reward=1

# === 이체 실행 ===
# Step 9: 이체 금액 확인 (50만원) → reward=1
# Step 10: 이체 정보 최종 확인 (송금인→수취인 요약) → reward=1
# Step 11: 이체 실행 → reward=1
# Step 12: 결과 안내 (잔액, 이체내역) → reward=1
# Final: 1.0
```

**수취인 검색 소스:**
```
┌─────────────────────────────────────────────────────────┐
│              수취인 정보 검색 소스                       │
├─────────────────────────────────────────────────────────┤
│  [1] 연락처 (contacts)                                  │
│      - 저장된 별칭으로 검색: "엄마", "동생"              │
│      - 예금주명으로 검색: "김순자"                       │
│      - 계좌번호로 검색: "220-YYY-YYYYY"                 │
├─────────────────────────────────────────────────────────┤
│  [2] 거래내역 (transaction_history)                     │
│      - 최근 이체 기록에서 검색                          │
│      - "지난번에 보낸 사람", "저번 주에 이체한 계좌"     │
│      - 이름/계좌번호 일부 매칭                          │
│      - 연락처에 없는 수취인도 검색 가능                  │
└─────────────────────────────────────────────────────────┘
```

**이체 정보 확인 프로세스:**
```
┌─────────────────────────────────────────────────────────┐
│                    이체 정보 확인                        │
├─────────────────────────────────────────────────────────┤
│  [송금인 정보]              [수취인 정보]                │
│  ・계좌: 110-XXX-XXXXX      ・계좌: 220-YYY-YYYYY       │
│  ・은행: 우리은행            ・은행: 국민은행            │
│  ・예금주: 홍길동            ・예금주: 김순자            │
│  ・잔액: 1,500,000원        ・별칭: 엄마                │
├─────────────────────────────────────────────────────────┤
│  이체금액: 500,000원                                    │
│  이체 후 잔액: 1,000,000원                              │
└─────────────────────────────────────────────────────────┘
```

**Actions:**
```python
available_actions = [
    # 수취인 검색 - 연락처
    {"tool": "search_contact", "params": {"query": str}},  # 별칭, 이름, 계좌번호로 검색

    # 수취인 검색 - 거래내역
    {"tool": "search_transaction_history", "params": {
        "query": str,           # 검색어 (이름, 계좌번호 일부)
        "transaction_type": str, # "이체", "입금", "출금"
        "date_range": str       # "최근 1개월", "최근 3개월"
    }},
    {"tool": "get_recent_transfers", "params": {"limit": int}},  # 최근 N건 이체내역

    # 이체 관련
    {"tool": "get_transfer_limit", "params": {"account_no": str}},
    {"tool": "execute_transfer", "params": {
        "from_account": str,
        "to_account": str,
        "to_bank": str,
        "amount": int,
        "memo": str
    }},
    {"tool": "request_otp", "params": {}},
    {"tool": "verify_otp", "params": {"otp": str}},
    {"tool": "respond", "params": {"message": str}}
]
```

### Task 2.2: 다중 이체
```python
Task(
    id="multi_transfer",
    description="엄마한테 50만원, 동생한테 30만원 보내줘",
    difficulty=0.4,
    category="transfer"
)

# Expected trajectory
# Step 1: 수취인들 확인 → reward=1
# Step 2: 총 금액 계산 (80만원) → reward=1
# Step 3: 잔액 및 한도 확인 → reward=1
# Step 4: 첫 번째 이체 실행 → reward=1
# Step 5: 두 번째 이체 실행 → reward=1
# Step 6: 결과 종합 안내 → reward=1
# Final: 1.0
```

### Task 2.3: 예약 이체
```python
Task(
    id="scheduled_transfer",
    description="매월 1일에 월세 85만원 자동이체 설정해줘",
    difficulty=0.45,
    category="transfer"
)

# State 추가 필드
state["scheduled_transfers"] = []
state["landlord"] = {"name": "임대인", "account_no": "333-ZZZ-ZZZZZ", "bank": "신한은행"}

# Expected trajectory
# Step 1: 수취인 정보 확인 → reward=1
# Step 2: 자동이체 조건 설정 (매월 1일) → reward=1
# Step 3: 출금 계좌 선택 → reward=1
# Step 4: 자동이체 등록 → reward=1
# Step 5: 등록 확인 및 안내 → reward=1
# Final: 1.0
```

### Task 2.4: 이체 한도 변경 후 이체
```python
Task(
    id="limit_change_transfer",
    description="오늘 전세금 2억 보내야 하는데 한도 늘려서 이체해줘",
    difficulty=0.6,
    category="transfer"
)

# 복합 태스크: 한도 조회 → 한도 변경 → 이체
# Step 1: 현재 한도 조회 → reward=1
# Step 2: 한도 변경 가능 여부 확인 → reward=1
# Step 3: 본인 인증 (OTP/생체) → reward=1
# Step 4: 한도 변경 신청 → reward=1
# Step 5: 변경 승인 대기/확인 → reward=1
# Step 6: 이체 실행 → reward=1
# Step 7: 결과 안내 → reward=1
# Final: 1.0
```

---

## 3. 모임통장 (Medium)

### Task 3.1: 모임통장 개설
```python
Task(
    id="group_account_create",
    description="동창회 모임통장 만들어줘. 회비는 월 3만원이야.",
    difficulty=0.5,
    category="group_account"
)

# State
state = {
    "user_id": "U12345",
    "authenticated": True,
    "can_create_group_account": True,
    "group_info": {
        "name": "동창회",
        "monthly_fee": 30000,
        "members": []
    }
}

# Expected trajectory
# Step 1: 모임통장 개설 자격 확인 → reward=1
# Step 2: 모임 정보 설정 (이름, 회비) → reward=1
# Step 3: 총무 권한 설정 → reward=1
# Step 4: 계좌 개설 실행 → reward=1
# Step 5: 초대 링크 생성 → reward=1
# Step 6: 결과 안내 → reward=1
# Final: 1.0
```

### Task 3.2: 회원 초대 및 회비 관리
```python
Task(
    id="group_member_invite",
    description="동창회 모임통장에 김철수, 이영희 초대하고 이번 달 회비 납부 현황 알려줘",
    difficulty=0.55,
    category="group_account"
)

# Expected trajectory
# Step 1: 모임통장 조회 → reward=1
# Step 2: 회원 초대 (김철수) → reward=1
# Step 3: 회원 초대 (이영희) → reward=1
# Step 4: 회비 납부 현황 조회 → reward=1
# Step 5: 미납자 목록 생성 → reward=1
# Step 6: 종합 결과 안내 → reward=1
# Final: 1.0
```

### Task 3.3: 모임통장 총무 업무
```python
Task(
    id="group_treasurer_task",
    description="이번 달 회비 미납자한테 독촉 메시지 보내고, 회식비 40만원 정산해줘",
    difficulty=0.65,
    category="group_account"
)

# State
state = {
    "user_id": "U12345",
    "role": "treasurer",  # 총무
    "group_account": {
        "account_no": "777-GRP-XXXXX",
        "balance": 850000,
        "members": [
            {"name": "김철수", "paid": True, "amount": 30000},
            {"name": "이영희", "paid": True, "amount": 30000},
            {"name": "박지민", "paid": False, "amount": 0},
            {"name": "최수진", "paid": False, "amount": 0}
        ],
        "monthly_fee": 30000
    },
    "pending_expense": {
        "description": "회식비",
        "amount": 400000,
        "paid_by": "김철수"
    }
}

# Expected trajectory
# Step 1: 미납자 목록 확인 → reward=1
# Step 2: 독촉 메시지 발송 (박지민) → reward=1
# Step 3: 독촉 메시지 발송 (최수진) → reward=1
# Step 4: 정산 금액 계산 (40만원 ÷ 4명) → reward=1
# Step 5: 정산 요청 생성 → reward=1
# Step 6: 김철수에게 선지급금 이체 → reward=1
# Step 7: 결과 종합 안내 → reward=1
# Final: 1.0
```

### Task 3.4: 모임통장 정기 보고
```python
Task(
    id="group_monthly_report",
    description="이번 달 모임통장 수입/지출 내역 정리해서 회원들한테 공유해줘",
    difficulty=0.6,
    category="group_account"
)

# Expected trajectory
# Step 1: 이번 달 거래내역 조회 → reward=1
# Step 2: 수입 항목 분류 (회비, 기타) → reward=1
# Step 3: 지출 항목 분류 (회식, 경조사, 기타) → reward=1
# Step 4: 보고서 생성 → reward=1
# Step 5: 회원들에게 공유 → reward=1
# Final: 1.0
```

---

## 4. 예금/적금 (Medium)

### Task 4.1: 적금 추천
```python
Task(
    id="savings_recommendation",
    description="월 50만원씩 1년 넣을 건데 적금 추천해줘",
    difficulty=0.45,
    category="savings"
)

# State
state = {
    "user_id": "U12345",
    "authenticated": True,
    "customer_grade": "우수",
    "available_products": [
        {
            "name": "자유적금",
            "type": "적금",
            "base_rate": 3.5,
            "bonus_rate": 0.5,  # 우대금리
            "min_amount": 10000,
            "max_amount": 1000000,
            "term_months": [6, 12, 24, 36]
        },
        {
            "name": "청년희망적금",
            "type": "적금",
            "base_rate": 4.5,
            "bonus_rate": 1.0,
            "min_amount": 50000,
            "max_amount": 500000,
            "term_months": [12, 24],
            "eligibility": "만 19-34세"
        }
    ],
    "user_profile": {
        "age": 28,
        "monthly_income": 3500000
    }
}

# Expected trajectory
# Step 1: 사용자 조건 파악 (월 50만원, 1년) → reward=1
# Step 2: 적격 상품 필터링 → reward=1
# Step 3: 금리 비교 (세전/세후) → reward=1
# Step 4: 만기 수령액 계산 → reward=1
# Step 5: 추천 상품 안내 (비교표) → reward=1
# Final: 1.0
```

### Task 4.2: 적금 가입
```python
Task(
    id="savings_subscription",
    description="청년희망적금 가입해줘. 월 50만원씩 2년.",
    difficulty=0.5,
    category="savings"
)

# Expected trajectory
# Step 1: 상품 상세 조회 → reward=1
# Step 2: 가입 자격 확인 (나이, 소득 등) → reward=1
# Step 3: 약관 동의 처리 → reward=1
# Step 4: 자동이체 계좌 설정 → reward=1
# Step 5: 가입 실행 → reward=1
# Step 6: 가입 확인 및 안내 → reward=1
# Final: 1.0
```

### Task 4.3: 예금 만기 처리
```python
Task(
    id="deposit_maturity",
    description="내일 만기되는 정기예금 있는데, 금리 좋은 상품으로 재예치해줘",
    difficulty=0.55,
    category="savings"
)

# State
state = {
    "user_id": "U12345",
    "maturing_deposits": [
        {
            "account_no": "DEP-12345",
            "product_name": "정기예금 1년",
            "principal": 10000000,
            "interest": 350000,
            "maturity_date": "2024-11-23",
            "current_rate": 3.5
        }
    ],
    "available_deposits": [
        {"name": "정기예금 1년", "rate": 3.8, "term": 12},
        {"name": "정기예금 2년", "rate": 4.0, "term": 24},
        {"name": "특판예금", "rate": 4.5, "term": 12, "limit": 50000000}
    ]
}

# Expected trajectory
# Step 1: 만기 예금 확인 → reward=1
# Step 2: 만기금액 계산 (원금 + 이자) → reward=1
# Step 3: 현재 예금 상품 비교 → reward=1
# Step 4: 최적 상품 선택 → reward=1
# Step 5: 재예치 실행 → reward=1
# Step 6: 결과 안내 → reward=1
# Final: 1.0
```

### Task 4.4: 중도해지 상담
```python
Task(
    id="early_termination",
    description="적금 깨야 하는데 손해 얼마나 보는지 알려주고, 대안 있으면 추천해줘",
    difficulty=0.6,
    category="savings"
)

# Expected trajectory
# Step 1: 적금 현황 조회 → reward=1
# Step 2: 중도해지 시 손실 계산 → reward=1
# Step 3: 대안 검토 (적금담보대출 등) → reward=1
# Step 4: 비교 분석 제공 → reward=1
# Step 5: 사용자 선택 안내 → reward=1
# Final: 1.0
```

---

## 5. 대출 (Hard)

### Task 5.1: 대출 가능 금액 조회
```python
Task(
    id="loan_eligibility",
    description="신용대출 얼마까지 받을 수 있어?",
    difficulty=0.6,
    category="loan"
)

# State
state = {
    "user_id": "U12345",
    "authenticated": True,
    "credit_info": {
        "credit_score": 850,
        "credit_grade": "1등급",
        "annual_income": 60000000,
        "existing_loans": [
            {"type": "학자금대출", "balance": 5000000, "monthly_payment": 200000}
        ],
        "dti_ratio": 0.15  # 총부채상환비율
    },
    "loan_products": [
        {
            "name": "직장인신용대출",
            "max_amount": 100000000,
            "rate_range": [4.5, 7.0],
            "max_dti": 0.4
        }
    ]
}

# Expected trajectory
# Step 1: 신용정보 조회 → reward=1
# Step 2: 소득 정보 확인 → reward=1
# Step 3: 기존 대출 현황 파악 → reward=1
# Step 4: DSR/DTI 계산 → reward=1
# Step 5: 대출 가능 금액 산출 → reward=1
# Step 6: 예상 금리 및 상환액 안내 → reward=1
# Final: 1.0
```

### Task 5.2: 신용대출 신청
```python
Task(
    id="credit_loan_apply",
    description="신용대출 3천만원 신청해줘. 36개월 상환으로.",
    difficulty=0.7,
    category="loan"
)

# Expected trajectory
# Step 1: 대출 가능 여부 확인 → reward=1
# Step 2: 금리 및 조건 확인 → reward=1
# Step 3: 상환 스케줄 시뮬레이션 → reward=1
# Step 4: 서류 제출 (소득증빙 등) → reward=1
# Step 5: 약관 동의 → reward=1
# Step 6: 대출 신청 실행 → reward=1
# Step 7: 심사 결과 안내 → reward=1
# Final: 1.0
```

### Task 5.3: 대출 금리 비교 및 갈아타기
```python
Task(
    id="loan_refinance",
    description="지금 대출 금리가 7%인데 더 낮은 데로 갈아탈 수 있어?",
    difficulty=0.75,
    category="loan"
)

# State
state = {
    "user_id": "U12345",
    "current_loan": {
        "type": "신용대출",
        "balance": 20000000,
        "rate": 7.0,
        "remaining_months": 24,
        "early_repayment_fee_rate": 0.02
    },
    "refinance_options": [
        {"bank": "A은행", "rate": 5.5, "max_amount": 30000000},
        {"bank": "B은행", "rate": 5.8, "max_amount": 50000000}
    ]
}

# Expected trajectory
# Step 1: 현재 대출 조건 확인 → reward=1
# Step 2: 중도상환 수수료 계산 → reward=1
# Step 3: 타행 대출 상품 조회 → reward=1
# Step 4: 금리 비교 분석 → reward=1
# Step 5: 총 비용 비교 (수수료 포함) → reward=1
# Step 6: 갈아타기 추천 여부 안내 → reward=1
# Step 7: (선택) 대환대출 신청 → reward=1
# Final: 1.0
```

---

## 6. 주택담보대출 (Expert)

### Task 6.1: 주담대 한도 조회
```python
Task(
    id="mortgage_limit",
    description="아파트 담보로 대출 얼마까지 가능해? 시세 8억짜리야.",
    difficulty=0.8,
    category="mortgage"
)

# State
state = {
    "user_id": "U12345",
    "authenticated": True,
    "property_info": {
        "type": "아파트",
        "address": "서울시 강남구 XXX",
        "market_value": 800000000,  # 시세 8억
        "kb_price": 780000000,      # KB시세
        "owned": True
    },
    "credit_info": {
        "credit_score": 850,
        "annual_income": 80000000,
        "existing_mortgage": 0,
        "other_loans": 10000000
    },
    "regulations": {
        "ltv_limit": 0.5,      # 투기지역 LTV 50%
        "dti_limit": 0.4,      # DTI 40%
        "dsr_limit": 0.4       # DSR 40%
    }
}

# Expected trajectory
# Step 1: 담보물건 정보 확인 → reward=1
# Step 2: KB시세/감정가 조회 → reward=1
# Step 3: LTV 적용 한도 계산 → reward=1
# Step 4: DTI 적용 한도 계산 → reward=1
# Step 5: DSR 적용 한도 계산 → reward=1
# Step 6: 최종 한도 (min값) 산출 → reward=1
# Step 7: 예상 금리 및 상환액 안내 → reward=1
# Final: 1.0
```

### Task 6.2: 주담대 신청
```python
Task(
    id="mortgage_apply",
    description="주담대 4억 신청하고 싶어. 30년 원리금균등상환으로.",
    difficulty=0.85,
    category="mortgage"
)

# Expected trajectory
# Step 1: 대출 가능 여부 확인 (LTV/DTI/DSR) → reward=1
# Step 2: 금리 유형 선택 (고정/변동) → reward=1
# Step 3: 상환 스케줄 시뮬레이션 → reward=1
# Step 4: 필요 서류 안내 → reward=1
# Step 5: 서류 제출 (등기부등본, 소득증빙 등) → reward=1
# Step 6: 약관 동의 → reward=1
# Step 7: 대출 신청 실행 → reward=1
# Step 8: 심사 일정 안내 → reward=1
# Final: 1.0
```

### Task 6.3: 주담대 상환 전략
```python
Task(
    id="mortgage_repayment_strategy",
    description="주담대 잔액 3억 있는데, 여윳돈 5천만원 생겼어. 중도상환할까 투자할까?",
    difficulty=0.9,
    category="mortgage"
)

# State
state = {
    "user_id": "U12345",
    "mortgage": {
        "balance": 300000000,
        "rate": 4.5,
        "remaining_years": 25,
        "monthly_payment": 1700000,
        "early_repayment_fee": 0.012  # 1.2%
    },
    "available_funds": 50000000,
    "investment_options": [
        {"type": "예금", "rate": 4.0, "risk": "low"},
        {"type": "채권펀드", "expected_return": 5.5, "risk": "medium"},
        {"type": "주식ETF", "expected_return": 8.0, "risk": "high"}
    ]
}

# Expected trajectory
# Step 1: 현재 대출 조건 분석 → reward=1
# Step 2: 중도상환 효과 계산 (이자 절감액) → reward=1
# Step 3: 중도상환 수수료 계산 → reward=1
# Step 4: 투자 수익 시뮬레이션 → reward=1
# Step 5: 세후 수익률 비교 → reward=1
# Step 6: 리스크 요인 설명 → reward=1
# Step 7: 추천 전략 안내 (분할 등) → reward=1
# Final: 1.0
```

### Task 6.4: 전세자금대출 → 주담대 전환
```python
Task(
    id="jeonse_to_mortgage",
    description="전세 끝나면 이 집 매수하려고 해. 전세대출을 주담대로 바꿀 수 있어?",
    difficulty=0.9,
    category="mortgage"
)

# 복잡한 시나리오: 전세대출 상환 + 매매 + 주담대 신규
# Step 1: 현재 전세대출 현황 확인 → reward=1
# Step 2: 매매 예정 물건 정보 확인 → reward=1
# Step 3: 주담대 한도 조회 (LTV/DTI/DSR) → reward=1
# Step 4: 자금 계획 수립 → reward=1
# Step 5: 전세대출 상환 일정 조율 → reward=1
# Step 6: 주담대 신청 절차 안내 → reward=1
# Step 7: 필요 서류 및 일정 안내 → reward=1
# Final: 1.0
```

---

## 7. 복합 금융 서비스 (Expert)

### Task 7.1: 종합 자산 관리
```python
Task(
    id="wealth_management",
    description="여유자금 1억으로 포트폴리오 구성해줘. 중위험 중수익으로.",
    difficulty=0.85,
    category="investment"
)

# Expected trajectory
# Step 1: 투자 성향 확인 → reward=1
# Step 2: 현재 자산 현황 파악 → reward=1
# Step 3: 목표 수익률/위험 수준 설정 → reward=1
# Step 4: 자산배분 전략 수립 → reward=1
# Step 5: 상품별 추천 (예금/채권/펀드/ETF) → reward=1
# Step 6: 예상 수익/손실 시뮬레이션 → reward=1
# Step 7: 투자 실행 (사용자 승인 시) → reward=1
# Final: 1.0
```

### Task 7.2: 은퇴 자금 설계
```python
Task(
    id="retirement_planning",
    description="15년 후 은퇴인데 노후 자금 계획 세워줘. 월 300만원은 필요해.",
    difficulty=0.9,
    category="planning"
)

# State
state = {
    "user_id": "U12345",
    "age": 50,
    "retirement_age": 65,
    "target_monthly_expense": 3000000,
    "current_assets": {
        "deposits": 100000000,
        "pension": 50000000,  # 연금저축
        "real_estate": 500000000
    },
    "monthly_savings": 2000000
}

# Expected trajectory
# Step 1: 은퇴 목표 확인 → reward=1
# Step 2: 현재 자산 분석 → reward=1
# Step 3: 필요 노후자금 계산 (인플레이션 반영) → reward=1
# Step 4: 부족 금액 산출 → reward=1
# Step 5: 저축/투자 전략 수립 → reward=1
# Step 6: 연금 상품 추천 → reward=1
# Step 7: 실행 계획 안내 → reward=1
# Final: 1.0
```

### Task 7.3: 대출 + 투자 복합 상담
```python
Task(
    id="loan_investment_combo",
    description="주담대 금리가 4.5%인데, 대출 유지하면서 여유자금으로 투자하는 게 나을까? 아니면 상환이 나을까?",
    difficulty=0.95,
    category="planning"
)

# Expected trajectory
# Step 1: 대출 조건 분석 → reward=1
# Step 2: 세후 실질 금리 계산 → reward=1
# Step 3: 투자 예상 수익률 분석 → reward=1
# Step 4: 레버리지 효과 시뮬레이션 → reward=1
# Step 5: 리스크 시나리오 분석 → reward=1
# Step 6: 세금 영향 계산 → reward=1
# Step 7: 종합 추천 및 근거 제시 → reward=1
# Final: 1.0
```

---

## 8. 에러 처리 및 예외 상황

### Task 8.1: 잔액 부족 처리
```python
Task(
    id="insufficient_balance",
    description="100만원 이체해줘",  # 잔액 50만원일 때
    difficulty=0.35,
    category="error_handling"
)

# State
state = {
    "accounts": [
        {"account_no": "110-XXX", "balance": 500000}  # 50만원
    ]
}

# Expected trajectory
# Step 1: 이체 금액 확인 (100만원) → reward=1
# Step 2: 잔액 확인 (50만원) → reward=1
# Step 3: 부족 금액 안내 (50만원 부족) → reward=1
# Step 4: 대안 제시 (다른 계좌, 금액 조정) → reward=1
# Final: 1.0 (에러 상황을 적절히 처리)
```

### Task 8.2: 한도 초과 처리
```python
Task(
    id="limit_exceeded",
    description="1억 이체해줘",  # 일일 한도 5천만원일 때
    difficulty=0.4,
    category="error_handling"
)

# Expected trajectory
# Step 1: 이체 금액 확인 → reward=1
# Step 2: 일일 한도 확인 → reward=1
# Step 3: 한도 초과 안내 → reward=1
# Step 4: 대안 제시 (한도 상향, 분할 이체) → reward=1
# Final: 1.0
```

### Task 8.3: 자격 미달 처리
```python
Task(
    id="eligibility_failed",
    description="청년희망적금 가입해줘",  # 나이 초과일 때
    difficulty=0.4,
    category="error_handling"
)

# State
state = {
    "user_profile": {"age": 40},  # 34세 초과
    "product": {"eligibility": "만 19-34세"}
}

# Expected trajectory
# Step 1: 상품 조건 확인 → reward=1
# Step 2: 사용자 자격 확인 → reward=1
# Step 3: 자격 미달 안내 (나이 초과) → reward=1
# Step 4: 대안 상품 추천 → reward=1
# Final: 1.0
```

---

## 9. 멀티턴 대화 시나리오

### Task 9.1: 상담 후 가입
```python
Task(
    id="consult_then_subscribe",
    description="적금 상담받고 싶어",  # 여러 턴의 대화 후 가입까지
    difficulty=0.6,
    category="multi_turn"
)

# Multi-turn conversation
conversation = [
    {"role": "user", "content": "적금 상담받고 싶어"},
    {"role": "assistant", "content": "네, 어떤 목적으로 적금을 찾고 계신가요?"},
    {"role": "user", "content": "결혼자금 모으려고"},
    {"role": "assistant", "content": "결혼은 언제쯤 계획하고 계신가요?"},
    {"role": "user", "content": "2년 후에"},
    {"role": "assistant", "content": "월 얼마 정도 저축 가능하신가요?"},
    {"role": "user", "content": "100만원 정도?"},
    {"role": "assistant", "content": "[상품 추천]"},
    {"role": "user", "content": "첫번째 거 가입할게"},
    # → 가입 프로세스 진행
]

# 각 턴이 목표(결혼자금 적금 가입)에 기여하면 reward=1
```

### Task 9.2: 조건 변경 대화
```python
Task(
    id="condition_change",
    description="대출 상담 → 조건 변경 → 재계산",
    difficulty=0.7,
    category="multi_turn"
)

# Multi-turn with condition changes
conversation = [
    {"role": "user", "content": "신용대출 5천만원 받고 싶어"},
    {"role": "assistant", "content": "[한도 및 금리 안내]"},
    {"role": "user", "content": "금리가 높은데, 3천만원으로 줄이면?"},
    {"role": "assistant", "content": "[재계산 결과]"},
    {"role": "user", "content": "상환 기간을 48개월로 늘리면?"},
    {"role": "assistant", "content": "[재계산 결과]"},
    {"role": "user", "content": "그걸로 신청할게"},
    # → 신청 프로세스 진행
]
```

---

## 10. Tools 정의

### 계좌 관련
```python
tools = {
    "get_account_list": {
        "description": "사용자의 계좌 목록 조회",
        "params": {"user_id": str},
        "returns": "List[Account]"
    },
    "get_balance": {
        "description": "특정 계좌의 잔액 조회",
        "params": {"account_no": str},
        "returns": "int"
    },
    "get_transaction_history": {
        "description": "거래 내역 조회",
        "params": {"account_no": str, "start_date": str, "end_date": str},
        "returns": "List[Transaction]"
    },
    "search_transaction_history": {
        "description": "거래내역에서 수취인/송금인 검색 (이름, 계좌번호 일부 매칭)",
        "params": {
            "query": str,            # 검색어 (이름, 계좌번호 일부)
            "transaction_type": str,  # "이체", "입금", "출금", "전체"
            "date_range": str         # "최근 1개월", "최근 3개월", "최근 1년"
        },
        "returns": "List[Transaction]"
    },
    "get_recent_transfers": {
        "description": "최근 이체 내역 조회 (수취인 정보 포함)",
        "params": {"limit": int},  # 조회할 건수
        "returns": "List[TransferRecord]"  # to_name, to_account, to_bank, amount, date
    },
    "search_contact": {
        "description": "연락처에서 수취인 검색 (별칭, 이름, 계좌번호)",
        "params": {"query": str},
        "returns": "List[Contact]"
    }
}
```

### 이체 관련
```python
tools.update({
    "execute_transfer": {
        "description": "계좌 이체 실행",
        "params": {
            "from_account": str,
            "to_account": str,
            "to_bank": str,
            "amount": int,
            "memo": str
        },
        "returns": "TransferResult"
    },
    "schedule_transfer": {
        "description": "예약/자동 이체 설정",
        "params": {
            "from_account": str,
            "to_account": str,
            "amount": int,
            "schedule": str  # "매월 1일", "매주 월요일" 등
        },
        "returns": "ScheduleResult"
    },
    "get_transfer_limit": {
        "description": "이체 한도 조회",
        "params": {"account_no": str},
        "returns": "LimitInfo"
    },
    "change_transfer_limit": {
        "description": "이체 한도 변경 신청",
        "params": {"account_no": str, "new_limit": int},
        "returns": "ChangeResult"
    }
})
```

### 대출 관련
```python
tools.update({
    "get_loan_eligibility": {
        "description": "대출 가능 금액 조회",
        "params": {"user_id": str, "loan_type": str},
        "returns": "EligibilityInfo"
    },
    "calculate_loan_schedule": {
        "description": "대출 상환 스케줄 계산",
        "params": {
            "amount": int,
            "rate": float,
            "term_months": int,
            "repayment_type": str  # "원리금균등", "원금균등"
        },
        "returns": "RepaymentSchedule"
    },
    "apply_loan": {
        "description": "대출 신청",
        "params": {
            "loan_type": str,
            "amount": int,
            "term_months": int,
            "repayment_type": str
        },
        "returns": "ApplicationResult"
    },
    "get_mortgage_limit": {
        "description": "주담대 한도 조회 (LTV/DTI/DSR)",
        "params": {"property_id": str, "user_id": str},
        "returns": "MortgageLimitInfo"
    }
})
```

### 상품 관련
```python
tools.update({
    "get_product_list": {
        "description": "금융 상품 목록 조회",
        "params": {"category": str},  # "적금", "예금", "대출" 등
        "returns": "List[Product]"
    },
    "get_product_detail": {
        "description": "상품 상세 정보 조회",
        "params": {"product_id": str},
        "returns": "ProductDetail"
    },
    "subscribe_product": {
        "description": "상품 가입",
        "params": {
            "product_id": str,
            "amount": int,
            "options": dict
        },
        "returns": "SubscriptionResult"
    },
    "calculate_maturity": {
        "description": "만기 수령액 계산",
        "params": {
            "product_id": str,
            "monthly_amount": int,
            "term_months": int
        },
        "returns": "MaturityInfo"
    }
})
```

### 모임통장 관련
```python
tools.update({
    "create_group_account": {
        "description": "모임통장 개설",
        "params": {"group_name": str, "monthly_fee": int},
        "returns": "GroupAccount"
    },
    "invite_member": {
        "description": "모임통장 회원 초대",
        "params": {"group_id": str, "member_info": dict},
        "returns": "InviteResult"
    },
    "get_fee_status": {
        "description": "회비 납부 현황 조회",
        "params": {"group_id": str, "month": str},
        "returns": "FeeStatus"
    },
    "send_reminder": {
        "description": "회비 독촉 메시지 발송",
        "params": {"group_id": str, "member_ids": List[str]},
        "returns": "ReminderResult"
    },
    "create_settlement": {
        "description": "정산 요청 생성",
        "params": {
            "group_id": str,
            "amount": int,
            "description": str,
            "paid_by": str
        },
        "returns": "SettlementResult"
    }
})
```

---

## 11. 커리큘럼 구성

### 난이도별 태스크 분포
```python
curriculum = {
    "easy": [  # 0.1-0.3
        "balance_inquiry",
        "transaction_history",
        "maturity_check",
        "simple_transfer"
    ],
    "medium": [  # 0.4-0.6
        "multi_transfer",
        "scheduled_transfer",
        "group_account_create",
        "group_member_invite",
        "savings_recommendation",
        "savings_subscription",
        "deposit_maturity",
        "early_termination"
    ],
    "hard": [  # 0.7-0.9
        "limit_change_transfer",
        "group_treasurer_task",
        "loan_eligibility",
        "credit_loan_apply",
        "loan_refinance",
        "mortgage_limit"
    ],
    "expert": [  # 0.9+
        "mortgage_apply",
        "mortgage_repayment_strategy",
        "jeonse_to_mortgage",
        "wealth_management",
        "retirement_planning",
        "loan_investment_combo"
    ]
}
```

### 학습 순서 예시
```python
# Phase 1: 기본기 (Easy)
# - 조회 기능 완벽 습득
# - 단순 이체 마스터

# Phase 2: 확장 (Medium)
# - 복합 이체 (다중, 예약)
# - 모임통장 기본
# - 예금/적금 상담 및 가입

# Phase 3: 심화 (Hard)
# - 대출 상담 및 신청
# - 모임통장 총무 업무
# - 금융 상품 비교 분석

# Phase 4: 전문가 (Expert)
# - 주택담보대출 전체 프로세스
# - 종합 자산 관리
# - 복합 금융 의사결정
```

---

## 12. 평가 지표

### Task 성공률
```python
def evaluate_task_success(trajectory, task) -> bool:
    """
    모든 스텝이 목표에 기여했는지 확인
    """
    for step in trajectory.steps:
        if step.reward != 1.0:
            return False
    return True
```

### 효율성 지표
```python
def evaluate_efficiency(trajectory, optimal_steps) -> float:
    """
    최적 스텝 수 대비 실제 스텝 수
    """
    return optimal_steps / len(trajectory.steps)
```

### 사용자 만족도 (시뮬레이션)
```python
def evaluate_satisfaction(trajectory, task) -> float:
    """
    - 정확성: 요청한 작업이 정확히 수행되었는가?
    - 완전성: 필요한 정보가 모두 제공되었는가?
    - 적시성: 불필요한 확인 없이 빠르게 처리되었는가?
    """
    accuracy = check_accuracy(trajectory, task)
    completeness = check_completeness(trajectory, task)
    timeliness = check_timeliness(trajectory, task)

    return (accuracy + completeness + timeliness) / 3
```

---

## Summary

| 카테고리 | 태스크 수 | 난이도 범위 |
|---------|---------|-----------|
| 계좌 조회 | 3 | 0.1-0.2 |
| 이체 | 4 | 0.3-0.6 |
| 모임통장 | 4 | 0.5-0.65 |
| 예금/적금 | 4 | 0.45-0.6 |
| 대출 | 3 | 0.6-0.75 |
| 주택담보대출 | 4 | 0.8-0.9 |
| 복합 금융 | 3 | 0.85-0.95 |
| 에러 처리 | 3 | 0.35-0.4 |
| 멀티턴 | 2 | 0.6-0.7 |
| **총계** | **30** | **0.1-0.95** |

이 태스크들을 DreamGym 커리큘럼에 적용하여 뱅킹 에이전트를 학습시킬 수 있습니다.
