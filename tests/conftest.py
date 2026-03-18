"""
공유 픽스처 — 모든 테스트 파일에서 사용.
"""
import pytest
from graph.state import WorkflowState, AgentOutput, default_agent_output


def make_agent_output(
    task_id: str,
    content: str = "",
    sources: list[str] | None = None,
    quantitative_check: bool = False,
    fallback_items: list[str] | None = None,
) -> AgentOutput:
    return AgentOutput(
        task_id=task_id,
        content=content,
        sources=sources or [],
        quantitative_check=quantitative_check,
        fallback_items=fallback_items or [],
    )


def make_state(**overrides) -> WorkflowState:
    """기본 WorkflowState를 생성하고 overrides로 필드를 덮어쓴다."""
    base: WorkflowState = {
        "user_query": "EV 캐즘 환경에서 LGES와 CATL 포트폴리오 전략 비교",
        "market_research": default_agent_output("T1"),
        "lges_strategy": default_agent_output("T2"),
        "catl_strategy": default_agent_output("T3"),
        "critic_result": default_agent_output("T4"),
        "swot_comparison": default_agent_output("T5"),
        "final_report": default_agent_output("T6"),
        "current_task": "INIT",
        "retry_count": {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        "failed_criteria": [],
        "fallback_items": [],
        "total_llm_calls": 0,
        "is_completed": False,
        "termination_reason": "",
    }
    base.update(overrides)
    return base


# ── 자주 쓰이는 픽스처 ──────────────────────────────────────

LONG_CONTENT = "배터리 전략 분석 내용입니다. " * 100  # ~1400자


@pytest.fixture
def base_state():
    return make_state()


@pytest.fixture
def t1_pass_state():
    """T1 완료 (quantitative_check=True) 상태."""
    return make_state(
        current_task="T1",
        market_research=make_agent_output(
            "T1",
            content=LONG_CONTENT,
            sources=["url1", "url2", "url3"],
            quantitative_check=True,
        ),
        retry_count={"T1": 1, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        total_llm_calls=1,
    )


@pytest.fixture
def t1_fail_state():
    """T1 완료 (quantitative_check=False, retry=0) 상태."""
    return make_state(
        current_task="T1",
        market_research=make_agent_output(
            "T1",
            content="짧은 내용",
            sources=["url1"],
            quantitative_check=False,
            fallback_items=["출처 부족"],
        ),
        retry_count={"T1": 1, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        total_llm_calls=1,
    )


@pytest.fixture
def t2_t3_pass_state():
    """T2·T3 모두 완료 (quantitative_check=True) 상태."""
    return make_state(
        current_task="T2",
        market_research=make_agent_output("T1", content=LONG_CONTENT, quantitative_check=True),
        lges_strategy=make_agent_output(
            "T2", content=LONG_CONTENT, sources=["lges1", "lges2", "lges3"], quantitative_check=True
        ),
        catl_strategy=make_agent_output(
            "T3", content=LONG_CONTENT, sources=["catl1", "catl2", "catl3"], quantitative_check=True
        ),
        retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 0, "T5": 0, "T6": 0},
        total_llm_calls=3,
    )


@pytest.fixture
def t4_pass_state(t2_t3_pass_state):
    """T4 완료 (quantitative_check=True) 상태."""
    t2_t3_pass_state.update({
        "current_task": "T4",
        "critic_result": make_agent_output(
            "T4", content=LONG_CONTENT, sources=["critic1"], quantitative_check=True
        ),
        "retry_count": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 0, "T6": 0},
        "total_llm_calls": 4,
    })
    return t2_t3_pass_state


@pytest.fixture
def t5_pass_state(t4_pass_state):
    """T5 완료 (quantitative_check=True) 상태."""
    t4_pass_state.update({
        "current_task": "T5",
        "swot_comparison": make_agent_output(
            "T5", content=LONG_CONTENT, sources=[], quantitative_check=True
        ),
        "retry_count": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1, "T6": 0},
        "total_llm_calls": 5,
    })
    return t4_pass_state


@pytest.fixture
def t6_pass_state(t5_pass_state):
    """T6 완료 (quantitative_check=True) 상태."""
    t5_pass_state.update({
        "current_task": "T6",
        "final_report": make_agent_output(
            "T6", content=LONG_CONTENT, sources=[], quantitative_check=True
        ),
        "retry_count": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1, "T6": 1},
        "total_llm_calls": 6,
    })
    return t5_pass_state
