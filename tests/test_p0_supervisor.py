"""
P0 테스트: Supervisor 라우팅 로직 + LLM 품질 검토

검증 항목:
  1. supervisor_router — 각 Task 완료 상태별 다음 노드 반환값
  2. supervisor_router — 재시도 / Fallback / 비용 한도 초과 분기
  3. supervisor_node  — LLM 품질 검토 호출 여부 및 결과 반영
  4. termination_reason / is_completed 설정 여부 (P0 버그 수정 검증)
"""
import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import END

from graph.graph import supervisor_router, supervisor_node, termination_node
from tests.conftest import make_state, make_agent_output, LONG_CONTENT

# P0 설계: supervisor_router는 END 대신 "termination" 노드를 반환하고,
# termination_node가 state를 업데이트한 뒤 END로 연결됨.
TERMINAL = "termination"


# ──────────────────────────────────────────────
# 1. supervisor_router — 정상 흐름
# ──────────────────────────────────────────────

class TestSupervisorRouterNormalFlow:

    def test_init_routes_to_market_research(self, base_state):
        """초기 진입(INIT) → market_research로 라우팅."""
        assert supervisor_router(base_state) == "market_research"

    def test_t1_pass_routes_to_parallel(self, t1_pass_state):
        """T1 통과 → T2·T3 병렬 실행 (Send 리스트 반환)."""
        result = supervisor_router(t1_pass_state)
        assert isinstance(result, list), "T2·T3 병렬 실행은 리스트여야 함"
        assert len(result) == 2

    def test_t2_pass_and_t3_done_routes_to_critic(self, t2_t3_pass_state):
        """T2 완료 + T3 이미 완료 → critic으로 라우팅."""
        t2_t3_pass_state["current_task"] = "T2"
        assert supervisor_router(t2_t3_pass_state) == "critic"

    def test_t3_pass_and_t2_done_routes_to_critic(self, t2_t3_pass_state):
        """T3 완료 + T2 이미 완료 → critic으로 라우팅."""
        t2_t3_pass_state["current_task"] = "T3"
        assert supervisor_router(t2_t3_pass_state) == "critic"

    def test_t4_pass_routes_to_swot(self, t4_pass_state):
        """T4 통과 → swot으로 라우팅."""
        assert supervisor_router(t4_pass_state) == "swot"

    def test_t5_pass_routes_to_report(self, t5_pass_state):
        """T5 통과 → report로 라우팅."""
        assert supervisor_router(t5_pass_state) == "report"

    def test_t6_pass_routes_to_termination(self, t6_pass_state):
        """T6 통과 → termination 노드로 라우팅 (termination_node → END)."""
        assert supervisor_router(t6_pass_state) == TERMINAL


# ──────────────────────────────────────────────
# 2. supervisor_router — 재시도 / Fallback 분기
# ──────────────────────────────────────────────

class TestSupervisorRouterRetryAndFallback:

    def test_t1_fail_retry_under_max_retries_to_market_research(self):
        """T1 미통과 + retry < 2 → market_research 재시도."""
        state = make_state(
            current_task="T1",
            market_research=make_agent_output("T1", quantitative_check=False),
            retry_count={"T1": 1, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        )
        assert supervisor_router(state) == "market_research"

    def test_t1_fail_at_max_retries_proceeds_to_parallel(self):
        """T1 미통과 + retry = 2 → Fallback 후 T2·T3 병렬 진행."""
        state = make_state(
            current_task="T1",
            market_research=make_agent_output("T1", quantitative_check=False),
            retry_count={"T1": 2, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        )
        result = supervisor_router(state)
        assert isinstance(result, list), "재시도 소진 시에도 T2·T3 병렬로 진행해야 함"

    def test_t2_fail_retry_under_max_retries_to_lges_strategy(self):
        """T2 미통과 + retry < 2 → lges_strategy 재시도."""
        state = make_state(
            current_task="T2",
            market_research=make_agent_output("T1", content=LONG_CONTENT, quantitative_check=True),
            lges_strategy=make_agent_output("T2", quantitative_check=False),
            catl_strategy=make_agent_output("T3", content=LONG_CONTENT, quantitative_check=True),
            retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 0, "T5": 0, "T6": 0},
        )
        assert supervisor_router(state) == "lges_strategy"

    def test_t3_fail_retry_under_max_retries_to_catl_strategy(self):
        """T3 미통과 + retry < 2 → catl_strategy 재시도."""
        state = make_state(
            current_task="T3",
            market_research=make_agent_output("T1", content=LONG_CONTENT, quantitative_check=True),
            lges_strategy=make_agent_output("T2", content=LONG_CONTENT, quantitative_check=True),
            catl_strategy=make_agent_output("T3", quantitative_check=False),
            retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 0, "T5": 0, "T6": 0},
        )
        assert supervisor_router(state) == "catl_strategy"

    def test_t4_fail_retry_under_max_retries_to_critic(self):
        """T4 미통과 + retry < 2 → critic 재시도."""
        state = make_state(
            current_task="T4",
            critic_result=make_agent_output("T4", quantitative_check=False),
            retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 0, "T6": 0},
        )
        assert supervisor_router(state) == "critic"

    def test_t4_fail_at_max_retries_proceeds_to_swot(self):
        """T4 미통과 + retry = 2 → Fallback 후 swot 진행."""
        state = make_state(
            current_task="T4",
            critic_result=make_agent_output("T4", quantitative_check=False),
            retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 2, "T5": 0, "T6": 0},
        )
        assert supervisor_router(state) == "swot"

    def test_t6_fail_retry_under_max_retries_to_report(self):
        """T6 미통과 + retry < 2 → report 재시도."""
        state = make_state(
            current_task="T6",
            final_report=make_agent_output("T6", quantitative_check=False),
            retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1, "T6": 1},
        )
        assert supervisor_router(state) == "report"

    def test_t6_fail_at_max_retries_ends(self):
        """T6 미통과 + retry = 2 → termination 노드로 fallback 종료."""
        state = make_state(
            current_task="T6",
            final_report=make_agent_output("T6", quantitative_check=False),
            retry_count={"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1, "T6": 2},
        )
        assert supervisor_router(state) == TERMINAL

    def test_t2_done_but_t3_empty_routes_to_catl(self):
        """T2 통과 + T3 미완료 (content='') → catl_strategy 대기."""
        state = make_state(
            current_task="T2",
            market_research=make_agent_output("T1", content=LONG_CONTENT, quantitative_check=True),
            lges_strategy=make_agent_output("T2", content=LONG_CONTENT, quantitative_check=True),
            catl_strategy=make_agent_output("T3", content=""),  # 아직 미완료
            retry_count={"T1": 1, "T2": 1, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        )
        assert supervisor_router(state) == "catl_strategy"


# ──────────────────────────────────────────────
# 3. supervisor_router — 비용 한도 초과
# ──────────────────────────────────────────────

class TestSupervisorRouterCostLimit:

    def test_cost_limit_routes_to_termination(self, base_state):
        """total_llm_calls >= 20 → termination 노드로 라우팅."""
        base_state["total_llm_calls"] = 20
        assert supervisor_router(base_state) == TERMINAL

    def test_cost_limit_overrides_normal_routing(self, t1_pass_state):
        """T1 통과 상태라도 비용 한도 초과 시 termination으로."""
        t1_pass_state["total_llm_calls"] = 20
        assert supervisor_router(t1_pass_state) == TERMINAL

    def test_cost_limit_boundary_19_still_routes(self, t1_pass_state):
        """total_llm_calls = 19 → 아직 정상 라우팅 (termination 아님)."""
        t1_pass_state["total_llm_calls"] = 19
        result = supervisor_router(t1_pass_state)
        assert result != TERMINAL


# ──────────────────────────────────────────────
# 4. supervisor_node — LLM 품질 검토 (P0 핵심)
# ──────────────────────────────────────────────

class TestSupervisorNodeLLMCheck:

    def test_init_task_skips_llm_check(self, base_state):
        """INIT 상태에서는 LLM 품질 검토 없이 빈 dict 반환."""
        with patch("graph.graph.ChatOpenAI") as mock_llm:
            result = supervisor_node(base_state)
            mock_llm.assert_not_called()
        assert result == {}

    def test_cost_limit_skips_llm_check(self, t1_pass_state):
        """비용 한도 초과 시 LLM 품질 검토 생략."""
        t1_pass_state["total_llm_calls"] = 20
        with patch("graph.graph.ChatOpenAI") as mock_llm:
            supervisor_node(t1_pass_state)
            mock_llm.assert_not_called()

    def test_agent_fail_quantitative_skips_llm_check(self, t1_fail_state):
        """Agent 자체 계량 체크 실패 시 LLM 검토 생략 (비용 절약)."""
        with patch("graph.graph.ChatOpenAI") as mock_llm:
            supervisor_node(t1_fail_state)
            mock_llm.assert_not_called()

    def test_t1_pass_calls_llm_quality_check(self, t1_pass_state):
        """T1 quantitative_check=True → LLM 품질 검토 호출."""
        with patch("graph.graph._llm_quality_check", return_value=(True, "PASS")) as mock_check:
            supervisor_node(t1_pass_state)
            mock_check.assert_called_once_with("T1", t1_pass_state["market_research"]["content"])

    def test_llm_pass_returns_llm_call_count(self, t1_pass_state):
        """LLM 품질 검토 통과 → total_llm_calls: 1 반환."""
        with patch("graph.graph._llm_quality_check", return_value=(True, "PASS")):
            result = supervisor_node(t1_pass_state)
        assert result.get("total_llm_calls") == 1

    def test_llm_fail_sets_quantitative_check_false(self, t1_pass_state):
        """LLM 품질 검토 실패 → quantitative_check=False로 재설정."""
        with patch("graph.graph._llm_quality_check", return_value=(False, "FAIL: 출처 없음")):
            result = supervisor_node(t1_pass_state)
        assert result["market_research"]["quantitative_check"] is False

    def test_llm_fail_appends_failed_criteria(self, t1_pass_state):
        """LLM 품질 검토 실패 → failed_criteria에 사유 추가."""
        with patch("graph.graph._llm_quality_check", return_value=(False, "FAIL: 출처 없음")):
            result = supervisor_node(t1_pass_state)
        assert any("T1" in c for c in result.get("failed_criteria", []))

    def test_llm_fail_appends_fallback_item_in_output(self, t1_pass_state):
        """LLM 품질 검토 실패 → AgentOutput.fallback_items에 사유 추가."""
        with patch("graph.graph._llm_quality_check", return_value=(False, "FAIL: 내용 부족")):
            result = supervisor_node(t1_pass_state)
        fallback = result["market_research"].get("fallback_items", [])
        assert any("Supervisor" in item for item in fallback)

    @pytest.mark.parametrize("task_id,state_key", [
        ("T2", "lges_strategy"),
        ("T3", "catl_strategy"),
        ("T4", "critic_result"),
        ("T5", "swot_comparison"),
        ("T6", "final_report"),
    ])
    def test_each_task_calls_correct_state_key(self, task_id, state_key):
        """각 Task ID별로 올바른 state 키의 content를 읽어 LLM 검토."""
        state = make_state(
            current_task=task_id,
            **{state_key: make_agent_output(task_id, content=LONG_CONTENT, quantitative_check=True)},
        )
        with patch("graph.graph._llm_quality_check", return_value=(True, "PASS")) as mock_check:
            supervisor_node(state)
        mock_check.assert_called_once_with(task_id, LONG_CONTENT)


# ──────────────────────────────────────────────
# 5. termination_reason / is_completed (P0 버그 수정)
# ──────────────────────────────────────────────

class TestTerminationReason:

    def test_t6_pass_termination_reason_success(self, t6_pass_state):
        """
        T6 통과 시 termination_node가 termination_reason='success', is_completed=True를 반환.
        P0 설계: supervisor_router → "termination" → termination_node → END
        """
        result = termination_node(t6_pass_state)
        assert result["termination_reason"] == "success", \
            f"T6 통과 시 termination_reason='success' 기대, 실제: {result.get('termination_reason')}"
        assert result["is_completed"] is True

    def test_cost_limit_termination_reason(self):
        """비용 한도 초과 → termination_node가 termination_reason='cost_limit' 설정."""
        state = make_state(
            current_task="T3",
            total_llm_calls=20,
        )
        result = termination_node(state)
        assert result["termination_reason"] == "cost_limit"
        assert result["is_completed"] is False

    def test_fallback_termination_reason(self):
        """T6 미통과 상태 → termination_node가 termination_reason='fallback' 설정."""
        state = make_state(
            current_task="T6",
            final_report=make_agent_output("T6", quantitative_check=False),
            total_llm_calls=6,
        )
        result = termination_node(state)
        assert result["termination_reason"] == "fallback"
        assert result["is_completed"] is False
