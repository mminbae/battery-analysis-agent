"""
배터리 시장 전략 분석 Multi-Agent 시스템 — LangGraph 그래프 구성.

구조:
  START → supervisor → T1 → supervisor → [T2 || T3] → supervisor
        → T4 → supervisor → T5 → supervisor → T6 → supervisor → END

Supervisor는 모든 Agent 출력을 수신하고
계량 조건·품질 기준에 따라 다음 Task를 지시하거나 재실행한다.
"""
from typing import Union
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from graph.state import WorkflowState, default_agent_output
from agents.market_research import market_research_node
from agents.lges_strategy import lges_strategy_node
from agents.catl_strategy import catl_strategy_node
from agents.critic import critic_node
from agents.swot import swot_node
from agents.report import report_node

MAX_LLM_CALLS = 20
MAX_RETRY = 2


# ────────────────────────────────────────────────
# Supervisor Node
# ────────────────────────────────────────────────
def supervisor_node(state: WorkflowState) -> dict:
    """
    Supervisor는 State를 읽어 흐름을 제어한다.
    실제 라우팅은 supervisor_router(조건부 엣지)가 담당.
    이 노드에서는 상태 업데이트만 수행.
    """
    print(f"[Supervisor] current_task={state.get('current_task', 'INIT')}, "
          f"total_llm_calls={state.get('total_llm_calls', 0)}")
    return {}


# ────────────────────────────────────────────────
# Supervisor Router (조건부 엣지 함수)
# ────────────────────────────────────────────────
def supervisor_router(state: WorkflowState) -> Union[str, list]:
    """
    PDF 설계 문서의 supervisor_router 로직을 그대로 구현.
    반환값:
      - str: 단일 노드로 라우팅
      - list[Send]: 병렬 실행 (T2/T3)
      - END: 종료
    """
    task = state.get("current_task", "INIT")
    retry = state.get("retry_count", {})
    total_calls = state.get("total_llm_calls", 0)

    # ── 최우선: 비용 초과 종료 ──
    if total_calls >= MAX_LLM_CALLS:
        print(f"[Supervisor] cost_limit 도달 (총 {total_calls}회)")
        return END

    # ── 초기 진입: T1 시작 ──
    if task == "INIT":
        print("[Supervisor] → T1 (market_research)")
        return "market_research"

    # ── T1 완료 → T2/T3 병렬 또는 재시도 ──
    if task == "T1":
        market = state.get("market_research", {})
        if not market.get("quantitative_check", False):
            if retry.get("T1", 0) < MAX_RETRY:
                print(f"[Supervisor] T1 미통과 → 재시도 (retry={retry.get('T1', 0)})")
                return "market_research"
            else:
                print("[Supervisor] T1 최대 재시도 초과 → Fallback 후 T2/T3 진행")
        print("[Supervisor] T1 완료 → T2/T3 병렬 실행")
        return [
            Send("lges_strategy", state),
            Send("catl_strategy", state),
        ]

    # ── T2 완료 → 품질 체크 ──
    if task == "T2":
        lges = state.get("lges_strategy", {})
        if not lges.get("quantitative_check", False):
            if retry.get("T2", 0) < MAX_RETRY:
                print(f"[Supervisor] T2 미통과 → 재시도 (retry={retry.get('T2', 0)})")
                return "lges_strategy"
        # T3도 완료됐는지 확인 후 T4로
        catl = state.get("catl_strategy", {})
        if catl.get("content", "") == "":
            print("[Supervisor] T3 대기 중...")
            return "catl_strategy"
        print("[Supervisor] T2 완료 → T4")
        return "critic"

    # ── T3 완료 → 품질 체크 ──
    if task == "T3":
        catl = state.get("catl_strategy", {})
        if not catl.get("quantitative_check", False):
            if retry.get("T3", 0) < MAX_RETRY:
                print(f"[Supervisor] T3 미통과 → 재시도 (retry={retry.get('T3', 0)})")
                return "catl_strategy"
        # T2도 완료됐는지 확인 후 T4로
        lges = state.get("lges_strategy", {})
        if lges.get("content", "") == "":
            print("[Supervisor] T2 대기 중...")
            return "lges_strategy"
        print("[Supervisor] T3 완료 → T4")
        return "critic"

    # ── T4 완료 → T5 또는 재시도 ──
    if task == "T4":
        critic = state.get("critic_result", {})
        if not critic.get("quantitative_check", False):
            if retry.get("T4", 0) < MAX_RETRY:
                print(f"[Supervisor] T4 미통과 → 재시도 (retry={retry.get('T4', 0)})")
                return "critic"
            else:
                print("[Supervisor] T4 최대 재시도 초과 → Fallback 후 T5 진행")
        print("[Supervisor] T4 완료 → T5")
        return "swot"

    # ── T5 완료 → T6 또는 재시도 ──
    if task == "T5":
        swot = state.get("swot_comparison", {})
        if not swot.get("quantitative_check", False):
            if retry.get("T5", 0) < MAX_RETRY:
                print(f"[Supervisor] T5 미통과 → 재시도 (retry={retry.get('T5', 0)})")
                return "swot"
        print("[Supervisor] T5 완료 → T6")
        return "report"

    # ── T6 완료 → 최종 검토 후 종료 ──
    if task == "T6":
        report = state.get("final_report", {})
        if not report.get("quantitative_check", False):
            if retry.get("T6", 0) < MAX_RETRY:
                print(f"[Supervisor] T6 미통과 → 재시도 (retry={retry.get('T6', 0)})")
                return "report"
            else:
                print("[Supervisor] T6 최대 재시도 초과 → fallback 종료")
                return END
        print("[Supervisor] T6 완료 → success 종료")
        return END

    return END


# ────────────────────────────────────────────────
# Termination State Update (각 END 직전에 처리)
# ────────────────────────────────────────────────
def _update_termination_reason(state: WorkflowState) -> dict:
    """종료 사유 업데이트 (supervisor_router에서 END 반환 전 호출)."""
    total_calls = state.get("total_llm_calls", 0)
    if total_calls >= MAX_LLM_CALLS:
        return {"termination_reason": "cost_limit", "is_completed": False}

    report = state.get("final_report", {})
    if report.get("quantitative_check", False):
        return {"termination_reason": "success", "is_completed": True}

    return {"termination_reason": "fallback", "is_completed": False}


# ────────────────────────────────────────────────
# 초기 상태 초기화
# ────────────────────────────────────────────────
def get_initial_state(user_query: str) -> WorkflowState:
    """그래프 실행을 위한 초기 WorkflowState 반환."""
    return WorkflowState(
        user_query=user_query,
        market_research=default_agent_output("T1"),
        lges_strategy=default_agent_output("T2"),
        catl_strategy=default_agent_output("T3"),
        critic_result=default_agent_output("T4"),
        swot_comparison=default_agent_output("T5"),
        final_report=default_agent_output("T6"),
        current_task="INIT",
        retry_count={"T1": 0, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        failed_criteria=[],
        fallback_items=[],
        total_llm_calls=0,
        is_completed=False,
        termination_reason="",
    )


# ────────────────────────────────────────────────
# 그래프 구성
# ────────────────────────────────────────────────
def build_graph():
    """배터리 분석 Multi-Agent 그래프 생성."""
    builder = StateGraph(WorkflowState)

    # 노드 등록
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("market_research", market_research_node)
    builder.add_node("lges_strategy", lges_strategy_node)
    builder.add_node("catl_strategy", catl_strategy_node)
    builder.add_node("critic", critic_node)
    builder.add_node("swot", swot_node)
    builder.add_node("report", report_node)

    # START → supervisor
    builder.add_edge(START, "supervisor")

    # 모든 Agent → supervisor (고정 엣지)
    for agent_node in ["market_research", "lges_strategy", "catl_strategy", "critic", "swot", "report"]:
        builder.add_edge(agent_node, "supervisor")

    # supervisor → ? (조건부 엣지)
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "market_research": "market_research",
            "lges_strategy": "lges_strategy",
            "catl_strategy": "catl_strategy",
            "critic": "critic",
            "swot": "swot",
            "report": "report",
            END: END,
        },
    )

    return builder.compile()


# 전역 그래프 인스턴스
graph = build_graph()
