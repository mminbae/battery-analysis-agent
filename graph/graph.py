"""
배터리 시장 전략 분석 Multi-Agent 시스템 — LangGraph 그래프 구성.

구조:
  START → supervisor → T1 → supervisor → [T2 || T3] → supervisor
        → T4 → supervisor → T5 → supervisor → T6 → supervisor
        → termination → END

Supervisor는 모든 Agent 출력을 수신하고
계량 조건·품질 기준에 따라 다음 Task를 지시하거나 재실행한다.

[P0 수정]
- supervisor_node: LLM 기반 품질 검토 실제 구현 (각 Task 완료 시 호출)
- termination_node: termination_reason / is_completed 버그 수정
  supervisor_router에서 END 직전 상태를 업데이트할 수 없는 LangGraph 제약을
  별도 termination 노드로 해결
"""
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from graph.state import WorkflowState, default_agent_output
from agents.market_research import market_research_node
from agents.lges_strategy import lges_strategy_node
from agents.catl_strategy import catl_strategy_node
from agents.critic import critic_node
from agents.swot import swot_node
from agents.report import report_node
from prompts.prompts import SUPERVISOR_QUALITY_CHECK, SUPERVISOR_CRITERIA

MAX_LLM_CALLS = 20
MAX_RETRY = 2
SUPERVISOR_MODEL = "gpt-4o-mini"

# Task ID → State 키 매핑
TASK_TO_KEY = {
    "T1": "market_research",
    "T2": "lges_strategy",
    "T3": "catl_strategy",
    "T4": "critic_result",
    "T5": "swot_comparison",
    "T6": "final_report",
}


# ────────────────────────────────────────────────
# Supervisor LLM 품질 검토 헬퍼
# ────────────────────────────────────────────────
def _llm_quality_check(task_id: str, content: str) -> tuple[bool, str]:
    """
    LLM을 호출하여 Agent 출력의 품질을 검토한다.
    Returns: (passed: bool, reason: str)
    """
    criteria = SUPERVISOR_CRITERIA.get(task_id, "내용이 완전하고 출처가 병기됐는가?")
    llm = ChatOpenAI(model=SUPERVISOR_MODEL, temperature=0)
    prompt = SUPERVISOR_QUALITY_CHECK.format(
        task_id=task_id,
        content=content[:3000],  # 토큰 절약: 앞 3000자만 검토
        criteria=criteria,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()
    passed = result.upper().startswith("PASS")
    return passed, result


# ────────────────────────────────────────────────
# Supervisor Node
# ────────────────────────────────────────────────
def supervisor_node(state: WorkflowState) -> dict:
    """
    Supervisor는 State를 읽어 흐름을 제어한다.
    실제 라우팅은 supervisor_router(조건부 엣지)가 담당.

    [P0] 각 Task 완료 시 LLM 기반 품질 검토를 수행하고,
    미통과 시 quantitative_check를 False로 재설정하여 재시도를 유도한다.
    Agent 자체 계량 체크가 이미 실패한 경우 LLM 검토를 생략해 비용을 절약한다.
    """
    task = state.get("current_task", "INIT")
    total_calls = state.get("total_llm_calls", 0)

    print(f"[Supervisor] current_task={task}, total_llm_calls={total_calls}")

    # 초기 진입 또는 비용 한도 초과 시 품질 검토 생략
    if task == "INIT" or total_calls >= MAX_LLM_CALLS:
        return {}

    state_key = TASK_TO_KEY.get(task)
    if not state_key:
        return {}

    agent_output = state.get(state_key, {})
    content = agent_output.get("content", "")

    # Agent 자체 계량 체크가 이미 실패했으면 LLM 검토 생략 (비용 절약)
    if not agent_output.get("quantitative_check", False):
        print(f"[Supervisor] {task} 계량 조건 미통과 → LLM 검토 생략")
        return {}

    # LLM 품질 검토 수행
    print(f"[Supervisor] {task} LLM 품질 검토 중...")
    passed, reason = _llm_quality_check(task, content)
    print(f"[Supervisor] {task} 품질 검토 결과: {reason}")

    if not passed:
        # quantitative_check를 False로 재설정 → supervisor_router가 재시도 유도
        updated_output = dict(agent_output)
        updated_output["quantitative_check"] = False
        updated_output["fallback_items"] = agent_output.get("fallback_items", []) + [
            f"Supervisor LLM 품질 검토 미통과: {reason}"
        ]
        print(f"[Supervisor] {task} 품질 미통과 → 재시도 유도")
        return {
            state_key: updated_output,
            "failed_criteria": state.get("failed_criteria", []) + [f"{task}: {reason}"],
            "total_llm_calls": 1,  # Annotated[int, operator.add]로 합산됨
        }

    print(f"[Supervisor] {task} 품질 통과 ✓")
    return {
        "total_llm_calls": 1,  # Annotated[int, operator.add]로 합산됨
    }


# ────────────────────────────────────────────────
# Supervisor Router (조건부 엣지 함수)
# ────────────────────────────────────────────────
def supervisor_router(state: WorkflowState) -> Union[str, list]:
    """
    PDF 설계 문서의 supervisor_router 로직을 그대로 구현.
    반환값:
      - str: 단일 노드로 라우팅
      - list[Send]: 병렬 실행 (T2/T3)
      - "termination": 종료 전 상태 업데이트 노드
    """
    task = state.get("current_task", "INIT")
    retry = state.get("retry_count", {})
    total_calls = state.get("total_llm_calls", 0)

    # ── 최우선: 비용 초과 종료 ──
    if total_calls >= MAX_LLM_CALLS:
        print(f"[Supervisor] cost_limit 도달 (총 {total_calls}회)")
        return "termination"

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
        print("[Supervisor] T6 완료 → termination")
        return "termination"

    return "termination"


# ────────────────────────────────────────────────
# Termination Node  [P0 버그 수정]
# ────────────────────────────────────────────────
def termination_node(state: WorkflowState) -> dict:
    """
    [P0] termination_reason / is_completed 버그 수정.

    supervisor_router는 조건부 엣지 함수라 State를 직접 수정할 수 없다.
    따라서 END 직전에 별도 노드를 두어 종료 상태를 설정한다.
    """
    total_calls = state.get("total_llm_calls", 0)

    if total_calls >= MAX_LLM_CALLS:
        reason = "cost_limit"
        completed = False
    elif state.get("final_report", {}).get("quantitative_check", False):
        reason = "success"
        completed = True
    else:
        reason = "fallback"
        completed = False

    print(f"[Termination] 종료 사유: {reason}, 완료: {completed}")
    return {
        "termination_reason": reason,
        "is_completed": completed,
    }


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
    builder.add_node("termination", termination_node)  # [P0]

    # START → supervisor
    builder.add_edge(START, "supervisor")

    # 모든 Agent → supervisor (고정 엣지)
    for agent_node in ["market_research", "lges_strategy", "catl_strategy", "critic", "swot", "report"]:
        builder.add_edge(agent_node, "supervisor")

    # termination → END (고정 엣지)  [P0]
    builder.add_edge("termination", END)

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
            "termination": "termination",  # [P0]
        },
    )

    return builder.compile()


# 전역 그래프 인스턴스
graph = build_graph()
