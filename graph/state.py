import operator
from typing import Annotated, TypedDict


def _last_value(a, b):
    """병렬 쓰기 시 마지막 값을 취하는 reducer."""
    return b


def _merge_dicts(a: dict, b: dict) -> dict:
    """병렬 쓰기 시 딕셔너리를 병합하는 reducer (각 Agent가 자신의 키만 업데이트)."""
    return {**a, **b}


class AgentOutput(TypedDict):
    """각 Agent의 개별 출력 단위."""
    task_id: str                 # "T1" ~ "T6"
    content: str                 # 분석 결과 본문
    sources: list[str]           # 출처 목록 (RAG 문서명 or URL)
    quantitative_check: bool     # 계량 조건 통과 여부 (Agent 자체 판단)
    fallback_items: list[str]    # 근거 불충분으로 Fallback 처리된 항목


def default_agent_output(task_id: str) -> AgentOutput:
    """초기화용 기본 AgentOutput 반환."""
    return AgentOutput(
        task_id=task_id,
        content="",
        sources=[],
        quantitative_check=False,
        fallback_items=[],
    )


class WorkflowState(TypedDict):
    """Supervisor가 관리하는 전체 워크플로우 상태."""
    # 입력
    user_query: str

    # 각 Task 출력
    market_research: AgentOutput   # T1
    lges_strategy: AgentOutput     # T2
    catl_strategy: AgentOutput     # T3
    critic_result: AgentOutput     # T4
    swot_comparison: AgentOutput   # T5
    final_report: AgentOutput      # T6

    # 흐름 제어
    # Annotated: 병렬 실행(T2/T3) 시 두 노드가 동시에 쓰는 필드에 reducer 필요
    current_task: Annotated[str, _last_value]              # 현재 실행 중인 Task ID
    retry_count: Annotated[dict[str, int], _merge_dicts]   # {"T1":0, "T2":0, ..., "T6":0}
    failed_criteria: list[str]                             # 미통과 Criteria 항목명
    fallback_items: list[str]                              # 전체 Fallback 처리 항목 누적

    # 종료 조건
    total_llm_calls: Annotated[int, operator.add]  # 누적 LLM 호출 수 (병렬 실행 시 합산)
    is_completed: bool             # 정상 완료 여부
    termination_reason: str        # "success" / "fallback" / "cost_limit"
