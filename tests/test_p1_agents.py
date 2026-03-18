"""
P1 테스트: Agent 내용 밀도 + Agentic RAG 재검색

검증 항목:
  1. MIN_CONTENT_CHARS — 출력 내용이 짧으면 quantitative_check=False
  2. Agentic RAG — RAG 문서 < 3건 시 대안 쿼리로 재검색 트리거
  3. 확증 편향 방지 — 프롬프트에 부정 쿼리 키워드 포함 여부
  4. 기업 혼입 방지 — CATL/LGES 혼입 비율 초과 시 체크
"""
import pytest
from unittest.mock import patch, MagicMock, call
from langchain_core.documents import Document

from agents.lges_strategy import _run_lges_strategy, MIN_CONTENT_CHARS as LGES_MIN_CHARS
from agents.catl_strategy import _run_catl_strategy, MIN_CONTENT_CHARS as CATL_MIN_CHARS
from agents.critic import _count_negative_evidence
from prompts.prompts import LGES_STRATEGY_SYSTEM, CATL_STRATEGY_SYSTEM


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def make_doc(content: str, source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


def mock_llm_response(text: str):
    """ChatOpenAI.invoke() 응답 mock."""
    mock = MagicMock()
    mock.content = text
    return mock


# ──────────────────────────────────────────────
# 1. MIN_CONTENT_CHARS 체크
# ──────────────────────────────────────────────

class TestMinContentChars:

    def _make_patches(self, llm_response_text: str):
        """_run_lges_strategy / _run_catl_strategy 실행에 필요한 공통 mock 패치."""
        docs = [make_doc("LGES 전략 내용", "lges.pdf")] * 5

        patches = {
            "retriever": patch("agents.lges_strategy.get_retriever"),
            "web_bidir": patch("agents.lges_strategy.web_search_bidirectional",
                               return_value=([], [])),
            "web_extra": patch("agents.lges_strategy.web_search", return_value=[]),
            "llm": patch("agents.lges_strategy.ChatOpenAI"),
        }
        return patches, docs, llm_response_text

    def test_short_content_sets_quantitative_check_false(self):
        """LLM 출력이 MIN_CONTENT_CHARS 미만 → quantitative_check=False."""
        short_text = "짧은 LGES 분석"  # << 1200자

        with patch("agents.lges_strategy.get_retriever") as mock_ret, \
             patch("agents.lges_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.lges_strategy.web_search", return_value=[]), \
             patch("agents.lges_strategy.ChatOpenAI") as mock_llm:

            mock_ret.return_value.search.return_value = [make_doc("내용", "lges.pdf")] * 5
            mock_llm.return_value.invoke.return_value = mock_llm_response(short_text)

            _, _, quantitative_check, fallback_items = _run_lges_strategy(
                "EV 캐즘 LGES 전략 분석", "시장 배경 내용"
            )

        assert quantitative_check is False
        assert any("짧음" in item or "최소" in item for item in fallback_items), \
            "내용 길이 부족 사유가 fallback_items에 포함돼야 함"

    def test_sufficient_content_passes_length_check(self):
        """LLM 출력이 MIN_CONTENT_CHARS 이상 → 길이 조건 통과."""
        long_text = "LGES 전략 분석 내용입니다. " * 100  # ~1400자

        with patch("agents.lges_strategy.get_retriever") as mock_ret, \
             patch("agents.lges_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.lges_strategy.web_search", return_value=[]), \
             patch("agents.lges_strategy.ChatOpenAI") as mock_llm:

            mock_ret.return_value.search.return_value = [make_doc("내용", "lges.pdf")] * 5
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)

            _, _, quantitative_check, fallback_items = _run_lges_strategy(
                "EV 캐즘 LGES 전략 분석", "시장 배경 내용"
            )

        length_failures = [item for item in fallback_items if "짧음" in item or "최소" in item]
        assert len(length_failures) == 0, f"충분한 내용인데 길이 체크 실패: {length_failures}"

    def test_catl_short_content_sets_quantitative_check_false(self):
        """CATL Agent도 동일하게 내용 길이 체크."""
        short_text = "짧은 CATL 분석"

        with patch("agents.catl_strategy.get_retriever") as mock_ret, \
             patch("agents.catl_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.catl_strategy.web_search", return_value=[]), \
             patch("agents.catl_strategy.ChatOpenAI") as mock_llm:

            mock_ret.return_value.search.return_value = [make_doc("내용", "catl.pdf")] * 5
            mock_llm.return_value.invoke.return_value = mock_llm_response(short_text)

            _, _, quantitative_check, fallback_items = _run_catl_strategy(
                "EV 캐즘 CATL 전략 분석", "시장 배경 내용"
            )

        assert quantitative_check is False

    @pytest.mark.parametrize("min_chars,label", [
        (LGES_MIN_CHARS, "LGES"),
        (CATL_MIN_CHARS, "CATL"),
    ])
    def test_min_content_chars_threshold(self, min_chars, label):
        """MIN_CONTENT_CHARS가 설계 최솟값(1200자) 이상인지 확인."""
        assert min_chars >= 1200, f"{label} MIN_CONTENT_CHARS({min_chars})가 너무 낮음 (최소 1200자)"


# ──────────────────────────────────────────────
# 2. Agentic RAG — 재검색 트리거
# ──────────────────────────────────────────────

class TestAgenticRAGRetry:

    def test_lges_rag_retries_when_initial_docs_empty(self):
        """
        LGES RAG 1차 검색 결과 0건 → 대안 쿼리로 재검색 수행.
        retriever.search 호출 횟수가 기본 쿼리 수보다 많아야 함.
        """
        long_text = "LGES 분석 내용 " * 100

        with patch("agents.lges_strategy.get_retriever") as mock_ret, \
             patch("agents.lges_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.lges_strategy.web_search", return_value=[]), \
             patch("agents.lges_strategy.ChatOpenAI") as mock_llm:

            # 1차 검색: 0건 반환 → 재검색 필요
            # 재검색: 문서 3건 반환
            fallback_doc = make_doc("LG Energy Solution battery strategy", "lges_en.pdf")
            mock_ret.return_value.search.side_effect = (
                [[] for _ in range(5)] +          # 1차 5개 쿼리: 모두 빈 결과
                [[fallback_doc] for _ in range(3)] # 재검색 3개 쿼리: 각 1건
            )
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)

            _run_lges_strategy("LGES 전략", "시장 배경")

        # 기본 쿼리(5개) + 재검색 쿼리(3개) = 최소 8회 이상
        total_calls = mock_ret.return_value.search.call_count
        assert total_calls > 5, \
            f"Agentic RAG 재검색이 실행되지 않았음 (search 호출 횟수: {total_calls})"

    def test_catl_rag_retries_when_initial_docs_empty(self):
        """CATL RAG 1차 검색 결과 0건 → 재검색 트리거."""
        long_text = "CATL 분석 내용 " * 100

        with patch("agents.catl_strategy.get_retriever") as mock_ret, \
             patch("agents.catl_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.catl_strategy.web_search", return_value=[]), \
             patch("agents.catl_strategy.ChatOpenAI") as mock_llm:

            fallback_doc = make_doc("CATL battery strategy annual report", "catl_en.pdf")
            mock_ret.return_value.search.side_effect = (
                [[] for _ in range(5)] +
                [[fallback_doc] for _ in range(3)]
            )
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)

            _run_catl_strategy("CATL 전략", "시장 배경")

        total_calls = mock_ret.return_value.search.call_count
        assert total_calls > 5, \
            f"CATL Agentic RAG 재검색이 실행되지 않았음 (search 호출 횟수: {total_calls})"

    def test_lges_rag_no_retry_when_docs_sufficient(self):
        """LGES RAG 1차 검색 결과 충분(≥3건) → 재검색 미실행."""
        long_text = "LGES 분석 내용 " * 100
        sufficient_docs = [make_doc(f"LGES 전략 내용 {i}", "lges.pdf") for i in range(5)]

        with patch("agents.lges_strategy.get_retriever") as mock_ret, \
             patch("agents.lges_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.lges_strategy.web_search", return_value=[]), \
             patch("agents.lges_strategy.ChatOpenAI") as mock_llm:

            # 1차 검색부터 충분한 문서 반환
            mock_ret.return_value.search.return_value = sufficient_docs
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)

            _run_lges_strategy("LGES 전략", "시장 배경")

        # 5개 기본 쿼리만 실행 (재검색 없음)
        total_calls = mock_ret.return_value.search.call_count
        assert total_calls == 5, \
            f"문서가 충분한데 재검색이 실행됨 (search 호출 횟수: {total_calls})"


# ──────────────────────────────────────────────
# 3. 확증 편향 방지 — 프롬프트 키워드 검증
# ──────────────────────────────────────────────

class TestBiasPreventionPrompt:

    def test_lges_system_prompt_has_negative_query_keywords(self):
        """LGES 시스템 프롬프트에 부정 쿼리 관련 키워드 포함.
        P1에서 "위협" 키워드가 추가되면 negative_keywords에 포함할 것.
        """
        negative_keywords = ["리스크", "약점", "부진"]  # "위협"은 P1 프롬프트 강화 후 추가
        missing = [kw for kw in negative_keywords if kw not in LGES_STRATEGY_SYSTEM]
        assert not missing, f"LGES 프롬프트에 부정 쿼리 키워드 누락: {missing}"

    def test_catl_system_prompt_has_negative_query_keywords(self):
        """CATL 시스템 프롬프트에 부정 쿼리 관련 키워드 포함."""
        negative_keywords = ["리스크", "약점", "관세"]  # "위협"은 P1 프롬프트 강화 후 추가
        missing = [kw for kw in negative_keywords if kw not in CATL_STRATEGY_SYSTEM]
        assert not missing, f"CATL 프롬프트에 부정 쿼리 키워드 누락: {missing}"

    def test_lges_calls_bidirectional_web_search(self):
        """LGES Agent가 web_search_bidirectional을 호출하는지 확인."""
        long_text = "LGES 분석 " * 100

        with patch("agents.lges_strategy.get_retriever") as mock_ret, \
             patch("agents.lges_strategy.web_search_bidirectional",
                   return_value=([], [])) as mock_bidir, \
             patch("agents.lges_strategy.web_search", return_value=[]), \
             patch("agents.lges_strategy.ChatOpenAI") as mock_llm:

            mock_ret.return_value.search.return_value = [make_doc("내용")] * 5
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)
            _run_lges_strategy("LGES 전략", "시장 배경")

        mock_bidir.assert_called_once()

    def test_catl_calls_bidirectional_web_search(self):
        """CATL Agent가 web_search_bidirectional을 호출하는지 확인."""
        long_text = "CATL 분석 " * 100

        with patch("agents.catl_strategy.get_retriever") as mock_ret, \
             patch("agents.catl_strategy.web_search_bidirectional",
                   return_value=([], [])) as mock_bidir, \
             patch("agents.catl_strategy.web_search", return_value=[]), \
             patch("agents.catl_strategy.ChatOpenAI") as mock_llm:

            mock_ret.return_value.search.return_value = [make_doc("내용")] * 5
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)
            _run_catl_strategy("CATL 전략", "시장 배경")

        mock_bidir.assert_called_once()


# ──────────────────────────────────────────────
# 4. 기업 혼입 방지
# ──────────────────────────────────────────────

class TestCrossContaminationCheck:

    def test_lges_catl_mixture_fails_check(self):
        """LGES 문서에 CATL 내용이 30% 초과 → quantitative_check=False."""
        long_text = "LGES 분석 내용 " * 100
        # CATL 키워드를 포함한 문서를 다수 반환
        catl_docs = [make_doc("CATL 배터리 전략 내용 포함", "catl.pdf")] * 4
        lges_docs = [make_doc("LGES 전략 내용", "lges.pdf")] * 1  # 비율: 4/5 = 80% 혼입

        with patch("agents.lges_strategy.get_retriever") as mock_ret, \
             patch("agents.lges_strategy.web_search_bidirectional", return_value=([], [])), \
             patch("agents.lges_strategy.web_search", return_value=[]), \
             patch("agents.lges_strategy.ChatOpenAI") as mock_llm:

            mock_ret.return_value.search.return_value = catl_docs + lges_docs
            mock_llm.return_value.invoke.return_value = mock_llm_response(long_text)

            _, _, quantitative_check, fallback_items = _run_lges_strategy(
                "LGES 전략", "시장 배경"
            )

        assert quantitative_check is False
        assert any("혼입" in item for item in fallback_items)
