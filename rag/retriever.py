"""
MultiPDFRetriever: bge-m3 임베딩 + FAISS 벡터스토어 기반 RAG 검색기.

11-RAG/rag/base.py, pdf.py 패턴을 기반으로 bge-m3 임베딩으로 교체.
PDF 문서 유형별로 별도 retriever 인스턴스를 생성한다.
"""

from pathlib import Path
from typing import Optional
import hashlib

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# 문서 카테고리별 PDF 파일명 매핑
DOCUMENT_CATEGORIES = {
    "market": [
        "IEA_GlobalEVOutlook2025_summary.pdf",
        "배터리 산업 - ESS 배터리의 빛과 그림자.pdf",
    ],
    "lges": [
        "[LG에너지솔루션]사업보고서_사업개요.pdf",
    ],
    "catl": [
        "CATL 사업보고서_summary.pdf",
    ],
    "critic": [
        "IEA_GlobalEVOutlook2025_summary.pdf",
        "[LG에너지솔루션]사업보고서_사업개요.pdf",
        "CATL 사업보고서_summary.pdf",
        "배터리 산업 - ESS 배터리의 빛과 그림자.pdf",
    ],
}


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """BAAI/bge-m3 임베딩 모델 반환 (싱글톤)."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_retriever(pdf_paths: list[str], cache_key: str, k: int = 8):
    """
    주어진 PDF 파일 목록으로 FAISS retriever를 생성한다.
    캐시가 있으면 재사용, 없으면 새로 생성.
    """
    cache_dir = Path(f".cache/faiss_index/{cache_key}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(cache_dir / "faiss_index")
    hash_file = cache_dir / "doc_hash.txt"

    # 존재하는 PDF만 필터링
    valid_paths = [p for p in pdf_paths if Path(p).exists()]
    missing = [p for p in pdf_paths if not Path(p).exists()]
    if missing:
        print(f"[RAG] 누락된 PDF 파일: {missing}")

    if not valid_paths:
        print(
            f"[RAG] '{cache_key}' 카테고리에 사용 가능한 PDF 없음. Dummy retriever 반환."
        )
        return None

    # 문서 로딩 + 분할
    all_docs = []
    for path in valid_paths:
        print(f"[RAG] Loading: {path}")
        loader = PDFPlumberLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(all_docs)

    # 문서 해시 계산
    content_hash = hashlib.md5(
        "\n".join([d.page_content for d in split_docs]).encode()
    ).hexdigest()

    embeddings = _get_embedding_model()

    # 캐시 확인
    if (
        hash_file.exists()
        and Path(index_path + ".faiss").exists()
        and hash_file.read_text().strip() == content_hash
    ):
        try:
            vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            print(f"[RAG] Loaded FAISS cache: {cache_key}")
            return vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            )
        except Exception as e:
            print(f"[RAG] 캐시 로드 실패 ({e}), 재생성...")

    # 새 인덱스 생성
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    try:
        vectorstore.save_local(index_path)
        hash_file.write_text(content_hash)
        print(f"[RAG] FAISS index saved: {cache_key}")
    except Exception as e:
        print(f"[RAG] 인덱스 저장 실패: {e}")

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


class BatteryRAGRetriever:
    """
    배터리 분석 Agent용 카테고리별 RAG 검색기.

    사용법:
        retriever = BatteryRAGRetriever(data_dir="data/")
        retriever.initialize()
        docs = retriever.search("market", "EV 캐즘 현황")
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self._retrievers: dict[str, Optional[object]] = {}

    def initialize(self):
        """모든 카테고리 retriever 초기화. 앱 시작 시 한 번만 호출."""
        print("[RAG] Initializing BatteryRAGRetriever...")
        for category, filenames in DOCUMENT_CATEGORIES.items():
            pdf_paths = [str(self.data_dir / fn) for fn in filenames]
            self._retrievers[category] = _build_retriever(pdf_paths, cache_key=category)
        print("[RAG] Initialization complete.")

    def search(self, category: str, query: str) -> list:
        """
        카테고리에 해당하는 retriever로 검색.
        retriever가 없으면 빈 리스트 반환.
        """
        retriever = self._retrievers.get(category)
        if retriever is None:
            print(f"[RAG] '{category}' retriever 없음 (PDF 미배치). 빈 결과 반환.")
            return []
        docs = retriever.invoke(query)
        return docs

    def is_available(self, category: str) -> bool:
        return self._retrievers.get(category) is not None


# 전역 싱글톤 (app.py에서 초기화 후 사용)
_global_retriever: Optional[BatteryRAGRetriever] = None


def get_retriever() -> BatteryRAGRetriever:
    global _global_retriever
    if _global_retriever is None:
        raise RuntimeError(
            "BatteryRAGRetriever가 초기화되지 않았습니다. app.py에서 init_retriever()를 먼저 호출하세요."
        )
    return _global_retriever


def init_retriever(data_dir: str = "data/") -> BatteryRAGRetriever:
    """앱 시작 시 retriever 초기화 및 전역 등록."""
    global _global_retriever
    _global_retriever = BatteryRAGRetriever(data_dir=data_dir)
    _global_retriever.initialize()
    return _global_retriever
