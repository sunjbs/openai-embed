import json
import re
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    각 페이지의 텍스트를 리스트로 반환합니다.
    스캔 PDF(이미지)인 경우 빈 문자열이 나올 수 있습니다.
    """
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        texts.append(text)
    return texts


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    간단한 문자 길이 기준 청킹. 필요 시 토큰 단위로 바꿀 수 있습니다.
    """
    chunks = []
    n = len(text)
    if n == 0:
        return chunks
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def pdf_to_chunks(pdf_path: str, chunk_size: int = 1200, overlap: int = 200):
    """
    PDF -> (ids, texts, metadatas)
    """
    pages = extract_text_from_pdf(pdf_path)
    ids, texts, metadatas = [], [], []
    stem = Path(pdf_path).stem
    for pi, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for ci, chunk in enumerate(chunks, start=1):
            _id = f"{stem}-p{pi}-c{ci}"
            ids.append(_id)
            texts.append(chunk)
            metadatas.append(
                {
                    "id": _id,
                    "source": str(pdf_path),
                    "page": pi,
                    "chunk": ci,
                    "chunk_size": len(chunk),
                }
            )
    return ids, texts, metadatas


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> np.ndarray:
    """
    OpenAI 임베딩을 배치로 생성하고 numpy float32 배열로 반환합니다.
    """
    embs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([d.embedding for d in resp.data])
    arr = np.array(embs, dtype=np.float32)
    return arr


def build_faiss_index(
    ids: List[str],
    texts: List[str],
    embs: np.ndarray,
    metadatas: List[Dict],
    out_dir: str,
) -> Path:
    """
    코사인 유사도로 검색하기 위해 L2 정규화 후 Inner Product 인덱스를 만듭니다.
    인덱스(.faiss), ids(.json), 메타(.jsonl)를 저장합니다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))

    with open(out / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for _id, text, meta in zip(ids, texts, metadatas):
            rec = dict(meta)
            rec["text"] = text
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return index_path


def load_faiss_index(out_dir: str):
    """
    저장된 인덱스와 메타 데이터를 로드합니다.
    """
    out = Path(out_dir)
    index = faiss.read_index(str(out / "index.faiss"))

    with open(out / "ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)

    id_to_meta: Dict[str, Dict] = {}
    with open(out / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_meta[rec["id"]] = rec

    return index, ids, id_to_meta


def search(
    out_dir: str, query: str, top_k: int = 5, model: str = "text-embedding-3-small"
):
    """
    쿼리 임베딩 -> FAISS 검색 -> 결과 반환
    """
    client = OpenAI()
    q_emb = embed_texts(client, [query], model=model, batch_size=1)
    faiss.normalize_L2(q_emb)

    index, ids, id_to_meta = load_faiss_index(out_dir)
    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _id = ids[idx]
        meta = id_to_meta.get(_id, {})
        results.append(
            {
                "id": _id,
                "score": float(score),
                "page": meta.get("page"),
                "chunk": meta.get("chunk"),
                "text": meta.get("text", "")[:500],  # 미리보기
            }
        )
    return results


def main():
    import argparse

    # .env 파일에서 환경 변수를 로드합니다.
    # 이 코드는 os.getenv를 호출하기 전에 실행되어야 합니다.
    load_dotenv()
    return

    parser = argparse.ArgumentParser(
        description="Embed a PDF via OpenAI and store locally with FAISS."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Create FAISS index from a PDF")
    b.add_argument("--pdf", required=True, help="Path to the PDF file")
    b.add_argument("--out", required=True, help="Directory to store the index")
    b.add_argument("--chunk-size", type=int, default=1200)
    b.add_argument("--overlap", type=int, default=200)
    b.add_argument("--model", default="text-embedding-3-small")

    s = sub.add_parser("search", help="Search the FAISS index with a query")
    s.add_argument("--out", required=True, help="Directory where index is stored")
    s.add_argument("--query", required=True, help="Search query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--model", default="text-embedding-3-small")

    args = parser.parse_args()

    if args.cmd == "build-index":
        client = OpenAI()
        ids, texts, metas = pdf_to_chunks(
            args.pdf, chunk_size=args.chunk_size, overlap=args.overlap
        )
        if not texts:
            raise SystemExit(
                "PDF에서 텍스트를 추출하지 못했습니다. 스캔 PDF인 경우 OCR이 필요할 수 있습니다."
            )
        embs = embed_texts(client, texts, model=args.model)
        path = build_faiss_index(ids, texts, embs, metas, args.out)
        print(f"FAISS index saved to: {path}")
        print(
            f"Meta and ids saved to: {Path(args.out) / 'meta.jsonl'}, {Path(args.out) / 'ids.json'}"
        )

    elif args.cmd == "search":
        results = search(args.out, args.query, top_k=args.k, model=args.model)
        for r in results:
            preview = r["text"].replace("\n", " ")
            print(
                f"[score={r['score']:.3f}] id={r['id']} page={r.get('page')} chunk={r.get('chunk')} :: {preview[:200]}..."
            )


if __name__ == "__main__":
    main()
