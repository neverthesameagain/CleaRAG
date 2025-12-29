
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import numpy as np

# ---------------- CONFIG ----------------
CHUNK_SIZE = 300
OVERLAP = 50
TOP_K = 4
FINAL_K = 2
MAX_CONTEXT_WORDS = 350
RELEVANCE_THRESHOLD = 0.15  # cosine similarity threshold

# ---------------- CACHE ----------------
query_cache = {}

# ---------------- MODELS ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

# ---------------- FILE READING ----------------
def read_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    else:
        return file.read().decode("utf-8").strip()

# ---------------- CHUNKING ----------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

# ---------------- RERANKING ----------------
def rerank(chunks, embeddings, query_embedding, final_k=2):
    scores = embeddings @ query_embedding
    ranked_idx = np.argsort(scores)[::-1]
    return [chunks[i] for i in ranked_idx[:final_k]]

# ---------------- WEB SEARCH (SAFE FALLBACK) ----------------
def web_search_fallback(query):
    """
    Placeholder for real web search.
    This is intentionally explicit and honest.
    """
    return (
        "WEB SEARCH FALLBACK USED\n\n"
        "No sufficiently relevant context was found in the uploaded document.\n"
        "A real deployment would query a web search API here (e.g., Bing, Tavily).\n\n"
        f"Query: {query}"
    )

# ---------------- CORE RAG ----------------
def rag_pipeline(text, query):
    if query in query_cache:
        return query_cache[query]

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
    scores, indices = index.search(query_embedding.reshape(1, -1), TOP_K)

    top_score = scores[0][0]

    # ---------- NO CONTEXT FOUND ----------
    if top_score < RELEVANCE_THRESHOLD:
        answer = web_search_fallback(query)
        result = {
            "source": "web_search",
            "context": None,
            "used_chunks": None,
            "answer": answer
        }
        query_cache[query] = result
        return result

    # ---------- CONTEXT FOUND ----------
    candidate_chunks = [chunks[i] for i in indices[0]]
    candidate_embeddings = embeddings[indices[0]]

    selected_chunks = rerank(
        candidate_chunks,
        candidate_embeddings,
        query_embedding,
        FINAL_K
    )

    context = "\n\n".join(selected_chunks)
    context = " ".join(context.split()[:MAX_CONTEXT_WORDS])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present in the context, say "Not found in the provided documents."

Context:
{context}

Question:
{query}
"""

    response = llm(prompt, max_new_tokens=150)
    answer = response[0]["generated_text"]

    result = {
        "source": "document",
        "context": context,
        "used_chunks": selected_chunks,
        "answer": answer
    }

    query_cache[query] = result
    return result

# ---------------- UI ----------------
def ui(file, question):
    if file is None or question.strip() == "":
        return "Please upload a document and enter a question."

    extracted_text = read_file(file)

    if not extracted_text:
        return "Could not extract text from the document (possibly scanned PDF)."

    result = rag_pipeline(extracted_text, question)

    output = "===== EXTRACTED TEXT =====\n"
    output += extracted_text[:1500] + "\n\n"

    if result["source"] == "document":
        output += "===== RETRIEVED CONTEXT =====\n"
        output += result["context"] + "\n\n"

        output += "===== CONTEXT CHUNKS USED =====\n"
        for i, chunk in enumerate(result["used_chunks"], 1):
            output += f"[Chunk {i}]\n{chunk}\n\n"

        output += "===== ANSWER (FROM DOCUMENT) =====\n"
        output += result["answer"]

    else:
        output += "===== NO RELEVANT CONTEXT FOUND =====\n"
        output += "Falling back to web search.\n\n"
        output += "===== ANSWER (FROM WEB SEARCH) =====\n"
        output += result["answer"]

    return output

# ---------------- GRADIO APP ----------------
gr.Interface(
    fn=ui,
    inputs=[
        gr.File(label="Upload TXT or PDF"),
        gr.Textbox(lines=2, label="Question")
    ],
    outputs="text",
    title="Mini RAG QA System with Fallback Search",
    description="Shows extracted text, retrieved context, source attribution, and fallback behavior."
).launch()
