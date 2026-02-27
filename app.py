import streamlit as st
import hashlib
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 780px; }

.hero {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #e8ecf4;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.hero .subtitle {
    font-size: 0.82rem;
    color: #6b7694;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero .badge {
    display: inline-block;
    background: #1a2540;
    border: 1px solid #2a3a60;
    color: #7a9fd4;
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: 0.6rem;
    font-weight: 500;
}

[data-testid="stChatMessageContent"] {
    background: #161b2e !important;
    border: 1px solid #1e2744 !important;
    border-radius: 12px !important;
    color: #cdd5e8 !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #1a2d4a !important;
    border-color: #2a4470 !important;
    color: #dde4f5 !important;
}

.source-box {
    background: #0d1420;
    border: 1px solid #1a2744;
    border-left: 3px solid #3a6bc4;
    border-radius: 8px;
    padding: 0.5rem 0.85rem;
    margin-top: 0.6rem;
    font-size: 0.75rem;
    color: #5a7ab4;
    font-style: italic;
}

.status-bar {
    text-align: center;
    font-size: 0.72rem;
    color: #3a4a6a;
    margin-bottom: 0.75rem;
    letter-spacing: 0.03em;
}
.status-dot  { display: inline-block; width: 6px; height: 6px; background: #2d8a4e; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
.status-warn { display: inline-block; width: 6px; height: 6px; background: #c8922a; border-radius: 50%; margin-right: 5px; vertical-align: middle; }

.stChatInputContainer {
    background: #161b2e !important;
    border: 1px solid #1e2744 !important;
    border-radius: 12px !important;
}
.stChatInputContainer textarea {
    color: #cdd5e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
.stSpinner > div { border-top-color: #3a6bc4 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

CONTENT_ROOT = Path(__file__).parent / "content"
SUPPORTED    = {".txt", ".html", ".htm", ".pdf", ".docx"}


def extract_text(filepath: Path) -> str | None:
    """Extract plain text from a supported file type."""
    ext = filepath.suffix.lower()
    try:
        if ext == ".txt":
            return filepath.read_text(encoding="utf-8", errors="ignore")

        elif ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            return BeautifulSoup(filepath.read_bytes(), "html.parser").get_text(separator="\n")

        elif ext == ".pdf":
            import io
            import pypdf
            reader = pypdf.PdfReader(str(filepath))
            return "\n".join(p.extract_text() or "" for p in reader.pages)

        elif ext == ".docx":
            import docx
            doc = docx.Document(str(filepath))
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception as e:
        st.warning(f"Could not read {filepath.name}: {e}")
    return None


def load_course_content(course_id: str) -> tuple[list[dict], str]:
    """
    Load all content files from content/<course_id>/.
    Returns (list of {title, text} dicts, content hash).
    """
    course_dir = CONTENT_ROOT / course_id
    if not course_dir.exists():
        raise FileNotFoundError(
            f"No content folder found for course '{course_id}'. "
            f"Create the folder content/{course_id}/ and upload your course files."
        )

    docs = []
    for filepath in sorted(course_dir.iterdir()):
        if filepath.suffix.lower() not in SUPPORTED:
            continue
        text = extract_text(filepath)
        if text and text.strip():
            docs.append({"title": filepath.stem, "text": text.strip()})

    content_hash = hashlib.md5(
        "".join(d["text"] for d in docs).encode()
    ).hexdigest()

    return docs, content_hash


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_pipeline(course_id: str, content_hash: str):
    """
    Build RAG pipeline for a course. Cached by (course_id, content_hash)
    so it only rebuilds when files actually change.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.schema import Document

    raw_docs = st.session_state.get("raw_docs", [])
    if not raw_docs:
        raise ValueError("No content loaded — cannot build pipeline.")

    documents = [
        Document(page_content=d["text"], metadata={"title": d["title"]})
        for d in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        max_output_tokens=1024,
    )

    course_name = st.session_state.get("course_name", "this course")
    prompt = ChatPromptTemplate.from_template(
        f"""You are a helpful course assistant for {course_name}.
Answer the student's question using ONLY the course content provided below.
Be concise, friendly, and precise.
If the answer is not in the provided content, say clearly:
"I can't find that in the course materials — please check with your instructor."
Do NOT make up information.

Course content:
{{context}}

Student question: {{question}}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('title', 'Untitled')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain_and_retriever, question: str) -> tuple[str, list]:
    chain, retriever = chain_and_retriever
    answer  = chain.invoke(question).strip()
    sources = retriever.invoke(question)
    return answer, sources


# ══════════════════════════════════════════════════════════════════════════════
# COURSE ID + COURSE NAME
# ══════════════════════════════════════════════════════════════════════════════

# Read course_id from URL param; fall back to listing available courses
params      = st.query_params
course_id   = params.get("course_id", "").strip()

# Derive a display name from the folder name or a names.txt mapping
def get_course_name(course_id: str) -> str:
    names_file = CONTENT_ROOT / "course_names.txt"
    if names_file.exists():
        for line in names_file.read_text().splitlines():
            if "=" in line:
                cid, name = line.split("=", 1)
                if cid.strip() == course_id:
                    return name.strip()
    return f"Course {course_id}"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_course_id" not in st.session_state:
    st.session_state.last_course_id = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

course_name = st.session_state.get("course_name", get_course_name(course_id))
st.markdown(f"""
<div class="hero">
    <h1>📚 Course Assistant</h1>
    <div class="subtitle">{course_name}</div>
    <div class="badge">Powered by Gemini · Course Content RAG</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE COURSE ID
# ══════════════════════════════════════════════════════════════════════════════

if not course_id:
    # Show available courses if no course_id provided
    available = [
        d.name for d in CONTENT_ROOT.iterdir()
        if d.is_dir() and any(f.suffix.lower() in SUPPORTED for f in d.iterdir())
    ] if CONTENT_ROOT.exists() else []

    st.error("No course specified in the URL.")
    if available:
        st.info(f"Available courses: {', '.join(available)}\n\nAdd `?course_id=COURSE_ID` to the URL.")
    else:
        st.info("No course content folders found. Create `content/COURSE_ID/` and upload files.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# BUILD / REFRESH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

needs_build = (
    not st.session_state.pipeline_ready
    or st.session_state.last_course_id != course_id
)

if needs_build:
    with st.spinner("Loading course content…"):
        try:
            raw_docs, content_hash = load_course_content(course_id)
            if not raw_docs:
                st.warning(
                    f"No readable files found in content/{course_id}/. "
                    f"Upload .pdf, .docx, .txt, or .html files to that folder."
                )
                st.stop()

            st.session_state["raw_docs"]    = raw_docs
            st.session_state["course_name"] = get_course_name(course_id)
            st.session_state["file_count"]  = len(raw_docs)

        except FileNotFoundError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Could not load content: {e}")
            st.stop()

    with st.spinner("Building knowledge base…"):
        try:
            pipeline = build_pipeline(course_id, content_hash)
            st.session_state["pipeline"]      = pipeline
            st.session_state["pipeline_ready"] = True
            st.session_state["last_course_id"] = course_id
            st.session_state.messages          = []
        except Exception as e:
            st.error(f"⚠️ Could not build pipeline: {e}")
            st.stop()

file_count = st.session_state.get("file_count", 0)
st.markdown(f"""
<div class="status-bar">
    <span class="status-dot"></span>
    {file_count} file{"s" if file_count != 1 else ""} indexed
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            st.markdown(
                f'<div class="source-box">📄 {msg["source"]}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════

if question := st.chat_input("Ask a question about this course…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching course content…"):
            try:
                answer, sources = ask(st.session_state["pipeline"], question)
                source_label = None
                if sources:
                    title   = sources[0].metadata.get("title", "Course material")
                    snippet = sources[0].page_content[:80].replace("\n", " ").strip()
                    source_label = f"{title} — {snippet}…"
            except Exception as e:
                answer       = f"Sorry, something went wrong: {e}"
                source_label = None

        st.markdown(answer)
        if source_label:
            st.markdown(
                f'<div class="source-box">📄 {source_label}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "source": source_label,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;margin-top:1.5rem;font-size:0.7rem;color:#2a3a5a;">
        Answers are grounded in course content only · Always verify with your instructor
    </div>
    """, unsafe_allow_html=True)
