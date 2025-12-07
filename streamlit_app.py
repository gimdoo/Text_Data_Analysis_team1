import streamlit as st
import requests
import json
import zipfile
import io
import numpy as np
import faiss
import re

from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --------------------------------------------------------------------------
# 0. Streamlit 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="계약서 이해 AI",
    page_icon="📄",
    layout="wide",
)


# --------------------------------------------------------------------------
# 1. API Key (Streamlit Secrets)
# --------------------------------------------------------------------------
GENERAL_API_KEY = st.secrets.get("GENERAL_API_KEY")
FINETUNE_API_KEY = st.secrets.get("FINETUNE_API_KEY")

if not GENERAL_API_KEY or not FINETUNE_API_KEY:
    st.error(
        "🔐 API 키가 설정되지 않았습니다.\n\n"
        "Streamlit Cloud → Settings → Secrets 에서\n"
        "`GENERAL_API_KEY`, `FINETUNE_API_KEY` 값을 추가하세요."
    )
    st.stop()

openai_client = OpenAI(api_key=GENERAL_API_KEY)
finetune_client = OpenAI(api_key=FINETUNE_API_KEY)

# ⚠️ 파인튜닝 모델 ID
FINETUNED_MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"


# --------------------------------------------------------------------------
# 2. GitHub Release 데이터 다운로드
# --------------------------------------------------------------------------
DOC_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_documents.json"
EMB_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_embeddings.zip"


@st.cache_data
def download_and_load_data():
    # ---- 문서 JSON 다운로드 ----
    doc_res = requests.get(DOC_URL)
    doc_res.raise_for_status()

    try:
        docs_json = json.loads(doc_res.text)
    except Exception:
        st.error("📄 _documents.json 파싱 실패!")
        st.stop()

    # 다양한 형식을 허용
    if isinstance(docs_json, dict):
        if "documents" in docs_json:
            docs = docs_json["documents"]
        else:
            docs = list(docs_json.values())
    elif isinstance(docs_json, list):
        docs = docs_json
    else:
        st.error("📄 JSON 구조가 예상과 다릅니다.")
        st.stop()

    # ---- 임베딩 ZIP 다운로드 ----
    emb_res = requests.get(EMB_URL)
    emb_res.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(emb_res.content))
    npz_files = [x for x in zf.namelist() if x.endswith(".npz")]

    if not npz_files:
        st.error("⚠️ 임베딩 ZIP 안에 .npz가 없습니다.")
        st.stop()

    with zf.open(npz_files[0]) as f:
        npz = np.load(f)
        vectors = npz[npz.files[0]]

    return docs, vectors


# --------------------------------------------------------------------------
# 3. FAISS Vectorstore 생성
# --------------------------------------------------------------------------
@st.cache_data
def create_vectorstore(_docs, _vectors):

    dim = _vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(_vectors.astype("float32"))

    wrapped_docs = [
        Document(page_content=d) if isinstance(d, str) else d
        for d in _docs
    ]

    doc_dict = {str(i): wrapped_docs[i] for i in range(len(wrapped_docs))}
    index_to_docstore_id = {i: str(i) for i in range(len(wrapped_docs))}

    docstore = InMemoryDocstore(doc_dict)

    # ❗ 사전 임베딩 사용 → embedding_function=None
    vectorstore = FAISS(
        embedding_function=None,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vectorstore


# --------------------------------------------------------------------------
# 4. RAG 체인 초기화
# --------------------------------------------------------------------------
@st.cache_data
def initialize_rag_chain(_vectorstore):

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_prompt = ChatPromptTemplate.from_template("""
당신은 계약서 조항 검색 AI입니다.
반드시 문서를 기반으로 정확히 답변하세요.
⚠️ 문서 파일명이나 식별자는 절대 출력하지 마세요.

[참고 문서]
{context}

[질문]
{input}
""")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=GENERAL_API_KEY,
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# --------------------------------------------------------------------------
# 5. 파인튜닝 모델 "쉬운 설명"
# --------------------------------------------------------------------------
def explain_with_finetuned_model(clause: str):

    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 계약서를 쉽게 설명하는 도우미입니다. 반드시 1~4문장으로 핵심만 요약하세요."
                },
                {"role": "user", "content": clause},
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content

    except Exception as e:
        return f"⚠️ 쉬운 설명 모델 호출 오류: {e}"


# --------------------------------------------------------------------------
# 6. 일반 모델 "위험요소 분석"
# --------------------------------------------------------------------------
def analyze_risk_with_general_llm(clause: str):

    prompt = f"""
다음 계약서 조항에서 직원에게 불리하거나 주의해야 할 위험 요소를 2~3개 요약하고,
각 항목이 왜 위험한지도 간단히 설명하세요.

{clause}
"""

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return res.choices[0].message.content

    except Exception as e:
        return f"⚠️ 위험요소 분석 오류: {e}"


# --------------------------------------------------------------------------
# 7. UI 스타일 (카톡 느낌)
# --------------------------------------------------------------------------
st.markdown("""
<style>
/* (생략: 너가 만든 기존 CSS 100% 동일 유지) */
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 8. 세션 초기화
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "type": "intro",
            "content": (
                "안녕하세요, 계약서 이해 AI입니다.\n"
                "궁금한 조항이나 단어를 입력해 주세요.\n"
                '예: "근로시간 조항 설명해줘"'
            ),
        }
    ]


# --------------------------------------------------------------------------
# 9. 데이터 로딩
# --------------------------------------------------------------------------
docs, vectors = download_and_load_data()
vectorstore = create_vectorstore(docs, vectors)
rag_chain = initialize_rag_chain(vectorstore)


# --------------------------------------------------------------------------
# 10. 사용자 입력
# --------------------------------------------------------------------------
user_query = st.chat_input("메시지를 입력하세요")

if user_query:
    st.session_state.messages.append(
        {"role": "human", "type": "user", "content": user_query}
    )

    rag_response = rag_chain.invoke({"input": user_query})
    raw_clause = rag_response.get("answer", "")

    # 파일명 제거 (강화된 정규식)
    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause, flags=re.IGNORECASE)

    easy = explain_with_finetuned_model(clause)
    risk = analyze_risk_with_general_llm(clause)

    st.session_state.messages.append(
        {
            "role": "ai",
            "type": "answer",
            "clause": clause,
            "easy": easy,
            "risk": risk,
        }
    )


# --------------------------------------------------------------------------
# 11. 메시지 렌더링 (카톡 UI 유지)
# --------------------------------------------------------------------------
st.markdown('<div class="chat-inner">', unsafe_allow_html=True)

# 헤더 UI (생략 가능)
st.markdown("""
<div class="header-row">
  <div class="menu-icon"><div class="menu-icon-bar"></div></div>
  <div>
    <div class="app-title">계약서 이해 AI</div>
    <div class="app-subtitle">계약서 조항 검색 · 쉬운 설명 · 위험 요소 분석</div>
  </div>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "human":
        st.markdown(
            f"""
<div class="chat-row user">
  <div class="bubble user">{msg['content']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    elif msg["type"] == "intro":
        body = msg["content"].replace("\n", "<br />")
        st.markdown(
            f"""
<div class="chat-row">
  <div class="avatar bot">🤖</div>
  <div class="bot-card">
    <div class="bot-card-header">
      <div class="bot-card-avatar">🤖</div>
      <div>계약서 이해 도우미</div>
    </div>
    <div class="bot-card-body">{body}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    elif msg["type"] == "answer":
        clause_html = msg["clause"].replace("\n", "<br />")
        easy_html = msg["easy"].replace("\n", "<br />")
        risk_html = msg["risk"].replace("\n", "<br />")

        st.markdown(
            f"""
<div class="chat-row">
  <div class="avatar bot">🤖</div>
  <div class="answer-card">
    <div class="answer-section">
      <div class="answer-section-title">📘 관련 조항</div>
      <div class="answer-section-body">{clause_html}</div>
    </div>

    <div class="answer-section">
      <div class="answer-section-title">✨ 쉬운 설명</div>
      <div class="answer-section-body">{easy_html}</div>
    </div>

    <div class="answer-section">
      <div class="answer-section-title">⚠️ 위험 요소</div>
      <div class="answer-section-body">{risk_html}</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
