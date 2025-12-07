#streamlit cloud 와 연동을 위해 lawchatapp 코드를 변환시킨 코드 입니다.
#Release에 있는 streamlit_app.py를 실행시키기 위한 데이터 파일을 사용했습니다.

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
    page_title="계약서 도우미",
    page_icon="📄",
    layout="wide",
)

# --------------------------------------------------------------------------
# 1. API Key (Streamlit Secrets 사용)
# --------------------------------------------------------------------------
GENERAL_API_KEY = st.secrets.get("GENERAL_API_KEY")
FINETUNE_API_KEY = st.secrets.get("FINETUNE_API_KEY")

if not GENERAL_API_KEY or not FINETUNE_API_KEY:
    st.error(
        "🔐 API 키가 설정되지 않았습니다.\n\n"
        "Streamlit Cloud의 **Manage app → Settings → Secrets** 에서\n"
        "`GENERAL_API_KEY`, `FINETUNE_API_KEY` 값을 추가해 주세요."
    )
    st.stop()

openai_client = OpenAI(api_key=GENERAL_API_KEY)
finetune_client = OpenAI(api_key=FINETUNE_API_KEY)

# ⚠️ 너가 만든 파인튜닝 모델 ID
FINETUNED_MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"

# --------------------------------------------------------------------------
# 2. GitHub Release 에서 문서 / 임베딩 다운로드
# --------------------------------------------------------------------------
DOC_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_documents.json"
EMB_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_embeddings.zip"


@st.cache_resource
def download_and_load_data():
    # ---- 문서 JSON ----
    doc_res = requests.get(DOC_URL)
    doc_res.raise_for_status()

    try:
        docs_json = json.loads(doc_res.text)
    except json.JSONDecodeError:
        st.error("📄 문서 JSON을 파싱할 수 없습니다. _documents.json 형식을 확인해 주세요.")
        st.stop()

    # 형식 방어: dict 또는 list 모두 처리
    if isinstance(docs_json, dict) and "documents" in docs_json:
        docs = docs_json["documents"]
    elif isinstance(docs_json, list):
        docs = docs_json
    else:
        st.error("📄 _documents.json 형식이 예상과 다릅니다. (list 또는 { 'documents': [...] } 형태여야 합니다.)")
        st.stop()

    # ---- 임베딩 ZIP(npz) ----
    emb_res = requests.get(EMB_URL)
    emb_res.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(emb_res.content))
    npz_files = [name for name in zf.namelist() if name.endswith(".npz")]

    if not npz_files:
        st.error("📦 _embeddings.zip 안에 .npz 파일이 없습니다.")
        st.stop()

    # 첫 번째 npz 사용
    with zf.open(npz_files[0]) as f:
        npz = np.load(f)
        if "arr_0" in npz.files:
            vectors = npz["arr_0"]
        else:
            # 키 이름이 다른 경우: 첫 번째 배열 사용
            vectors = npz[npz.files[0]]

    return docs, vectors


# --------------------------------------------------------------------------
# 3. FAISS 벡터스토어 생성
# --------------------------------------------------------------------------
@st.cache_resource
def create_vectorstore(_docs, _vectors):
    # FAISS 인덱스 생성
    dim = _vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(_vectors.astype("float32"))

    # LangChain 문서 래핑
    wrapped_docs = [
        Document(page_content=d) if isinstance(d, str) else d
        for d in _docs
    ]

    doc_dict = {str(i): wrapped_docs[i] for i in range(len(wrapped_docs))}
    docstore = InMemoryDocstore(doc_dict)
    index_to_docstore_id = {i: str(i) for i in range(len(wrapped_docs))}

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=GENERAL_API_KEY,
    )

    vectorstore = FAISS(
        embedding_function=embeddings,  # 최신 버전에서도 동작
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vectorstore


# --------------------------------------------------------------------------
# 4. RAG 체인 초기화
# --------------------------------------------------------------------------
@st.cache_resource
def initialize_rag_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 계약서 조항 검색 AI입니다.
다음 문서를 참고하여 정확하게 답하세요.
⚠️ 문서의 파일명이나 식별자는 절대 출력하지 마세요.

{context}""",
            ),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=GENERAL_API_KEY,
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# --------------------------------------------------------------------------
# 5. 파인튜닝 LLM (쉬운 설명)
# --------------------------------------------------------------------------
def explain_with_finetuned_model(clause: str) -> str:
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 계약서를 쉽게 설명하는 도우미입니다. "
                        "반드시 1~4문장으로 핵심만 요약해서 말하세요."
                    ),
                },
                {"role": "user", "content": clause},
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content
    except Exception as e:
        # 여기서 API Key / 프로젝트 키 mismatch 같은 에러도 포착됨
        return f"⚠️ 현재 쉬운 설명 모델 호출에 문제가 있습니다: {e}"


# --------------------------------------------------------------------------
# 6. 일반 LLM (위험 요소 분석)
# --------------------------------------------------------------------------
def analyze_risk_with_general_llm(clause: str) -> str:
    prompt = f"""
다음 계약서 조항에서 근로자에게 불리하거나
주의해야 할 위험 요소를 2~3개 요약하고,
각 항목마다 왜 위험한지도 설명하세요.

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
        return f"⚠️ 위험 요소 분석 중 오류가 발생했습니다: {e}"


# --------------------------------------------------------------------------
# 7. UI 스타일 (너가 만든 카톡 느낌 그대로)
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
/* 전체 배경 */
body {
    background: #edf2f7;
}

/* Streamlit 기본 여백 조금 줄이기 */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* 카톡처럼 가운데 축 */
.chat-inner {
    max-width: 720px;
    margin: 0 auto;
}

/* 상단 제목 영역 */
.header-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.menu-icon {
    width: 30px;
    height: 30px;
    border-radius: 999px;
    border: 1px solid #d4e0f4;
    display: flex;
    align-items: center;
    justify-content: center;
}

.menu-icon-bar {
    width: 14px;
    height: 2px;
    border-radius: 999px;
    background: #4b6bb6;
    position: relative;
}
.menu-icon-bar::before,
.menu-icon-bar::after {
    content: "";
    position: absolute;
    width: 14px;
    height: 2px;
    border-radius: 999px;
    background: #4b6bb6;
    left: 0;
}
.menu-icon-bar::before { top: -4px; }
.menu-icon-bar::after  { top:  4px; }

.app-title {
    font-size: 26px;
    font-weight: 750;
    color: #1f2a4d;
}

.app-subtitle {
    font-size: 13px;
    color: #7a8aad;
}

/* 한 줄(행) */
.chat-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin-bottom: 12px;
}
.chat-row.user {
    justify-content: flex-end;
}

/* 아바타 */
.avatar {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: #e9f1ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    color: #4b7cf5;
}
.avatar.bot {
    background: #edf1f9;
    color: #7b8baa;
}

/* 기본 말풍선 */
.bubble {
    border-radius: 18px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.4;
    max-width: 420px;
    word-break: keep-all;
}
.bubble.user {
    background: #2f80ff;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: #ffffff;
    border: 1px solid #dfe7f5;
    color: #1f2937;
    border-bottom-left-radius: 4px;
}

/* 최초 인사 카드 */
.bot-card {
    max-width: 520px;
    border-radius: 16px;
    border: 1px solid #dde5f2;
    background: #ffffff;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.bot-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #34415f;
    padding-bottom: 6px;
    border-bottom: 1px solid #edf1f9;
}
.bot-card-avatar {
    width: 22px;
    height: 22px;
    border-radius: 999px;
    border: 1px solid #d1ddf5;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
}
.bot-card-body {
    font-size: 14px;
    line-height: 1.5;
    color: #2f3a54;
}

/* RAG 결과 카드 */
.answer-card {
    max-width: 520px;
    border-radius: 16px;
    border: 1px solid #dfe7f5;
    background: #ffffff;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.5;
}
.answer-section {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed #e4e9f5;
}
.answer-section:first-child {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}
.answer-section-title {
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
    color: #1f2a4d;
}
.answer-section-body {
    font-size: 13.5px;
    color: #374151;
}

/* 입력창도 가운데 정렬 */
.stChatInput {
    margin-top: 1.2rem;
}
.stChatInput > div {
    max-width: 960px;
    margin: 0 auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 8. 세션 채팅 기록 초기화
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "type": "intro",
            "content": (
                "안녕하세요, 계약서 도우미입니다.\n"
                "궁금한 계약서 조항이나 키워드를 아래 입력창에 적어 주세요.\n"
                '예: "근로시간 조항 설명해줘"'
            ),
        }
    ]

# --------------------------------------------------------------------------
# 9. 데이터 / 체인 준비
# --------------------------------------------------------------------------
docs, vectors = download_and_load_data()
vectorstore = create_vectorstore(docs, vectors)
rag_chain = initialize_rag_chain(vectorstore)

# --------------------------------------------------------------------------
# 10. 사용자 입력 처리
# --------------------------------------------------------------------------
user_query = st.chat_input("메시지를 입력하세요")

if user_query:
    # 1) 사용자 메시지 저장
    st.session_state.messages.append(
        {"role": "human", "type": "user", "content": user_query}
    )

    # 2) RAG 검색 + 파인튜닝 + 위험 분석
    rag_response = rag_chain.invoke({"input": user_query})
    raw_clause = rag_response.get("answer", "")

    # [파일명.json] 제거
    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause)

    easy = explain_with_finetuned_model(clause)
    risk = analyze_risk_with_general_llm(clause)

    # 3) AI 답변 구조화해서 저장
    st.session_state.messages.append(
        {
            "role": "ai",
            "type": "answer",
            "query": user_query,
            "clause": clause,
            "easy": easy,
            "risk": risk,
        }
    )

# --------------------------------------------------------------------------
# 11. 채팅 UI 렌더링
# --------------------------------------------------------------------------
st.markdown('<div class="chat-inner">', unsafe_allow_html=True)

# 헤더
st.markdown(
    """
<div class="header-row">
  <div class="menu-icon">
    <div class="menu-icon-bar"></div>
  </div>
  <div>
    <div class="app-title">계약서 도우미</div>
    <div class="app-subtitle">계약서 조항 검색 · 쉬운 설명 · 위험 요소 분석</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# 메시지들 렌더링
for msg in st.session_state.messages:
    if msg["role"] == "human":
        # 사용자 말풍선
        st.markdown(
            f"""
<div class="chat-row user">
  <div class="bubble user">{msg['content']}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        if msg["type"] == "intro":
            body = msg["content"].replace("\n", "<br />")
            st.markdown(
                f"""
<div class="chat-row">
  <div class="avatar bot">👤</div>
  <div class="bot-card">
    <div class="bot-card-header">
      <div class="bot-card-avatar">🤖</div>
      <div>계약서 도우미</div>
    </div>
    <div class="bot-card-body">
      {body}
    </div>
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
  <div class="avatar bot">👤</div>
  <div class="answer-card">
    <div class="answer-section">
      <div class="answer-section-title">🔵 관련 계약서 조항</div>
      <div class="answer-section-body">{clause_html}</div>
    </div>
    <div class="answer-section">
      <div class="answer-section-title">✨ 쉬운 설명</div>
      <div class="answer-section-body">{easy_html}</div>
    </div>
    <div class="answer-section">
      <div class="answer-section-title">⚠️ 위험 요소 요약</div>
      <div class="answer-section-body">{risk_html}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

st.markdown("</div>", unsafe_allow_html=True)




