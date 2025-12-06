import streamlit as st
import os
import json
import zipfile
import requests
import numpy as np
import faiss
import re

from io import BytesIO
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ============================================================
# 0. 환경 변수 (Streamlit Cloud Secrets 사용)
# ============================================================
GENERAL_API_KEY = st.secrets["GENERAL_API_KEY"]
FINETUNE_API_KEY = st.secrets["FINETUNE_API_KEY"]

os.environ["OPENAI_API_KEY"] = GENERAL_API_KEY

general_client = OpenAI(api_key=GENERAL_API_KEY)
finetune_client = OpenAI(api_key=FINETUNE_API_KEY)

FINETUNED_MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"


# ============================================================
# 1. Release v1.1 파일 다운로드
# ============================================================

DOC_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_documents.json"
EMB_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_embeddings.zip"


@st.cache_resource
def download_and_load_data():
    # ---- documents.json 다운로드 ----
    doc_res = requests.get(DOC_URL)
    docs_json = json.loads(doc_res.content)

    docs = docs_json["documents"]  # 리스트 of 텍스트

    # ---- embeddings.zip 다운로드 ----
    emb_res = requests.get(EMB_URL)
    zip_bytes = BytesIO(emb_res.content)

    with zipfile.ZipFile(zip_bytes, "r") as z:
        vecs = np.load(BytesIO(z.read("_embeddings.npy")))

    return docs, vecs


# ============================================================
# 2. VectorStore 생성
# ============================================================
@st.cache_resource
def create_vectorstore(_docs, _vectors):
    dim = _vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(_vectors).astype("float32"))

    wrapped = [Document(page_content=d) for d in _docs]
    doc_dict = {str(i): wrapped[i] for i in range(len(wrapped))}
    docstore = InMemoryDocstore(doc_dict)
    mapping = {i: str(i) for i in range(len(wrapped))}

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=mapping,
    )
    return vs


# ============================================================
# 3. RAG Chain 구성
# ============================================================
@st.cache_resource
def build_rag(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 계약서 검색 AI입니다.
아래 문서를 참고하여 질문에 대한 답을 정확하게 반환하세요.
문서 파일명은 절대 출력하지 마세요.

{context}
""",
            ),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain


# ============================================================
# 4. 쉬운 설명 (파인튜닝 모델)
# ============================================================
def explain_with_finetuned_model(clause):
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "계약서를 초등학생도 이해하게 쉽게 요약해줘. 반드시 1~4문장.",
                },
                {"role": "user", "content": clause},
            ],
        )
        return res.choices[0].message.content
    except:
        return "⚠️ 쉬운 설명 모델을 불러오지 못했습니다."


# ============================================================
# 5. 위험 요소 분석
# ============================================================
def analyze_risk_with_general_llm(clause):
    prompt = f"""
다음 계약서 조항에서 근로자에게 불리할 수 있는 위험 요소를 2~3개 찾고,
각각 왜 위험한지 설명해주세요.

{clause}
"""
    res = general_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# ============================================================
# 6. UI 스타일 (카톡형)
# ============================================================
st.set_page_config(page_title="계약서 이해 AI", layout="wide")

st.markdown(
    """
<style>
body { background:#ecf0f7; }
.block-container { padding-top:2rem; }

/* 채팅 스타일 */
.chat-row { display:flex; margin-bottom:12px; }
.chat-row.user { justify-content:flex-end; }
.bubble {
    padding:10px 14px;
    border-radius:16px;
    max-width:420px;
    line-height:1.4;
}
.bubble.user {
    background:#2f80ff; color:white;
    border-bottom-right-radius:4px;
}
.bubble.bot {
    background:white;
    border:1px solid #d8e2f1;
    border-bottom-left-radius:4px;
}
.answer-card{
    padding:12px; border-radius:16px;
    border:1px solid #d8e2f1;
    background:#fff;
}
.answer-title{ font-weight:600; margin-bottom:6px; }
</style>
""",
    unsafe_allow_html=True,
)

# 메시지 저장
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "type": "intro",
            "content": "안녕하세요! 계약서 이해 AI입니다. 궁금한 조항을 검색해보세요.",
        }
    ]


# ============================================================
# 7. 데이터 로드 + RAG 준비
# ============================================================
docs, vectors = download_and_load_data()
vectorstore = create_vectorstore(docs, vectors)
rag_chain = build_rag(vectorstore)


# ============================================================
# 8. 채팅 입력
# ============================================================
user_query = st.chat_input("검색할 계약서 조항을 입력하세요.")

if user_query:
    st.session_state.messages.append(
        {"role": "human", "type": "user", "content": user_query}
    )

    rag_out = rag_chain.invoke({"input": user_query})
    clause_raw = rag_out["answer"]
    clause_raw = re.sub(r"\\[[^\\]]+\\]", "", clause_raw)

    easy = explain_with_finetuned_model(clause_raw)
    risk = analyze_risk_with_general_llm(clause_raw)

    st.session_state.messages.append(
        {
            "role": "ai",
            "type": "answer",
            "clause": clause_raw,
            "easy": easy,
            "risk": risk,
        }
    )


# ============================================================
# 9. 렌더링
# ============================================================
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
        st.markdown(
            f"""
<div class="chat-row">
    <div class="bubble bot">{msg['content']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    elif msg["type"] == "answer":
        st.markdown(
            f"""
<div class="chat-row">
    <div class="answer-card">
        <div class="answer-title">🔵 관련 계약서 조항</div>
        <div>{msg['clause']}</div>

        <div class="answer-title" style="margin-top:10px;">✨ 쉬운 설명</div>
        <div>{msg['easy']}</div>

        <div class="answer-title" style="margin-top:10px;">⚠️ 위험 요소 분석</div>
        <div>{msg['risk']}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
