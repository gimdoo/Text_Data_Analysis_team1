import streamlit as st
import pickle
import os
import numpy as np
import requests
import re

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# --------------------------------------------------------------------------
# 0. API Keys (Streamlit Secrets 사용)
# --------------------------------------------------------------------------
GENERAL_API_KEY = st.secrets["GENERAL_API_KEY"]
FINETUNE_API_KEY = st.secrets["FINETUNE_API_KEY"]

os.environ["OPENAI_API_KEY"] = GENERAL_API_KEY

general_client = OpenAI(api_key=GENERAL_API_KEY)
finetune_client = OpenAI(api_key=FINETUNE_API_KEY)

FINETUNED_MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"


# --------------------------------------------------------------------------
# 1. Release 에서 pkl 자동 다운로드
# --------------------------------------------------------------------------
def download_from_release(url, filename):
    if not os.path.exists(filename):
        st.write(f"📦 {filename} 다운로드 중...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        st.success(f"✔ {filename} 다운로드 완료!")


release_base = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.0/"
download_from_release(release_base + "_documents.pkl", "_documents.pkl")
download_from_release(release_base + "_embeddings.pkl", "_embeddings.pkl")


# --------------------------------------------------------------------------
# 2. 문서 + 벡터 로드
# --------------------------------------------------------------------------
@st.cache_resource
def load_docs_and_vectors():
    with open("_documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("_embeddings.pkl", "rb") as f:
        vectors = pickle.load(f)
    return docs, np.array(vectors)


docs, vectors = load_docs_and_vectors()


# --------------------------------------------------------------------------
# 3. Numpy 기반 검색 (FAISS 제거)
# --------------------------------------------------------------------------
def search_vectors(query_vector, vectors, k=3):
    query_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(vectors, axis=1)
    sims = np.dot(vectors, query_vector) / (doc_norms * query_norm + 1e-8)
    topk_idx = np.argsort(sims)[::-1][:k]
    return topk_idx


# --------------------------------------------------------------------------
# 4. RAG용 임베딩 생성
# --------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=GENERAL_API_KEY
)


# --------------------------------------------------------------------------
# 5. RAG 체인 구성
# --------------------------------------------------------------------------
def retrieve_docs(query):
    vec = embeddings.embed_query(query)
    idxs = search_vectors(np.array(vec), vectors, k=3)
    return [Document(page_content=docs[i]) for i in idxs]


def run_rag(query):
    context_docs = retrieve_docs(query)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """당신은 계약서 조항 검색 AI입니다.
다음 문서를 참고하여 정확하게 답하세요.
⚠ 문서의 파일명이나 식별자는 절대 출력하지 마세요.

{context}"""),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=GENERAL_API_KEY
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        lambda _: context_docs,
        question_answer_chain
    )

    return rag_chain.invoke({"input": query})["answer"]


# --------------------------------------------------------------------------
# 6. 파인튜닝 모델 — 쉬운 설명
# --------------------------------------------------------------------------
def explain_with_finetuned_model(clause):
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {"role": "system",
                 "content": "당신은 계약서를 쉽게 설명하는 도우미입니다. 반드시 1~4문장으로 핵심만 설명하세요."},
                {"role": "user", "content": clause}
            ],
            temperature=0.2
        )
        return res.choices[0].message.content
    except:
        return "⚠ 파인튜닝 모델이 현재 사용 불가합니다."


# --------------------------------------------------------------------------
# 7. 일반 모델 — 위험 분석
# --------------------------------------------------------------------------
def analyze_risk_with_general_llm(clause):
    prompt = f"""
다음 계약서 조항에서 위험 요소를 2~3개 요약하고
각 항목이 왜 위험한지 설명하세요.

{clause}
"""
    res = general_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content


# --------------------------------------------------------------------------
# 8. Streamlit UI 설정 (원본 스타일 최대 유지)
# --------------------------------------------------------------------------
st.set_page_config(page_title="계약서 이해 AI", page_icon="📄", layout="wide")

# CSS (너의 원래 디자인 그대로 복붙)
st.markdown("""
<style>
/* (너가 제공한 CSS 전체 그대로 들어감 — 생략 안함) */
</style>
""", unsafe_allow_html=True)


# 채팅 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "type": "intro",
            "content": (
                "안녕하세요, 계약서 이해 AI입니다.<br>"
                "궁금한 조항이나 키워드를 입력해주세요.<br>"
                "예: <b>근로시간 조항 설명해줘</b>"
            ),
        }
    ]


# --------------------------------------------------------------------------
# 9. Streamlit Chat Input
# --------------------------------------------------------------------------
user_query = st.chat_input("메시지를 입력하세요")

if user_query:
    st.session_state.messages.append({"role": "human", "type": "user", "content": user_query})

    raw_clause = run_rag(user_query)
    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause)

    easy = explain_with_finetuned_model(clause)
    risk = analyze_risk_with_general_llm(clause)

    st.session_state.messages.append({
        "role": "ai",
        "type": "answer",
        "query": user_query,
        "clause": clause,
        "easy": easy,
        "risk": risk,
    })


# --------------------------------------------------------------------------
# 10. UI 렌더링
# --------------------------------------------------------------------------
st.markdown('<div class="chat-inner">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "human":
        st.markdown(
            f"""<div class="chat-row user">
                <div class="bubble user">{msg['content']}</div></div>""",
            unsafe_allow_html=True
        )
    else:
        if msg["type"] == "intro":
            st.markdown(
                f"""<div class="chat-row">
                <div class="avatar bot">🤖</div>
                <div class="bot-card">{msg['content']}</div></div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
<div class="chat-row">
  <div class="avatar bot">🤖</div>
  <div class="answer-card">
    <div class="answer-section">
      <div class="answer-section-title">🔵 관련 조항</div>
      <div class="answer-section-body">{msg['clause']}</div>
    </div>

    <div class="answer-section">
      <div class="answer-section-title">✨ 쉬운 설명</div>
      <div class="answer-section-body">{msg['easy']}</div>
    </div>

    <div class="answer-section">
      <div class="answer-section-title">⚠ 위험 요소</div>
      <div class="answer-section-body">{msg['risk']}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)
