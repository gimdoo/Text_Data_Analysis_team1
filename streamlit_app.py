import streamlit as st
import pickle
import requests
import numpy as np
import faiss
import re

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ---------------------------------------------------------
# 0. API KEY
# ---------------------------------------------------------
st.set_page_config(page_title="계약서 이해 AI", layout="wide")

GENERAL_API_KEY = st.secrets["GENERAL_API_KEY"]
FINETUNE_API_KEY = st.secrets["FINETUNE_API_KEY"]

general_client = OpenAI(api_key=GENERAL_API_KEY)
finetune_client = OpenAI(api_key=FINETUNE_API_KEY)

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"


# ---------------------------------------------------------
# 1. GitHub Release 파일 로딩
# ---------------------------------------------------------
@st.cache_resource
def load_pickle_from_url(url: str):
    res = requests.get(url)
    return pickle.loads(res.content)


DOC_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.0/_documents.pkl"
EMB_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.0/_embeddings.pkl"

docs = load_pickle_from_url(DOC_URL)
vectors = load_pickle_from_url(EMB_URL)


# ---------------------------------------------------------
# 2. FAISS VectorStore 생성
# ---------------------------------------------------------
@st.cache_resource
def create_vectorstore(docs, vectors):

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    wrapped_docs = [Document(page_content=d) for d in docs]
    doc_dict = {str(i): wrapped_docs[i] for i in range(len(docs))}

    docstore = InMemoryDocstore(doc_dict)
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=GENERAL_API_KEY
    )

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    return vectorstore


vectorstore = create_vectorstore(docs, vectors)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------------------------------------------------
# 3. RAG 검색 (직접 구현)
# ---------------------------------------------------------
def search_clause(query):
    results = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in results])


# ---------------------------------------------------------
# 4. 쉬운 설명 (파인튜닝 모델)
# ---------------------------------------------------------
def explain_easy(clause: str):
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL,
            messages=[
                {"role": "system", "content": "계약서를 쉽고 짧게 설명하십시오."},
                {"role": "user", "content": clause},
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content

    except Exception as e:
        return f"⚠️ 쉬운 설명 모델 오류: {e}"


# ---------------------------------------------------------
# 5. 위험 요소 분석 (gpt-4o)
# ---------------------------------------------------------
def analyze_risk(clause: str):
    prompt = f"""
계약서 조항에서 근로자에게 불리하거나 주의해야 할 위험 요소를 2~3개 요약하고,
각 항목이 왜 위험한지 짧게 설명하세요.

{clause}
"""
    res = general_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content


# ---------------------------------------------------------
# 6. UI CSS (그대로 유지)
# ---------------------------------------------------------
st.markdown("""
<style>
body { background: #edf2f7; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.chat-inner { max-width: 900px; margin: 0 auto; }

.chat-row.user { display: flex; justify-content: flex-end; margin-top: 12px; }
.bubble.user {
    background: #2563eb; color: white;
    padding: 12px 16px; border-radius: 14px;
    max-width: 420px; border-bottom-right-radius: 4px;
}

.chat-row.bot { display: flex; gap: 10px; margin-top: 16px; }
.avatar { width: 28px; height: 28px; border-radius: 999px; background: #e2e8f0;
          display: flex; align-items: center; justify-content: center; }

.answer-card {
    background: white;
    border: 1px solid #d1d5db;
    padding: 14px 18px;
    border-radius: 14px;
    margin-top: 10px;
}
.answer-title { font-weight: 700; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 7. 세션 초기화
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "type": "intro", "content": "안녕하세요! 계약서 이해 AI입니다.\n궁금한 계약서 조항을 입력해주세요."}
    ]


# ---------------------------------------------------------
# 8. 사용자 입력 처리
# ---------------------------------------------------------
user_query = st.chat_input("메시지를 입력하세요")

if user_query:
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_query})

    clause_raw = search_clause(user_query)
    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", clause_raw)

    easy = explain_easy(clause)
    risk = analyze_risk(clause)

    st.session_state.messages.append({
        "role": "bot",
        "type": "answer",
        "clause": clause,
        "easy": easy,
        "risk": risk,
    })


# ---------------------------------------------------------
# 9. 메시지 렌더링
# ---------------------------------------------------------
st.markdown('<div class="chat-inner">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="bubble user">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True
        )

    elif msg["type"] == "intro":
        st.markdown(
            f"""
            <div class="chat-row bot">
                <div class="avatar">🤖</div>
                <div class="bubble bot">{msg['content'].replace("\n", "<br>")}</div>
            </div>
            """, unsafe_allow_html=True
        )

    elif msg["type"] == "answer":
        st.markdown(
            f"""
            <div class="chat-row bot">
                <div class="avatar">🤖</div>
                <div class="answer-card">

                    <div class="answer-title">🔵 관련 계약서 조항</div>
                    <div>{msg['clause'].replace("\n", "<br>")}</div>

                    <div class="answer-title" style="margin-top:12px;">✨ 쉬운 설명</div>
                    <div>{msg['easy'].replace("\n", "<br>")}</div>

                    <div class="answer-title" style="margin-top:12px;">⚠️ 위험 요소</div>
                    <div>{msg['risk'].replace("\n", "<br>")}</div>

                </div>
            </div>
            """, unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
