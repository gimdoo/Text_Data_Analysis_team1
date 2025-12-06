import streamlit as st
import pickle
import os
import faiss
import numpy as np
import re
import requests

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# --------------------------------------------------------------------------
# 0. Streamlit Cloud: Secrets에서 API 불러오기
# --------------------------------------------------------------------------
GENERAL_API_KEY = st.secrets["GENERAL_API_KEY"]
FINETUNE_API_KEY = st.secrets["FINETUNE_API_KEY"]

os.environ["OPENAI_API_KEY"] = GENERAL_API_KEY

finetune_client = OpenAI(api_key=FINETUNE_API_KEY)
general_client = OpenAI(api_key=GENERAL_API_KEY)

FINETUNED_MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"


# --------------------------------------------------------------------------
# 1. GitHub Release에서 pkl 자동 다운로드
# --------------------------------------------------------------------------
def download_from_release(url, filename):
    if not os.path.exists(filename):
        st.write(f"📦 {filename} 다운로드 중...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        st.success(f"✅ {filename} 다운로드 완료!")


release_base = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.0/"
download_from_release(release_base + "_documents.pkl", "_documents.pkl")
download_from_release(release_base + "_embeddings.pkl", "_embeddings.pkl")


# --------------------------------------------------------------------------
# 2. pkl 로드
# --------------------------------------------------------------------------
@st.cache_resource
def load_docs_and_vectors():
    with open("_documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("_embeddings.pkl", "rb") as f:
        vectors = pickle.load(f)
    return docs, vectors


# --------------------------------------------------------------------------
# 3. 벡터스토어 구성 (FAISS)
# --------------------------------------------------------------------------
@st.cache_resource
def create_vectorstore(_docs, _vectors):
    dim = len(_vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(_vectors).astype("float32"))

    wrapped_docs = [
        Document(page_content=d) if isinstance(d, str) else d
        for d in _docs
    ]

    doc_dict = {str(i): wrapped_docs[i] for i in range(len(wrapped_docs))}
    docstore = InMemoryDocstore(doc_dict)
    index_to_docstore_id = {i: str(i) for i in range(len(wrapped_docs))}

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=GENERAL_API_KEY
    )

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vectorstore


# --------------------------------------------------------------------------
# 4. RAG 체인 생성
# --------------------------------------------------------------------------
@st.cache_resource
def initialize_rag_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """당신은 계약서 조항 검색 AI입니다.
다음 문서를 참고하여 정확하게 답하세요.
⚠️ 문서의 파일명이나 식별자는 절대 출력하지 마세요.

{context}"""),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=GENERAL_API_KEY
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# --------------------------------------------------------------------------
# 5. 파인튜닝 모델 – 쉬운 설명
# --------------------------------------------------------------------------
def explain_with_finetuned_model(clause: str):
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 계약서를 쉽게 설명하는 도우미입니다. 반드시 1~4문장으로 핵심만 요약해서 말하세요."
                },
                {"role": "user", "content": clause}
            ],
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception:
        return "⚠️ 현재 쉬운 설명 모델이 정상적으로 동작하지 않습니다."


# --------------------------------------------------------------------------
# 6. 일반 LLM – 위험 분석
# --------------------------------------------------------------------------
def analyze_risk_with_general_llm(clause: str):
    prompt = f"""
다음 계약서 조항에서 근로자에게 불리하거나
주의해야 할 위험 요소를 2~3개 요약하고,
각 항목마다 왜 위험한지도 설명하세요.

{clause}
"""
    res = general_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content


# --------------------------------------------------------------------------
# 7. Streamlit UI (너가 만든 고급 UI 그대로 살림)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="계약서 이해 AI",
    page_icon="📄",
    layout="wide",
)

# (생략) — 🎨 UI CSS 그대로 복붙 — 너무 길어서 여기엔 생략
# 👉 내가 위에서 본 lawchatapp.py CSS 전체를 그대로 넣어줄 테니까 걱정하지 마!

# --------------------------------------------------------------------------
# 8. 채팅 흐름 관리
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "type": "intro",
            "content": (
                "안녕하세요, 계약서 이해 AI입니다.\n"
                "궁금한 계약서 조항이나 키워드를 입력해주세요.\n"
                '예: "근로시간 조항 설명해줘"'
            ),
        }
    ]

docs, vectors = load_docs_and_vectors()
vectorstore = create_vectorstore(docs, vectors)
rag_chain = initialize_rag_chain(vectorstore)


# --------------------------------------------------------------------------
# 9. 채팅 입력
# --------------------------------------------------------------------------
user_query = st.chat_input("메시지를 입력하세요")

if user_query:
    st.session_state.messages.append(
        {"role": "human", "type": "user", "content": user_query}
    )

    rag_response = rag_chain.invoke({"input": user_query})
    raw_clause = rag_response["answer"]

    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause)

    easy = explain_with_finetuned_model(clause)
    risk = analyze_risk_with_general_llm(clause)

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
# 10. 렌더링 (UI 전체 유지)
# --------------------------------------------------------------------------
# 👉 너가 만든 UI 그대로 여기 붙여줄게 (지금 텍스트가 너무 길어져서 생략했지만 계속 이어서 완성 가능)


