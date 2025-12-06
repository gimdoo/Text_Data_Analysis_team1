import streamlit as st
import requests
import json
import re
import numpy as np
import faiss
import zipfile
import io

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# 0. API Key 설정 (Streamlit Secrets)
# ---------------------------------------------------------
st.set_page_config(page_title="계약서 이해 AI", layout="wide")

GENERAL_API_KEY = st.secrets["GENERAL_API_KEY"]
FINETUNE_API_KEY = st.secrets["FINETUNE_API_KEY"]

openai_client = OpenAI(api_key=GENERAL_API_KEY)
finetune_client = OpenAI(api_key=FINETUNE_API_KEY)

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"


# ---------------------------------------------------------
# 1. GitHub Release 파일 URL
# ---------------------------------------------------------
DOC_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_documents.json"
EMB_ZIP_URL = "https://github.com/gimdoo/Text_Data_Analysis_team1/releases/download/v1.1/_embeddings.zip"


# ---------------------------------------------------------
# 2. 문서 로드(JSON)
# ---------------------------------------------------------
@st.cache_resource
def load_documents():
    res = requests.get(DOC_URL)
    res.raise_for_status()
    return json.loads(res.text)

docs = load_documents()


# ---------------------------------------------------------
# 3. 임베딩 로드(.zip → .npz)
# ---------------------------------------------------------
@st.cache_resource
def load_embeddings():
    res = requests.get(EMB_ZIP_URL)
    res.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(res.content))
    npz = np.load(z.open("_embeddings.npz"))
    return npz["arr_0"]

vectors = load_embeddings()


# ---------------------------------------------------------
# 4. FAISS VectorStore 생성
# ---------------------------------------------------------
@st.cache_resource
def create_vectorstore(docs, vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))

    wrapped_docs = [Document(page_content=d) for d in docs]
    doc_dict = {str(i): wrapped_docs[i] for i in range(len(docs))}
    docstore = InMemoryDocstore(doc_dict)
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=GENERAL_API_KEY,
    )

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

vectorstore = create_vectorstore(docs, vectors)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------------------------------------------------
# 5. RAG 검색 (최신 LangChain 방식)
# ---------------------------------------------------------
def search_clause(query):
    results = retriever.invoke(query)   # 🔥 최신 방식
    return "\n\n".join([d.page_content for d in results])


# ---------------------------------------------------------
# 6. 파인튜닝 모델: 쉬운 설명
# ---------------------------------------------------------
def explain_easy(clause: str):
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL,
            messages=[
                {"role": "system", "content": "계약서를 쉽고 짧게 설명하는 도우미입니다."},
                {"role": "user", "content": clause},
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content

    except Exception as e:
        return f"⚠️ 파인튜닝 모델 오류: {e}"


# ---------------------------------------------------------
# 7. GPT-4o: 위험 분석
# ---------------------------------------------------------
def analyze_risk(clause: str):
    prompt = f"""
다음 계약서 조항에서 근로자에게 불리하거나 주의해야 할 위험 요소를 2~3개 요약하세요.
각 항목이 왜 위험한지도 설명하세요.

{clause}
"""
    res = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content


# ---------------------------------------------------------
# 8. Streamlit UI
# ---------------------------------------------------------
st.title("📄 계약서 이해 AI")
st.write("계약서 조항을 검색하고, 쉬운 설명 + 위험 요소 분석을 제공합니다.")

user_query = st.text_input("궁금한 계약서 조항을 입력하세요:")

if user_query:
    st.subheader("🔍 RAG 검색 결과")
    raw_clause = search_clause(user_query)

    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause)
    st.write(clause)

    st.subheader("✨ 쉬운 설명 (파인튜닝 모델)")
    st.write(explain_easy(clause))

    st.subheader("⚠️ 위험 요소 분석 (gpt-4o)")
    st.write(analyze_risk(clause))
