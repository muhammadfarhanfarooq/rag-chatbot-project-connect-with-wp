import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
print("DEBUG KEY VALUE:", openai_api_key[:10] if openai_api_key else "NOT FOUND")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing")


# ---------- STEP 1: LOAD OR CREATE VECTOR DB ----------



# ---------- os.path.abspath(__file__) 👉 Converts it to full path (absolute path) ----------
# ---------- os.path.dirname(...) 👉 Removes the file name → gives folder ----------
# ---------- BASE_DIR becomes your project folder ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 👉 Joins path safely: so if project path is this then it will join like this: /Users/muhammadfarhan/project + chroma_db ----------
# ---------- 👉 Final result: /Users/muhammadfarhan/project/chroma_db ----------
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

import glob
import shutil

# Always rebuild to pick up new docs
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

print("Creating vector database...")

documents = []

for file in glob.glob("docs/*.txt"):
    loader = TextLoader(file)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = file
    documents.extend(docs)

print("Total documents loaded:", len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
vector_store.persist()
   


 

# ---------- STEP 2: CREATE RETRIEVER + LLM ----------

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)


prompt_template = """
You are a helpful assistant. Use the context below to answer the question.
If the context contains relevant information, use it to give a helpful answer.
Only say you don't know if the context has absolutely nothing related.

Context:
{summaries}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["summaries", "question"]
)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    chain_type="stuff"
)

# ---------- STEP 3: MAIN FUNCTION ----------

def ask_pdf(question):
    try:
        refined_question = f"Answer based on school information: {question}"
        result = qa_chain.invoke({"question": refined_question})

        answer = result["answer"]
        sources= result.get("sources","")
        print("DEBUG ANSWER:", answer)      
        print("DEBUG SOURCES:", sources)

        # Escalate only if no real answer was found
        if not answer or "i don't know" in answer.lower():
            return "ESCALATE"

        return answer

    except Exception as e:
        print("Error:", e)
        return "Something went wrong"


if __name__ == "__main__":
    print(ask_pdf("What is number system?"))


