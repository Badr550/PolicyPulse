!pip install -qU langchain langchain-groq langchain-community langchain-text-splitters langchain-huggingface chromadb pypdf sentence-transformers gradio


%%writefile app.py

import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser




GROQ_API_KEY = ""




def process_data(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = i

        if "page" not in chunk.metadata:
            chunk.metadata["page"] = "Unknown"

    return chunks




def build_vectorstore(_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    return vectorstore




def ask_gdpr_question(pdf_file, query):

    if pdf_file is None:

        return (
            "Please upload a GDPR PDF file.",
            "No verification available.",
            "No retrieved chunks."
        )

    pdf_path = pdf_file.name



    chunks = process_data(pdf_path)

    vectorstore = build_vectorstore(chunks)



    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )



    docs_and_scores = vectorstore.similarity_search_with_score(
        query,
        k=5
    )

    processed_results = []

    seen_chunks = set()


    for doc, distance in docs_and_scores:

        chunk_id = doc.metadata.get("chunk_id")

    # منع التكرار
        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(chunk_id)

        cosine_similarity = 1 - distance

        processed_results.append(
            (doc, cosine_similarity)
        )

    processed_results.sort(
        key=lambda x: x[1],
        reverse=True
    )



    SIMILARITY_THRESHOLD = 0.5

    filtered_results = [
        (doc, score)
        for doc, score in processed_results
        if score >= SIMILARITY_THRESHOLD
    ]

    if not filtered_results:

        return (
            "⚠️ No strongly relevant GDPR text found.",
            "WARNING",
            "No relevant chunks retrieved."
        )



    retrieved_docs = [
        doc for doc, score in filtered_results
    ]

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )



    template = """
You are a GDPR legal assistant.

Answer ONLY using the provided context.

Rules:
- Cite article numbers clearly.
- Only cite article numbers explicitly mentioned in the context.
- Do NOT invent citations.
- If information is missing, say:
  "Not found in retrieved GDPR text."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )


    answer = rag_chain.invoke({
        "context": context,
        "question": query
    })



    fact_template = """
You are a strict GDPR fact checker.

Check whether the answer is FULLY supported by the context.

Context:
{context}

Answer:
{answer}

If fully supported reply exactly:
VERIFIED

Otherwise reply exactly:
WARNING
"""

    fact_prompt = PromptTemplate.from_template(
        fact_template
    )

    fact_chain = (
        fact_prompt
        | llm
        | StrOutputParser()
    )

    verification = fact_chain.invoke({
        "context": context,
        "answer": answer
    })


    chunks_text = ""

    for i, (doc, score) in enumerate(filtered_results):

        chunk_id = doc.metadata.get(
            "chunk_id",
            "Unknown"
        )

        page_num = doc.metadata.get(
            "page",
            "Unknown"
        )

        chunks_text += f"""
==================================================
Chunk {i+1}

Chunk ID: {chunk_id}

Page Number: {page_num}

Cosine Similarity: {round(score, 4)}

Retriever selected this chunk because its embedding
was semantically similar to the user query.

Results are ranked from highest to lowest cosine similarity.

Chunk Content:
{doc.page_content}

==================================================

"""

    if "VERIFIED" in verification:

        verification_message = (
            "✅ VERIFIED\n\n"
            "The answer is fully supported by retrieved GDPR text."
        )

    else:

        verification_message = (
            "❌ WARNING\n\n"
            "Possible hallucinated information detected."
        )

    return (
        answer,
        verification_message,
        chunks_text
    )




custom_css = """
body {
    background-color: #0f172a;
}

.gradio-container {
    font-family: Arial;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 20px;
}

textarea {
    font-size: 15px !important;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.HTML("""
    <div class='main-title'>
        ⚖️ PolicyPulse GDPR Assistant
    </div>

    <div class='sub-title'>
        RAG with Cosine Similarity + Fact Checking
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="📂 Upload GDPR PDF",
                file_types=[".pdf"]
            )

            query_input = gr.Textbox(
                label="❓ Ask Question",
                lines=4,
                placeholder="Example: What are the lawful bases for processing personal data?"
            )

            submit_btn = gr.Button(
                "Generate Answer"
            )

        with gr.Column(scale=2):

            answer_output = gr.Textbox(
                label="💡 Final Answer",
                lines=10
            )

            verification_output = gr.Textbox(
                label="🔎 Verification",
                lines=4
            )

    chunks_output = gr.Textbox(
        label="📄 Retrieved Chunks",
        lines=25
    )

    submit_btn.click(
        fn=ask_gdpr_question,
        inputs=[
            pdf_input,
            query_input
        ],
        outputs=[
            answer_output,
            verification_output,
            chunks_output
        ]
    )

demo.launch(share=True)



!python app.py
