import streamlit as st
import pdfplumber
import os
import pandas as pd

from sentence_transformers import SentenceTransformer

from preprocessing.document_preprocessing import (
    clean_legal_text,
    chunk_text_with_pages
)

from src.tfidf_search import (
    LegalTFIDFSearch
)

from src.semantic_search import (
    LegalSemanticSearch
)

from src.ipc_search import (
    IPCSearch
)

from src.rag_generator import (
    LegalRAGGenerator
)


# Cached models

@st.cache_resource
def load_semantic_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_data
def create_embeddings(texts):

    model = load_semantic_model()

    return model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=True
    )


@st.cache_data
def load_ipc_dataset():

    return pd.read_csv(
        "data/ipc_sections.csv"
    )


@st.cache_resource
def load_rag():

    return LegalRAGGenerator()


# Page configuration

st.set_page_config(
    page_title="Indian Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)


# Title

st.title(
    "⚖️ Indian Legal Document Analyzer"
)

st.write(
    "Search Indian legal provisions and "
    "analyze uploaded legal documents using "
    "NLP, semantic search and RAG."
)


# Load model and dataset

semantic_model = load_semantic_model()

ipc_data = load_ipc_dataset()


# Search mode

st.header(
    "🔎 Choose Search Mode"
)

search_mode = st.radio(
    "Select an option:",
    [
        "🇮🇳 Indian Law Search",
        "📄 Legal Document Q&A"
    ],
    horizontal=True
)


# Indian Law Search

if search_mode == "🇮🇳 Indian Law Search":

    st.header(
        "🇮🇳 Indian Law Search"
    )

    st.write(
        "Search IPC sections using a section number, "
        "offense, punishment or natural-language description."
    )

    st.write(
        f"📚 Total IPC records: "
        f"**{len(ipc_data)}**"
    )


    with st.expander(
        "View IPC Dataset"
    ):

        st.dataframe(
            ipc_data,
            use_container_width=True
        )


    # Create searchable IPC text

    ipc_search_texts = []

    for _, row in ipc_data.iterrows():

        searchable_text = (
            str(row["Section"])
            + " "
            + str(row["Offense"])
            + " "
            + str(row["Punishment"])
            + " "
            + str(row["Description"])
        )

        ipc_search_texts.append(
            searchable_text
        )


    ipc_search_texts = tuple(
        ipc_search_texts
    )


    # Create IPC embeddings

    with st.spinner(
        "Preparing Indian law search..."
    ):

        ipc_embeddings = create_embeddings(
            ipc_search_texts
        )


    question = st.text_input(
        "Enter your legal question:",
        placeholder=(
            "Example: What is the punishment "
            "for wearing a military uniform?"
        ),
        key="ipc_question"
    )


    if question.strip():

        # IPC semantic search

        ipc_search = IPCSearch(
            ipc_data,
            semantic_model,
            ipc_embeddings
        )


        semantic_results = ipc_search.search(
            question,
            top_k=5
        )


        # Semantic results

        st.subheader(
            "🧠 Semantic Search Results"
        )


        if semantic_results:

            for i, result in enumerate(
                semantic_results,
                start=1
            ):

                with st.expander(
                    f"Result {i} — "
                    f"{result['Section']}"
                ):

                    st.write(
                        "📑 **Section:**",
                        result["Section"]
                    )

                    st.write(
                        "⚖️ **Offense:**",
                        result["Offense"]
                    )

                    st.write(
                        "🔨 **Punishment:**",
                        result["Punishment"]
                    )

                    st.write(
                        "📖 **Description:**",
                        result["Description"]
                    )

                    st.write(
                        "📊 **Similarity:**",
                        round(
                            result["similarity"],
                            4
                        )
                    )

        else:

            st.warning(
                "No relevant IPC section found."
            )


        # TF-IDF demonstration

        st.subheader(
            "📊 TF-IDF Demonstration"
        )

        st.caption(
            "TF-IDF is shown for comparison only. "
            "It is not used for AI answer generation."
        )


        tfidf_records = []


        for _, row in ipc_data.iterrows():

            tfidf_records.append({

                "text": (
                    str(row["Section"])
                    + " "
                    + str(row["Offense"])
                    + " "
                    + str(row["Punishment"])
                    + " "
                    + str(row["Description"])
                ),

                "Section":
                    row["Section"],

                "Offense":
                    row["Offense"],

                "Punishment":
                    row["Punishment"],

                "Description":
                    row["Description"]

            })


        tfidf_engine = LegalTFIDFSearch(
            tfidf_records
        )


        tfidf_results = (
            tfidf_engine.search(
                question,
                top_k=3
            )
        )


        for i, result in enumerate(
            tfidf_results,
            start=1
        ):

            st.write(
                f"**{i}.** "
                f"{result.get('Section', 'N/A')} "
                f"— Similarity: "
                f"{round(result['similarity'], 4)}"
            )


        # AI answer

        st.subheader(
            "🤖 AI Legal Answer"
        )


        rag_chunks = []


        for result in semantic_results:

            rag_chunks.append({

                "chunk_id":
                    result["Section"],

                "document":
                    "IPC Dataset",

                "page":
                    result["Section"],

                "text":
                    (
                        f"Section: "
                        f"{result['Section']}\n\n"

                        f"Offense: "
                        f"{result['Offense']}\n\n"

                        f"Punishment: "
                        f"{result['Punishment']}\n\n"

                        f"Description: "
                        f"{result['Description']}"
                    )

            })


        with st.spinner(
            "Generating legal answer..."
        ):

            rag = load_rag()

            answer = rag.generate_answer(
                question,
                rag_chunks
            )


        st.write(
            answer
        )


# Legal Document Q&A

else:

    st.header(
        "📄 Upload Legal Documents"
    )


    uploaded_files = st.file_uploader(
        "Choose legal PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )


    all_chunks = []

    document_texts = {}


    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} "
            f"document(s) uploaded successfully!"
        )


        os.makedirs(
            "documents",
            exist_ok=True
        )


        for uploaded_pdf in uploaded_files:

            st.markdown(
                f"### 📄 {uploaded_pdf.name}"
            )


            pdf_path = os.path.join(
                "documents",
                uploaded_pdf.name
            )


            with open(
                pdf_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_pdf.getbuffer()
                )


            full_text = ""

            page_texts = []


            with pdfplumber.open(
                pdf_path
            ) as pdf:

                total_pages = len(
                    pdf.pages
                )


                st.info(
                    f"{uploaded_pdf.name} — "
                    f"{total_pages} page(s)"
                )


                for page_number, page in enumerate(
                    pdf.pages,
                    start=1
                ):

                    text = page.extract_text()


                    if text:

                        full_text += (
                            f"\n\n"
                            f"--- PAGE "
                            f"{page_number} ---"
                            f"\n\n"
                        )

                        full_text += text


                        page_texts.append(
                            (
                                page_number,
                                text
                            )
                        )


            if not full_text.strip():

                st.warning(
                    f"No selectable text could be "
                    f"extracted from "
                    f"{uploaded_pdf.name}."
                )

                continue


            document_texts[
                uploaded_pdf.name
            ] = full_text


            cleaned_page_texts = []


            for page_number, page_text in page_texts:

                cleaned_page_text = (
                    clean_legal_text(
                        page_text
                    )
                )


                if cleaned_page_text.strip():

                    cleaned_page_texts.append(
                        (
                            page_number,
                            cleaned_page_text
                        )
                    )


            document_chunks = (
                chunk_text_with_pages(
                    cleaned_page_texts,
                    chunk_size=300,
                    overlap=50
                )
            )


            for chunk in document_chunks:

                chunk["document"] = (
                    uploaded_pdf.name
                )

                all_chunks.append(
                    chunk
                )


            with st.expander(
                f"View {uploaded_pdf.name}"
            ):

                st.write(
                    f"Original characters: "
                    f"{len(full_text)}"
                )


                cleaned_full_text = (
                    clean_legal_text(
                        full_text
                    )
                )


                st.write(
                    f"Cleaned characters: "
                    f"{len(cleaned_full_text)}"
                )


                st.write(
                    f"Number of chunks: "
                    f"{len(document_chunks)}"
                )


    if uploaded_files and not all_chunks:

        st.error(
            "No searchable text was found "
            "in the uploaded documents."
        )

        st.stop()


    if all_chunks:

        st.subheader(
            "📚 Uploaded Documents"
        )


        for document_name in document_texts:

            st.write(
                f"📄 {document_name}"
            )


        st.write(
            f"Total searchable chunks: "
            f"**{len(all_chunks)}**"
        )


        with st.expander(
            "View all document chunks"
        ):

            for chunk in all_chunks:

                st.markdown(
                    f"### Chunk "
                    f"{chunk['chunk_id'] + 1}"
                )


                st.write(
                    f"📄 **Document:** "
                    f"{chunk['document']}"
                )


                st.write(
                    f"📑 **Page:** "
                    f"{chunk['page']}"
                )


                st.write(
                    chunk["text"]
                )


        st.subheader(
            "🔎 Ask a Question"
        )


        question = st.text_input(
            "Enter your question:",
            placeholder=(
                "What safeguards were laid down "
                "regarding arrest and detention?"
            ),
            key="document_question"
        )


        if question.strip():

            # TF-IDF demonstration

            tfidf_engine = LegalTFIDFSearch(
                all_chunks
            )


            tfidf_results = (
                tfidf_engine.search(
                    question,
                    top_k=3
                )
            )


            # Semantic search

            with st.spinner(
                "Searching legal documents..."
            ):

                chunk_texts = tuple(
                    chunk["text"]
                    for chunk in all_chunks
                )


                chunk_embeddings = (
                    create_embeddings(
                        chunk_texts
                    )
                )


                semantic_engine = (
                    LegalSemanticSearch(
                        all_chunks,
                        model=semantic_model,
                        embeddings=chunk_embeddings
                    )
                )


                semantic_results = (
                    semantic_engine.search(
                        question,
                        top_k=5
                    )
                )


            # Semantic results

            st.subheader(
                "🔍 Semantic Search Results"
            )


            if semantic_results:

                for i, result in enumerate(
                    semantic_results,
                    start=1
                ):

                    st.markdown(
                        f"### Result {i}"
                    )


                    st.write(
                        "📑 Page:",
                        result["page"]
                    )


                    st.write(
                        "📄 Document:",
                        result["document"]
                    )


                    st.write(
                        "📊 Similarity:",
                        round(
                            result["similarity"],
                            4
                        )
                    )


                    st.text_area(
                        f"Retrieved text - Result {i}",
                        result["text"],
                        height=200,
                        key=f"document_result_{i}"
                    )


            # AI answer

            st.subheader(
                "🤖 AI Legal Answer"
            )


            with st.spinner(
                "Generating answer..."
            ):

                rag = load_rag()


                answer = (
                    rag.generate_answer(
                        question,
                        semantic_results
                    )
                )


            st.write(
                answer
            )


            # Source

            if semantic_results:

                st.subheader(
                    "📚 Source"
                )


                source_pages = []


                for result in semantic_results:

                    page = result["page"]


                    if page not in source_pages:

                        source_pages.append(
                            page
                        )


                documents = []


                for result in semantic_results:

                    document = result[
                        "document"
                    ]


                    if document not in documents:

                        documents.append(
                            document
                        )


                st.write(
                    "📄 **Document:** "
                    + ", ".join(documents)
                )


                st.write(
                    "📑 **Pages:** "
                    + ", ".join(
                        str(page)
                        for page in source_pages
                    )
                )


# Footer

st.divider()

st.caption(
    "Educational NLP project — not a substitute "
    "for professional legal advice."
)