from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chatbot.settings import COMPANY_PDF
from rag_chatbot.vectorstore import get_vector_store

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def ingest() -> int:
    pages = PyPDFLoader(str(COMPANY_PDF)).load()
    for page in pages:
        page.metadata["source"] = COMPANY_PDF.name
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)

    store = get_vector_store()
    store.reset_collection()
    store.add_documents(chunks)
    return len(chunks)


def main() -> None:
    count = ingest()
    print(f"Ingested {COMPANY_PDF.name}: collection now holds {count} chunks.")


if __name__ == "__main__":
    main()
