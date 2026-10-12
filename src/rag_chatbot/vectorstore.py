from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag_chatbot.settings import get_settings

COLLECTION = "company_docs"


def get_vector_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        ),
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
