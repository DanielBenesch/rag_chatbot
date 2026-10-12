from functools import cache

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.middleware import OpenAIModerationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI

from rag_chatbot.settings import get_settings
from rag_chatbot.vectorstore import get_vector_store

TOP_K = 4
MLFLOW_EXPERIMENT = "rag-chatbot"

SYSTEM_PROMPT = """\
You are the friendly assistant of Voltrad GmbH, a German e-bike manufacturer. \
You answer questions about the company based on its official company documents.

Rules:
- Small talk (greetings, thanks, "what can you do?") needs no search. Answer briefly and \
mention that you can answer questions about Voltrad, e.g. its products, prices, history, \
locations, warranty, and service.
- For every question about Voltrad or its products, call `search_company_docs` before answering. \
The company documents are written in German, so phrase your search queries in German. \
For follow-up questions, write a self-contained query that includes the context from earlier turns.
- State facts about Voltrad only if they appear in the search results. Never use outside \
knowledge, never guess, and never fill gaps with plausible-sounding details.
- Search results are fragments that may start in the middle of a section. Attribute a fact to \
a product or topic only if the fragment clearly names it. If it doesn't, search again with a \
more specific query instead of guessing what the fact belongs to.
- If the search results don't answer the question, or the question is neither small talk nor \
about Voltrad, say clearly that you have no information about it in the Voltrad company \
documents. Don't speculate about the answer.
- Always reply in the language of the user's latest message, regardless of the language \
of the documents. Translate facts from the documents where needed.
- Keep answers concise. Don't list page numbers or quote the retrieved passages.
"""


@cache
def _get_retriever() -> BaseRetriever:
    return get_vector_store().as_retriever(search_kwargs={"k": TOP_K})


@tool(response_format="content_and_artifact")
def search_company_docs(query: str) -> tuple[str, list[Document]]:
    """Search the Voltrad company documents (written in German) for passages relevant to the query."""
    docs = _get_retriever().invoke(query)
    if not docs:
        return "No matching passages found.", docs
    content = "\n\n".join(f"[Page {_page_number(doc)}]\n{doc.page_content}" for doc in docs)
    return content, docs


def _page_number(doc: Document) -> int:
    return int(doc.metadata.get("page", 0)) + 1


@cache
def enable_tracing() -> None:
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlflow.langchain.autolog()


@cache
def build_agent() -> CompiledStateGraph:
    settings = get_settings()
    if settings.mlflow_enabled:
        enable_tracing()
    model = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key)
    moderation = OpenAIModerationMiddleware(
        client=OpenAI(api_key=settings.openai_api_key.get_secret_value()),
    )
    return create_agent(
        model,
        tools=[search_company_docs],
        system_prompt=SYSTEM_PROMPT,
        middleware=[moderation],
        checkpointer=InMemorySaver(),
    )


def ask(question: str, thread_id: str) -> str:
    state = build_agent().invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread_id}},
    )
    return state["messages"][-1].text
