from dataclasses import dataclass
from uuid import uuid4

import streamlit as st

from rag_chatbot.agent import ask

TITLE = "Voltrad Assistant"
CAPTION = "Ask me about Voltrad GmbH and its e-bikes. Answers are based on the official company documents."


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def render_message(message: ChatMessage) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content)


def main() -> None:
    st.set_page_config(page_title=TITLE)
    st.title(TITLE)
    st.caption(CAPTION)

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
    messages: list[ChatMessage] = st.session_state.messages

    for message in messages:
        render_message(message)

    prompt = st.chat_input("Ask a question about Voltrad")
    if prompt:
        user_message = ChatMessage(role="user", content=prompt)
        render_message(user_message)
        messages.append(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Answering…"):
                text = ask(prompt, st.session_state.thread_id)
            st.markdown(text)
        messages.append(ChatMessage(role="assistant", content=text))


if __name__ == "__main__":
    main()
