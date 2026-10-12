FROM ghcr.io/astral-sh/uv:0.12.24-python3.14-trixie-slim

WORKDIR /app

COPY . .
RUN uv sync --locked

ENV GIT_PYTHON_REFRESH=quiet \
    PATH="/app/.venv/bin:$PATH"

CMD ["streamlit", "run", "src/rag_chatbot/app.py"]
