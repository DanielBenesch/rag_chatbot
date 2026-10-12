# Voltrad Assistant: an agentic RAG chatbot

A chatbot for Voltrad GmbH, a fictional German e-bike manufacturer. It answers questions about the company's products, prices, history, locations, warranty and service from a single PDF ([`content/company.pdf`](content/company.pdf)).

The agent decides on its own when to search the document. It handles small talk without a search, says so when the document doesn't cover a question, and replies in the language of the question. It also remembers earlier turns within a browser session. MLflow traces every agent run, and a small evaluation scores the answers with LLM judges for correctness and groundedness.

**Stack:** Python 3.14 with uv, a LangChain v1 `create_agent` agent with one retriever tool, OpenAI chat and embedding models, a Chroma vector store server, a Streamlit chat UI, MLflow 3 for tracing and evaluation, and Docker Compose.

## Quick start

You need Docker Engine with the Compose plugin and an OpenAI API key.

```bash
cp .env.example .env                # then set OPENAI_API_KEY in .env
docker compose up -d --build        # starts chroma, mlflow and the chat app
docker compose run --rm app ingest  # loads the PDF into Chroma
```

- Chat: http://localhost:8501
- MLflow traces: http://localhost:5000, experiment `rag-chatbot`

Ingest rebuilds the collection from scratch on every run, so you can safely run it again. Chroma and MLflow keep their data in Docker volumes, so it survives restarts. `docker compose down -v` deletes it.

## Evaluation

```bash
docker compose run --rm app evaluate
```

This runs the 13 German and English questions in [`eval/dataset.jsonl`](eval/dataset.jsonl) through the agent. The questions cover facts, details, out-of-scope questions, small talk, and one request that OpenAI's moderation endpoint flags. MLflow's `Correctness` and `RetrievalGroundedness` judges score the answers, and the results show up under *Evaluation runs* in MLflow. The link the command prints uses the internal host `mlflow:5000`, so open it on `localhost:5000` instead. Groundedness is only scored for answers that searched the document, so the small-talk, out-of-scope, and moderation cases log a "No retrieval context found" scorer error. That error is expected.

## Configuration

All settings come from environment variables in `.env`. See [`.env.example`](.env.example) for the full list: the API key, chat and embedding models, the Chroma host and port, the MLflow tracking URI, and `MLFLOW_ENABLED`. Inside compose, the Chroma host and the MLflow tracking URI are set to the service names automatically. Set `MLFLOW_ENABLED=false` to run without tracing.

The PDF is the file that gets ingested. [`content/company.md`](content/company.md) is the same text, kept in git so the document is readable without opening the PDF.

## Deploy on EC2

Launch an Ubuntu Server 24.04 instance with at least 4 GB of RAM, 20 GB of storage, and a key pair. Allow inbound TCP 22, 8501, and 5000 from `0.0.0.0/0` (source type **Anywhere**). Auto-assign a public IP. Terminate the instance when you are done. Stopping it keeps the disk and continues to cost money.

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<public-ip>
sudo apt update
sudo apt install -y git ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
git clone https://github.com/DanielBenesch/rag_chatbot.git
cd rag_chatbot
```

`.env` is not in the repo. Copy it from your machine:

```bash
scp -i your-key.pem .env ubuntu@<public-ip>:~/rag_chatbot/.env
```

Before the first start, set the public IP in `.env` so the MLflow UI accepts browser requests. Keep `mlflow:*`: the app sends traces to that hostname inside Compose.

```bash
MLFLOW_ALLOWED_HOSTS=localhost:*,127.0.0.1:*,mlflow:*,<public-ip>:5000
MLFLOW_CORS_ALLOWED_ORIGINS=http://<public-ip>:5000
```

```bash
cd ~/rag_chatbot
sudo docker compose up -d --build
sudo docker compose run --rm app ingest
sudo docker compose run --rm app evaluate
```

- Chat: `http://<public-ip>:8501`
- MLflow: `http://<public-ip>:5000`, experiment `rag-chatbot`
