import json
import os
import sys
from typing import Any
from uuid import uuid4

import mlflow
from mlflow.genai.scorers import Correctness, RetrievalGroundedness

from rag_chatbot.agent import ask, enable_tracing
from rag_chatbot.settings import EVAL_DATASET, get_settings

JUDGE_MODEL = "openai:/gpt-6-luna"


def load_dataset() -> list[dict[str, Any]]:
    with EVAL_DATASET.open(encoding="utf-8") as file:
        cases = [json.loads(line) for line in file if line.strip()]
    return [
        {
            "inputs": {"question": case["question"]},
            "expectations": {"expected_response": case["expected_answer"]},
        }
        for case in cases
    ]


def predict(question: str) -> str:
    return ask(question, thread_id=str(uuid4()))


def main() -> None:
    settings = get_settings()
    if not settings.mlflow_enabled:
        sys.exit("The evaluation logs its results to MLflow. Set MLFLOW_ENABLED=true and retry.")
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
    enable_tracing()
    mlflow.genai.evaluate(
        data=load_dataset(),
        predict_fn=predict,
        scorers=[Correctness(model=JUDGE_MODEL), RetrievalGroundedness(model=JUDGE_MODEL)],
    )


if __name__ == "__main__":
    main()
