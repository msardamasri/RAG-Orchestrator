"""Optional judge-model metrics with RAGAS (extra: ``ragas``): faithfulness and answer relevancy.

The built-in evaluation needs no judge. Use this when you also want a model's opinion on whether each answer
is supported by its sources. It calls OpenAI (``OPENAI_API_KEY``), as the original project's evaluation did.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragcore.eval.dataset import Question
from ragcore.pipeline.rag import Rag


def ragas_scores(rag: Rag, questions: Sequence[Question], judge_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-large") -> dict[str, float]:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    rows = {"question": [], "answer": [], "contexts": []}
    for question in questions:
        if not question.answerable:
            continue
        answer = rag.ask(question.question)
        if answer.abstained:
            continue
        rows["question"].append(question.question)
        rows["answer"].append(answer.text)
        rows["contexts"].append([hit.chunk.text for hit in answer.hits])
    if not rows["question"]:
        return {}
    result = evaluate(Dataset.from_dict(rows), metrics=[faithfulness, answer_relevancy],
                      llm=ChatOpenAI(model=judge_model), embeddings=OpenAIEmbeddings(model=embedding_model))

    def mean(values) -> float:
        values = values if isinstance(values, list) else [values]
        return float(sum(values) / len(values))

    return {"faithfulness": mean(result["faithfulness"]), "answer_relevancy": mean(result["answer_relevancy"]),
            "answers_judged": len(rows["question"])}
