import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from data_loader import embed_texts
from vector_db import QdrantStorage
from openai import OpenAI
import json
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_rag_query(question: str, top_k: int = 5) -> dict:
    query_vector = embed_texts([question])[0]
    storage = QdrantStorage()
    search_results = storage.search(query_vector, top_k)
    
    contexts = search_results["contexts"]
    
    context_block = "\n\n".join(f"- {ctx}" for ctx in contexts)
    user_content = (
        "Use the following context to answer the question:\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer the question based on the context provided."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides answers based on provided context."},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    
    answer = response.choices[0].message.content
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": search_results["sources"]
    }

print("RAG System Evaluation with RAGAS")
print("="*80 + "\n")

test_questions = [
    "What is the main topic of the document?",
    "What methodology is described?",
    "What are the key findings?",
    "What problem does this address?",
    "What are the main components?"
]

print("Running RAG queries...")
results = []

for i, question in enumerate(test_questions, 1):
    print(f"{i}/{len(test_questions)}: {question[:50]}...")
    try:
        result = run_rag_query(question)
        results.append(result)
        print(f"   Answer: {result['answer'][:80]}...")
    except Exception as e:
        print(f"   Error: {e}")

if not results:
    print("\nNo results. Make sure documents are indexed in Qdrant.")
    exit(1)

print(f"\nProcessed {len(results)} queries")

print("\nEvaluating with RAGAS...")

ground_truths = [
    "The document discusses main concepts and methodology",
    "The document describes research methodology",
    "The document presents findings and conclusions",
    "The document addresses a specific problem",
    "The document contains key sections and components"
]

data = {
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "contexts": [r["contexts"] for r in results],
    "ground_truth": ground_truths[:len(results)]
}

dataset = Dataset.from_dict(data)

ragas_llm = ChatOpenAI(model="gpt-4o-mini")
ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\n" + "="*80)
print("RESULTS")
print("="*80 + "\n")

# RAGAS returns lists, get the mean
faithfulness_list = result['faithfulness']
relevancy_list = result['answer_relevancy']

if isinstance(faithfulness_list, list):
    faithfulness_score = sum(faithfulness_list) / len(faithfulness_list)
else:
    faithfulness_score = float(faithfulness_list)

if isinstance(relevancy_list, list):
    relevancy_score = sum(relevancy_list) / len(relevancy_list)
else:
    relevancy_score = float(relevancy_list)

avg_score = (faithfulness_score + relevancy_score) / 2

print(f"Faithfulness:     {faithfulness_score:.4f} ({faithfulness_score*100:.1f}%)")
print(f"Answer Relevancy: {relevancy_score:.4f} ({relevancy_score*100:.1f}%)")
print(f"Average:          {avg_score:.4f} ({avg_score*100:.1f}%)")

print("\n" + "="*80)

output = {
    "timestamp": datetime.now().isoformat(),
    "scores": {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "average": avg_score
    },
    "queries": results
}

filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to: {filename}")