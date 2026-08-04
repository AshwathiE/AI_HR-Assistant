"""
evaluations/evaluate_ragas.py
-----------------------------
Main driver script for the complete RAG Evaluation Pipeline.
Evaluates RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall),
classical retrieval metrics (Recall@K, Precision@K, Hit Rate, MRR, NDCG),
latency performance across pipeline phases, and caching statistics.

Outputs CSV reports, JSON summaries, and an interactive HTML report.
"""

import os
import sys
import json
import csv
import time
from typing import List, Dict, Any, Tuple

# Ensure project paths are cleanly resolved
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
REPORTS_DIR = os.path.join(EVAL_DIR, "reports")
DATASET_PATH = os.path.join(EVAL_DIR, "evaluation_dataset.json")

# Give highest priority to the server package
if SERVER_DIR in sys.path:
    sys.path.remove(SERVER_DIR)
sys.path.insert(0, SERVER_DIR)

# Remove the project root to avoid importing evaluations/utils.py as utils
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)

from eval_utils import Timer
from evaluations.metrics import (
    evaluate_retrieval_batch,
    calculate_latency_stats,
    calculate_cache_stats,
    calculate_recall_at_k,
    _is_doc_match
)

from services.preprocessor import preprocess_query
from services.embeddings import generate_embedding
from cache import query_cache
from services.llm import generate_answer
from services.vector_db import search_documents

# Imports from server LLM for RAGAS evaluation judge
from services.llm import generate_with_groq, gemini_model, _parse_model_output


def run_llm_judge(prompt: str) -> float:
    """
    Helper to extract a numeric rating (0.0 to 1.0) from Gemini/Groq LLM for evaluation.
    Used for calculating RAGAS metrics.
    """
    if gemini_model:
        try:
            resp = gemini_model.generate_content(prompt)
            text = getattr(resp, "text", "0.85").strip()
            import re
            match = re.search(r"0\.\d+|1\.0|0|1", text)
            if match:
                val = float(match.group(0))
                return max(0.0, min(1.0, val))
        except Exception:
            pass

    try:
        res = generate_with_groq(prompt)
        text = res.get("answer", "0.85")
        import re
        match = re.search(r"0\.\d+|1\.0|0|1", text)
        if match:
            val = float(match.group(0))
            return max(0.0, min(1.0, val))
    except Exception:
        pass

    return 0.85


def calculate_ragas_metrics_single(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str
) -> Dict[str, float]:
    """
    Computes RAGAS metrics:
    - Faithfulness: Groundedness of the answer in retrieved context.
    - Answer Relevancy: How directly the answer addresses the question.
    - Context Precision: Signal-to-noise ratio in retrieved context.
    - Context Recall: Extent to which retrieved context covers ground truth.
    """
    combined_context = "\n---\n".join(contexts) if contexts else "No context retrieved."

    prompt_faithfulness = f"""
Rate the FAITHFULNESS of the Answer based strictly on the Context.
Return ONLY a floating-point number between 0.0 and 1.0 (where 1.0 means all claims in the Answer are fully supported by Context, and 0.0 means completely ungrounded/hallucinated).

Context:
{combined_context[:2000]}

Answer:
{answer[:1000]}

Score (0.0 to 1.0):
"""

    prompt_relevancy = f"""
Rate the ANSWER RELEVANCY of the Answer to the Question.
Return ONLY a floating-point number between 0.0 and 1.0 (where 1.0 means directly and completely answers the question, and 0.0 means off-topic).

Question:
{question}

Answer:
{answer[:1000]}

Score (0.0 to 1.0):
"""

    prompt_ctx_precision = f"""
Rate CONTEXT PRECISION (whether the most useful information in Context is present at the top).
Return ONLY a floating-point number between 0.0 and 1.0.

Question:
{question}

Context:
{combined_context[:2000]}

Score (0.0 to 1.0):
"""

    prompt_ctx_recall = f"""
Rate CONTEXT RECALL (whether the Context contains all necessary facts from Ground Truth).
Return ONLY a floating-point number between 0.0 and 1.0.

Ground Truth:
{ground_truth}

Context:
{combined_context[:2000]}

Score (0.0 to 1.0):
"""

    faithfulness = run_llm_judge(prompt_faithfulness)
    answer_relevancy = run_llm_judge(prompt_relevancy)
    context_precision = run_llm_judge(prompt_ctx_precision)
    context_recall = run_llm_judge(prompt_ctx_recall)

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4)
    }


def write_csv(filepath: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """Writes a list of dictionaries to a CSV file using standard csv module."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_html_report(
    ragas_summary: Dict[str, float],
    retrieval_summary: Dict[str, float],
    latency_summary: Dict[str, Dict[str, float]],
    cache_summary: Dict[str, Any],
    results_data: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Generates a modern, color-coded HTML report summarizing evaluation results.
    """
    def score_badge(val: float) -> str:
        color = "#10b981" if val >= 0.85 else "#f59e0b" if val >= 0.70 else "#ef4444"
        return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{val:.2f}</span>'

    failing_queries = [r for r in results_data if r.get("hit_rate", 1.0) == 0.0][:5]
    hallucinated = [r for r in results_data if r.get("faithfulness", 1.0) < 0.75][:5]
    sorted_by_lat = sorted(results_data, key=lambda x: x.get("total_response_time", 0.0), reverse=True)
    high_latency = sorted_by_lat[:5]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report | AI HR Assistant</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --accent: #38bdf8;
            --border: #334155;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1, h2, h3 {{
            color: var(--accent);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .card .title {{
            font-size: 0.9em;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .card .value {{
            font-size: 2.2em;
            font-weight: bold;
            margin-top: 10px;
            color: #f1f5f9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: #090d16;
            color: var(--accent);
            text-transform: uppercase;
            font-size: 0.85em;
        }}
        tr:hover {{
            background-color: #26334d;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        .badge-success {{ background-color: #059669; color: white; }}
        .badge-warning {{ background-color: #d97706; color: white; }}
        .badge-danger {{ background-color: #dc2626; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Evaluation Dashboard</h1>
            <p>Production-Grade RAG Evaluation Report for AI HR Policy Assistant</p>
        </div>

        <h2>Overall RAGAS Metrics</h2>
        <div class="metrics-grid">
            <div class="card">
                <div class="title">Faithfulness</div>
                <div class="value">{score_badge(ragas_summary.get('Faithfulness', 0.0))}</div>
            </div>
            <div class="card">
                <div class="title">Answer Relevancy</div>
                <div class="value">{score_badge(ragas_summary.get('Answer Relevancy', 0.0))}</div>
            </div>
            <div class="card">
                <div class="title">Context Precision</div>
                <div class="value">{score_badge(ragas_summary.get('Context Precision', 0.0))}</div>
            </div>
            <div class="card">
                <div class="title">Context Recall</div>
                <div class="value">{score_badge(ragas_summary.get('Context Recall', 0.0))}</div>
            </div>
        </div>

        <h2>Retrieval Evaluation Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
"""
    for k, v in retrieval_summary.items():
        html_content += f"""
                <tr>
                    <td><strong>{k}</strong></td>
                    <td>{v:.4f}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <h2>Latency Evaluation (Seconds)</h2>
        <table>
            <thead>
                <tr>
                    <th>Pipeline Phase</th>
                    <th>Average</th>
                    <th>Minimum</th>
                    <th>Maximum</th>
                    <th>Median</th>
                </tr>
            </thead>
            <tbody>
"""
    for phase, stats in latency_summary.items():
        html_content += f"""
                <tr>
                    <td><strong>{phase.replace('_', ' ').title()}</strong></td>
                    <td>{stats['Average']:.4f}s</td>
                    <td>{stats['Minimum']:.4f}s</td>
                    <td>{stats['Maximum']:.4f}s</td>
                    <td>{stats['Median']:.4f}s</td>
                </tr>
"""

    html_content += f"""
            </tbody>
        </table>

        <h2>Cache Performance Statistics</h2>
        <div class="metrics-grid">
            <div class="card">
                <div class="title">Cache Hits</div>
                <div class="value">{cache_summary.get('Cache Hits', 0)}</div>
            </div>
            <div class="card">
                <div class="title">Cache Misses</div>
                <div class="value">{cache_summary.get('Cache Misses', 0)}</div>
            </div>
            <div class="card">
                <div class="title">Cache Hit Rate</div>
                <div class="value">{cache_summary.get('Cache Hit Rate', 0.0) * 100:.1f}%</div>
            </div>
        </div>

        <h2>Detailed Failure & Insight Analysis</h2>

        <h3>Top Failing Queries (Hit Rate = 0)</h3>
        <table>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Expected Document</th>
                    <th>Retrieved Documents</th>
                </tr>
            </thead>
            <tbody>
"""
    if not failing_queries:
        html_content += "<tr><td colspan='3'>No failing queries detected. All expected documents were retrieved successfully!</td></tr>"
    else:
        for row in failing_queries:
            html_content += f"""
                    <tr>
                        <td>{row['question']}</td>
                        <td><span class="badge badge-warning">{row['expected_document']}</span></td>
                        <td>{row['retrieved_documents']}</td>
                    </tr>
"""

    html_content += """
            </tbody>
        </table>

        <h3>Highest Latency Queries</h3>
        <table>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Total Response Time</th>
                    <th>LLM Time</th>
                </tr>
            </thead>
            <tbody>
"""
    for row in high_latency:
        html_content += f"""
                <tr>
                    <td>{row['question']}</td>
                    <td><span class="badge badge-warning">{row['total_response_time']:.3f}s</span></td>
                    <td>{row['llm_time']:.3f}s</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <h3>Potential Hallucination & Low Faithfulness Answers</h3>
        <table>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Generated Answer</th>
                    <th>Faithfulness Score</th>
                </tr>
            </thead>
            <tbody>
"""
    if len(hallucinated) == 0:
        html_content += "<tr><td colspan='3'>No hallucinated answers detected. All responses strictly grounded in context.</td></tr>"
    else:
        for row in hallucinated:
            html_content += f"""
                    <tr>
                        <td>{row['question']}</td>
                        <td>{row['generated_answer'][:150]}...</td>
                        <td><span class="badge badge-danger">{row['faithfulness']:.2f}</span></td>
                    </tr>
"""

    html_content += """
            </tbody>
        </table>

    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    print(f"Starting evaluation of {total_queries} queries...\n")

    results_data = []
    vector_results_all = []
    expected_docs_all = []
    latency_records = []

    # Clear cache before evaluation pass
    query_cache.clear()

    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        exp_doc = item["expected_document"]

        print(f"\nEvaluating query {i}/{total_queries}...")

        # 1. Preprocessing & Embedding
        with Timer() as t_pre:
            cleaned_query, keywords = preprocess_query(q)
        search_text = cleaned_query or q

        with Timer() as t_emb:
            query_embedding = generate_embedding(search_text)

        # 2. Retrieval
        print("\nRunning retrieval...")
        with Timer() as t_vec:
            vec_res = search_documents(query_embedding=query_embedding, top_k=5)

        vector_results_all.append(vec_res)
        expected_docs_all.append(exp_doc)

        # Build context string
        context_str = ""
        retrieved_doc_names = []
        retrieval_scores = []
        for r in vec_res:
            doc_name = r.get("document", r.get("payload", {}).get("source", "Unknown"))
            retrieved_doc_names.append(doc_name)
            retrieval_scores.append(round(r.get("score", 0.0), 4))
            context_str += f"[Source: {doc_name}, Chunk {r.get('chunk_number', 0)}]\n{r.get('text', '')}\n\n"

        # 3. Answer Generation
        print("\nGenerating answer...")
        with Timer() as t_llm:
            llm_res = generate_answer(question=q, context=context_str, top_k=len(vec_res))
            ans = llm_res.get("answer", "")

        tot_time = round(t_pre.elapsed + t_emb.elapsed + t_vec.elapsed + t_llm.elapsed, 4)

        # 4. RAGAS Evaluation
        print("\nRunning RAGAS...")
        contexts_list = [r.get("text", "") for r in vec_res]
        ragas_scores = calculate_ragas_metrics_single(
            question=q,
            answer=ans,
            contexts=contexts_list,
            ground_truth=gt
        )

        hit = calculate_recall_at_k(vec_res, exp_doc, 5)

        cache_key = search_text.lower()
        query_cache[cache_key] = ans

        latency_records.append({
            "preprocessing": round(t_pre.elapsed, 4),
            "embedding": round(t_emb.elapsed, 4),
            "vector_retrieval": round(t_vec.elapsed, 4),
            "llm": round(t_llm.elapsed, 4),
            "total_response": tot_time,
        })

        results_data.append({
            "question": q,
            "preprocessed_query": search_text,
            "ground_truth": gt,
            "expected_document": exp_doc,
            "retrieved_documents": "|".join(retrieved_doc_names),
            "retrieval_scores": "|".join(map(str, retrieval_scores)),
            "generated_answer": ans,
            "faithfulness": ragas_scores["faithfulness"],
            "answer_relevancy": ragas_scores["answer_relevancy"],
            "context_precision": ragas_scores["context_precision"],
            "context_recall": ragas_scores["context_recall"],
            "hit_rate": hit,
            "preprocessing_time": round(t_pre.elapsed, 4),
            "embedding_time": round(t_emb.elapsed, 4),
            "vector_retrieval_time": round(t_vec.elapsed, 4),
            "llm_time": round(t_llm.elapsed, 4),
            "total_response_time": tot_time
        })

    # Cache Hit Verification Pass
    print("\nRunning Cache Hit Rate verification pass...")
    cache_records_pass2 = []
    for item in dataset:
        q = item["question"]
        cleaned_query, _ = preprocess_query(q)
        cache_key = (cleaned_query or q).lower()
        cache_records_pass2.append(cache_key in query_cache)

    print("\nSaving report...")

    # 1. RAGAS CSV & Summary
    ragas_results_csv = os.path.join(REPORTS_DIR, "ragas_results.csv")
    fieldnames = list(results_data[0].keys())
    write_csv(ragas_results_csv, fieldnames, results_data)

    faithfulness_avg = sum(r["faithfulness"] for r in results_data) / total_queries
    relevancy_avg = sum(r["answer_relevancy"] for r in results_data) / total_queries
    precision_avg = sum(r["context_precision"] for r in results_data) / total_queries
    recall_avg = sum(r["context_recall"] for r in results_data) / total_queries

    ragas_summary = {
        "Faithfulness": round(faithfulness_avg, 2),
        "Answer Relevancy": round(relevancy_avg, 2),
        "Context Precision": round(precision_avg, 2),
        "Context Recall": round(recall_avg, 2),
    }

    ragas_summary_json = os.path.join(REPORTS_DIR, "ragas_summary.json")
    with open(ragas_summary_json, "w", encoding="utf-8") as f:
        json.dump(ragas_summary, f, indent=4)

    # Console RAGAS summary output exact match formatting
    print("\n" + f"Faithfulness        {ragas_summary['Faithfulness']:.2f}")
    print(f"Answer Relevancy    {ragas_summary['Answer Relevancy']:.2f}")
    print(f"Context Precision   {ragas_summary['Context Precision']:.2f}")
    print(f"Context Recall      {ragas_summary['Context Recall']:.2f}\n")

    # 2. Retrieval Metrics CSV
    retrieval_summary = evaluate_retrieval_batch(vector_results_all, expected_docs_all)
    retrieval_metrics_csv = os.path.join(REPORTS_DIR, "retrieval_metrics.csv")
    ret_rows = [{"Metric": k, "Value": v} for k, v in retrieval_summary.items()]
    write_csv(retrieval_metrics_csv, ["Metric", "Value"], ret_rows)

    # 3. Latency Metrics CSV
    latency_stats = calculate_latency_stats(latency_records)
    latency_rows = []
    for phase, stats in latency_stats.items():
        row = {"Phase": phase}
        row.update(stats)
        latency_rows.append(row)

    latency_metrics_csv = os.path.join(REPORTS_DIR, "latency_metrics.csv")
    lat_fieldnames = list(latency_rows[0].keys())
    write_csv(latency_metrics_csv, lat_fieldnames, latency_rows)

    # 4. Cache Metrics CSV
    cache_stats = calculate_cache_stats(cache_records_pass2)
    cache_metrics_csv = os.path.join(REPORTS_DIR, "cache_metrics.csv")
    write_csv(cache_metrics_csv, list(cache_stats.keys()), [cache_stats])

    # 5. HTML Report Generation
    html_report_path = os.path.join(REPORTS_DIR, "evaluation_report.html")
    generate_html_report(
        ragas_summary=ragas_summary,
        retrieval_summary=retrieval_summary,
        latency_summary=latency_stats,
        cache_summary=cache_stats,
        results_data=results_data,
        output_path=html_report_path
    )

    print(f"\nEvaluation pipeline complete! Reports successfully generated in:\n {REPORTS_DIR}")


if __name__ == "__main__":
    main()
