import time
import json
import random
from typing import List, Dict

# Import system modules (assume these exist and are importable)
import main  # For text/voice QA pipeline
import voice_input  # For ASR
import config
import embedding
import query_database
import ollama_llm
import translator

# --- Test Data ---
# Example queries (Bangla, English, mixed)
TEST_QUERIES = [
    {"query": "অ্যালগরিদম কী?", "expected_page": 1, "language": "bangla"},
    {"query": "What is an algorithm?", "expected_page": 1, "language": "english"},
    {"query": "Sorting কীভাবে কাজ করে?", "expected_page": 5, "language": "bangla"},
    {"query": "Explain merge sort.", "expected_page": 7, "language": "english"},
    {"query": "বাইনারি সার্চ কি?", "expected_page": 10, "language": "bangla"},
    {"query": "What is binary search?", "expected_page": 10, "language": "english"},
]

# Example audio files for ASR (must exist in the project)
ASR_TEST_FILES = [
    {"audio_path": "test_audio_bangla_1.wav", "expected_text": "অ্যালগরিদম কী?"},
    {
        "audio_path": "test_audio_english_1.wav",
        "expected_text": "What is an algorithm?",
    },
]


# --- Utility Functions ---
def run_qa_test(
    query: str, expected_page: int, language: str, config_overrides=None
) -> Dict:
    """Run a QA test and return result dict."""
    # Apply config overrides
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)
    start = time.time()
    result = ollama_llm.run_rag_query(query)
    elapsed = time.time() - start
    # Extract answer, cited page
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    cited_pages = [c.get("page_number") for c in citations if c.get("page_number")]
    correct_citation = expected_page in cited_pages
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "expected_page": expected_page,
        "cited_pages": cited_pages,
        "correct_citation": correct_citation,
        "response_time": elapsed,
        "language": language,
    }


def run_asr_test(audio_path: str, expected_text: str) -> Dict:
    """Run ASR test and return result dict."""
    transcribed = voice_input.transcribe_audio(audio_path)
    correct = transcribed.strip() == expected_text.strip()
    return {
        "audio_path": audio_path,
        "expected_text": expected_text,
        "transcribed": transcribed,
        "correct": correct,
    }


def ablation_configs():
    """Yield (name, config_overrides) for each ablation setting."""
    yield ("full_system", {})
    yield (
        "no_translation",
        {"SKIP_TRANSLATION_FOR_ENGLISH": True, "CACHE_TRANSLATIONS": False},
    )
    yield ("no_banglabert", {"USE_BANGLABERT": False})
    yield ("no_citation", {"INCLUDE_CITATION": False})


# --- Main Test Runner ---
def main_test():
    report = {"qa": {}, "asr": {}, "ablation": {}, "qualitative": []}

    # 1. QA, Citation, Response Time
    qa_results = []
    for q in TEST_QUERIES:
        res = run_qa_test(q["query"], q["expected_page"], q["language"])
        qa_results.append(res)
    report["qa"]["results"] = qa_results
    report["qa"]["accuracy"] = sum(r["correct_citation"] for r in qa_results) / len(
        qa_results
    )
    report["qa"]["avg_response_time"] = sum(
        r["response_time"] for r in qa_results
    ) / len(qa_results)

    # 2. ASR
    asr_results = []
    for a in ASR_TEST_FILES:
        res = run_asr_test(a["audio_path"], a["expected_text"])
        asr_results.append(res)
    report["asr"]["results"] = asr_results
    report["asr"]["accuracy"] = sum(r["correct"] for r in asr_results) / len(
        asr_results
    )

    # 3. Ablation Studies
    ablation_summary = {}
    for name, overrides in ablation_configs():
        ablation_qa = []
        for q in TEST_QUERIES:
            res = run_qa_test(
                q["query"],
                q["expected_page"],
                q["language"],
                config_overrides=overrides,
            )
            ablation_qa.append(res)
        accuracy = sum(r["correct_citation"] for r in ablation_qa) / len(ablation_qa)
        ablation_summary[name] = {"accuracy": accuracy, "results": ablation_qa}
    report["ablation"] = ablation_summary

    # 4. Qualitative Examples
    for q in random.sample(TEST_QUERIES, min(3, len(TEST_QUERIES))):
        res = run_qa_test(q["query"], q["expected_page"], q["language"])
        report["qualitative"].append(
            {
                "query": q["query"],
                "answer": res["answer"],
                "citations": res["citations"],
                "expected_page": q["expected_page"],
            }
        )

    # Save report
    with open("rag_paper_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Test report saved to rag_paper_test_report.json")


if __name__ == "__main__":
    main_test()
