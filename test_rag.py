import json
import time
from datetime import datetime
from ollama_llm import run_rag_query, query_ollama, get_available_models
from query_database import load_database


class RAGTester:
    """Class to test and evaluate RAG system performance with mixed-language support."""

    def __init__(self, db=None):
        """Initialize RAG tester."""
        self.db = db or load_database()
        self.test_results = []

    def evaluate_answer_similarity(self, expected, actual, judge_model="phi3"):
        """
        Use LLM to judge if two answers are equivalent in meaning.
        Enhanced to handle mixed-language evaluation.

        Args:
            expected: Expected answer (English or Bangla)
            actual: Actual RAG system answer
            judge_model: Model to use for evaluation

        Returns:
            Tuple of (is_equivalent: bool, confidence: str, explanation: str)
        """
        judge_prompt = f"""
You are an expert evaluator for a multilingual RAG system. Compare these two answers and determine if they convey the same core information and meaning.

Expected Answer: "{expected}"

Actual Answer: "{actual}"

Evaluation Criteria:
- Do both answers address the same key concepts?
- Is the factual information consistent?
- Minor differences in wording are acceptable
- Focus on semantic equivalence, not exact word matching
- Both answers may be in different languages (English/Bangla) but should convey same meaning
- Consider cross-language consistency where applicable

Respond with ONLY this format:
EQUIVALENT: [TRUE/FALSE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Brief explanation of your judgment]
"""

        try:
            response = query_ollama(
                judge_prompt, model=judge_model, max_tokens=200, temperature=0.1
            )

            if not response:
                return False, "LOW", "Failed to get evaluation from judge model"

            # Parse response
            lines = response.strip().split("\n")
            equivalent = False
            confidence = "LOW"
            explanation = "Could not parse evaluation"

            for line in lines:
                if line.startswith("EQUIVALENT:"):
                    equivalent = "TRUE" in line.upper()
                elif line.startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip()
                elif line.startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()

            return equivalent, confidence, explanation

        except Exception as e:
            return False, "LOW", f"Error in evaluation: {e}"

    def test_single_question(
        self, question, expected_answer, test_id=None, language="auto"
    ):
        """
        Test a single question and evaluate the result.
        Enhanced with language detection and mixed-language support.

        Args:
            question: Question to ask
            expected_answer: Expected answer for comparison
            test_id: Optional test identifier
            language: Language of the question (auto, english, bangla)

        Returns:
            Dictionary with test results
        """
        print(f"\n🧪 Testing ({language}): {question}")

        start_time = time.time()

        # Get RAG response
        rag_result = run_rag_query(question, db=self.db, k=3)

        end_time = time.time()
        response_time = end_time - start_time

        if not rag_result["success"]:
            return {
                "test_id": test_id,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": rag_result["answer"],
                "success": False,
                "equivalent": False,
                "confidence": "N/A",
                "explanation": "RAG query failed",
                "response_time": response_time,
                "sources_found": 0,
                "model_used": None,
                "language": language,
            }

        # Evaluate answer quality
        equivalent, confidence, explanation = self.evaluate_answer_similarity(
            expected_answer, rag_result["answer"]
        )

        result = {
            "test_id": test_id,
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": rag_result["answer"],
            "success": rag_result["success"],
            "equivalent": equivalent,
            "confidence": confidence,
            "explanation": explanation,
            "response_time": response_time,
            "sources_found": rag_result.get("num_sources", 0),
            "model_used": rag_result.get("model_used"),
            "source_pages": [s["page"] for s in rag_result.get("sources", [])],
            "language": language,
        }

        # Print results
        status = "✅ PASS" if equivalent else "❌ FAIL"
        print(f"{status} - Equivalent: {equivalent} (Confidence: {confidence})")
        print(f"Response time: {response_time:.2f}s")
        print(
            f"Sources: {result['sources_found']} chunks from pages {result['source_pages']}"
        )

        return result

    def run_test_suite(self):
        """Run comprehensive test suite with various question types in English and Bangla."""

        test_cases = [
            # === ENGLISH ALGORITHM QUESTIONS ===
            {
                "id": "algo_001_en",
                "question": "What is an algorithm?",
                "expected": "A finite sequence of instructions for solving a problem or computational procedure",
                "language": "english",
            },
            {
                "id": "sort_001_en",
                "question": "How does merge sort work?",
                "expected": "Merge sort divides array into halves, recursively sorts them, then merges sorted halves",
                "language": "english",
            },
            {
                "id": "complexity_001_en",
                "question": "What is the time complexity of quicksort?",
                "expected": "Average case O(n log n), worst case O(n²)",
                "language": "english",
            },
            {
                "id": "ds_001_en",
                "question": "What is a binary search tree?",
                "expected": "A binary tree where left subtree values are less than node, right subtree values are greater",
                "language": "english",
            },
            {
                "id": "dp_001_en",
                "question": "What is dynamic programming?",
                "expected": "An algorithmic technique that solves problems by breaking them into overlapping subproblems",
                "language": "english",
            },
            # === BANGLA ALGORITHM QUESTIONS ===
            {
                "id": "algo_001_bn",
                "question": "অ্যালগরিদম কি?",
                "expected": "অ্যালগরিদম হল একটি সমস্যা সমাধানের জন্য নির্দিষ্ট নির্দেশাবলীর একটি ক্রম",
                "language": "bangla",
            },
            {
                "id": "sort_001_bn",
                "question": "মার্জ সর্ট কিভাবে কাজ করে?",
                "expected": "মার্জ সর্ট অ্যারেকে অংশে ভাগ করে, প্রতিটি অংশ আলাদাভাবে সাজায়, তারপর সেগুলো একত্রিত করে",
                "language": "bangla",
            },
            {
                "id": "complexity_001_bn",
                "question": "কুইকসর্টের সময় জটিলতা কত?",
                "expected": "গড় ক্ষেত্রে O(n log n), সবচেয়ে খারাপ ক্ষেত্রে O(n²)",
                "language": "bangla",
            },
            {
                "id": "ds_001_bn",
                "question": "বাইনারি সার্চ ট্রি কি?",
                "expected": "একটি বাইনারি ট্রি যেখানে বাম সাবট্রিতে ছোট মান এবং ডান সাবট্রিতে বড় মান থাকে",
                "language": "bangla",
            },
            {
                "id": "dp_001_bn",
                "question": "ডাইনামিক প্রোগ্রামিং কি?",
                "expected": "একটি অ্যালগরিদমিক পদ্ধতি যা সমস্যাকে ছোট উপ-সমস্যায় ভাগ করে সমাধান করে",
                "language": "bangla",
            },
            # === MIXED LANGUAGE TEXTBOOK QUESTIONS ===
            {
                "id": "textbook_001_en",
                "question": "What is asymptotic notation?",
                "expected": "Mathematical notation used to describe the limiting behavior of functions, commonly used for algorithm analysis",
                "language": "english",
            },
            {
                "id": "textbook_001_bn",
                "question": "অ্যাসিম্পটোটিক নোটেশন কি?",
                "expected": "ফাংশনের সীমাবদ্ধতা বর্ণনার জন্য ব্যবহৃত গাণিতিক নোটেশন যা অ্যালগরিদম বিশ্লেষণে ব্যবহৃত হয়",
                "language": "bangla",
            },
            {
                "id": "heap_001_en",
                "question": "How do you maintain the heap property?",
                "expected": "Use heapify operations to ensure parent nodes satisfy heap property relative to children",
                "language": "english",
            },
            {
                "id": "heap_001_bn",
                "question": "হিপ প্রপার্টি কিভাবে বজায় রাখা হয়?",
                "expected": "হিপিফাই অপারেশন ব্যবহার করে প্যারেন্ট নোড এবং চাইল্ড নোডের মধ্যে হিপ প্রপার্টি বজায় রাখা হয়",
                "language": "bangla",
            },
            {
                "id": "graph_001_en",
                "question": "What is Dijkstra's algorithm used for?",
                "expected": "Finding shortest paths from a source vertex to all other vertices in weighted graphs",
                "language": "english",
            },
            {
                "id": "graph_001_bn",
                "question": "ডাইজস্ট্রার অ্যালগরিদম কি জন্য ব্যবহৃত হয়?",
                "expected": "ওজনযুক্ত গ্রাফে একটি উৎস থেকে অন্যান্য সব শীর্ষে সবচেয়ে ছোট পথ খুঁজে বের করার জন্য",
                "language": "bangla",
            },
            # === ADVANCED TEXTBOOK CONCEPTS ===
            {
                "id": "advanced_001_en",
                "question": "What is the master theorem?",
                "expected": "A method for solving recurrence relations commonly found in divide-and-conquer algorithms",
                "language": "english",
            },
            {
                "id": "advanced_001_bn",
                "question": "মাস্টার থিওরেম কি?",
                "expected": "ডিভাইড অ্যান্ড কনকার অ্যালগরিদমে পাওয়া রিকারেন্স রিলেশন সমাধানের একটি পদ্ধতি",
                "language": "bangla",
            },
            {
                "id": "complexity_002_en",
                "question": "What is NP-completeness?",
                "expected": "A class of computational problems that are among the most difficult problems in NP",
                "language": "english",
            },
            {
                "id": "complexity_002_bn",
                "question": "NP-completeness কি?",
                "expected": "NP-তে সবচেয়ে কঠিন সমস্যাগুলোর একটি শ্রেণি যা গণনাগত জটিলতার ক্ষেত্রে গুরুত্বপূর্ণ",
                "language": "bangla",
            },
            # === NEGATIVE TEST CASES ===
            {
                "id": "negative_001_en",
                "question": "How do I cook pasta?",
                "expected": "No relevant information found",
                "language": "english",
            },
            {
                "id": "negative_001_bn",
                "question": "আমি কিভাবে পাস্তা রান্না করব?",
                "expected": "কোন প্রাসঙ্গিক তথ্য পাওয়া যায়নি",
                "language": "bangla",
            },
            {
                "id": "negative_002_en",
                "question": "What is the weather today?",
                "expected": "No relevant information found",
                "language": "english",
            },
        ]

        print("🚀 Starting BanglaRAG System Test Suite")
        print(f"📊 Running {len(test_cases)} test cases...")
        print("🌐 Testing English & Bangla queries")
        print("📚 Evaluating textbook content comprehension")
        print("=" * 60)

        results = []

        for test_case in test_cases:
            result = self.test_single_question(
                test_case["question"],
                test_case["expected"],
                test_case["id"],
                test_case["language"],
            )
            results.append(result)
            time.sleep(1)  # Brief pause between tests

        self.test_results = results
        return results

    def generate_test_report(self, save_to_file=True):
        """Generate comprehensive test report with language-specific metrics."""
        if not self.test_results:
            print("No test results to report. Run test suite first.")
            return

        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["equivalent"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) * 100

        avg_response_time = (
            sum(r["response_time"] for r in self.test_results) / total_tests
        )

        # Calculate language-specific statistics
        english_tests = [r for r in self.test_results if r["language"] == "english"]
        bangla_tests = [r for r in self.test_results if r["language"] == "bangla"]

        english_passed = sum(1 for r in english_tests if r["equivalent"])
        bangla_passed = sum(1 for r in bangla_tests if r["equivalent"])

        english_pass_rate = (
            (english_passed / len(english_tests)) * 100 if english_tests else 0
        )
        bangla_pass_rate = (
            (bangla_passed / len(bangla_tests)) * 100 if bangla_tests else 0
        )

        # Calculate confidence distribution
        confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for result in self.test_results:
            if result["equivalent"]:
                confidence_counts[result["confidence"]] += 1

        # Generate report
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": round(pass_rate, 2),
                "average_response_time": round(avg_response_time, 2),
                "language_breakdown": {
                    "english": {
                        "total": len(english_tests),
                        "passed": english_passed,
                        "pass_rate": round(english_pass_rate, 2),
                    },
                    "bangla": {
                        "total": len(bangla_tests),
                        "passed": bangla_passed,
                        "pass_rate": round(bangla_pass_rate, 2),
                    },
                },
                "confidence_distribution": confidence_counts,
            },
            "detailed_results": self.test_results,
        }

        # Print summary
        print("\n" + "=" * 60)
        print("📋 BANGLARAG SYSTEM TEST REPORT")
        print("=" * 60)
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        print(f"   Avg Response Time: {avg_response_time:.2f}s")

        print(f"\n🌐 LANGUAGE-SPECIFIC RESULTS:")
        print(
            f"   English: {english_passed}/{len(english_tests)} ({english_pass_rate:.1f}%)"
        )
        print(
            f"   Bangla: {bangla_passed}/{len(bangla_tests)} ({bangla_pass_rate:.1f}%)"
        )

        print(f"\n📈 CONFIDENCE DISTRIBUTION:")
        for conf, count in confidence_counts.items():
            print(f"   {conf}: {count} tests")

        # Show failed tests
        failed_results = [r for r in self.test_results if not r["equivalent"]]
        if failed_results:
            print(f"\n❌ Failed Tests ({len(failed_results)}):")
            for result in failed_results:
                lang_flag = {"english": "🇺🇸", "bangla": "🇧🇩"}.get(
                    result["language"], "🌐"
                )
                print(f"  {lang_flag} {result['test_id']}: {result['question']}")
                print(f"    Reason: {result['explanation']}")

        # Show passed tests
        passed_results = [r for r in self.test_results if r["equivalent"]]
        if passed_results:
            print(f"\n✅ Passed Tests ({len(passed_results)}):")
            for result in passed_results:
                confidence_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(
                    result["confidence"], "⚪"
                )
                lang_flag = {"english": "🇺🇸", "bangla": "🇧🇩"}.get(
                    result["language"], "🌐"
                )
                print(
                    f"  {lang_flag} {result['test_id']}: {result['question']} {confidence_emoji}"
                )

        # Save to file
        if save_to_file:
            filename = (
                f"banglarag_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Detailed report saved to: {filename}")

        print("=" * 60)

        return report


def run_quick_test():
    """Run a quick test with a few questions."""
    print("🏃‍♂️ Running Quick RAG Test...")

    # Check if database exists
    db = load_database()
    if not db:
        print("❌ Database not found. Please run create_database.py first.")
        return

    # Check if Ollama is available
    models = get_available_models()
    if not models:
        print("❌ No Ollama models available. Please install a model: ollama pull phi3")
        return

    # Initialize tester
    tester = RAGTester(db)

    # Test a few questions
    quick_tests = [
        (
            "What is an algorithm?",
            "A procedure or set of instructions for solving a problem",
        ),
        (
            "How does quicksort work?",
            "Divides array using pivot, recursively sorts partitions",
        ),
        ("What is machine learning?", "No relevant information found"),  # Should fail
    ]

    results = []
    for question, expected in quick_tests:
        result = tester.test_single_question(question, expected)
        results.append(result)

    # Quick summary
    passed = sum(1 for r in results if r["equivalent"])
    print(f"\n📊 Quick Test Results: {passed}/{len(results)} passed")

    return results


def run_bangla_quick_test():
    """Run a quick test with mixed English and Bangla questions."""
    print("🏃‍♂️ Running Quick BanglaRAG Test...")

    # Check if database exists
    db = load_database()
    if not db:
        print("❌ Database not found. Please run create_database.py first.")
        return

    # Check if Ollama is available
    models = get_available_models()
    if not models:
        print("❌ No Ollama models available. Please install a model: ollama pull phi3")
        return

    # Initialize tester
    tester = RAGTester(db)

    # Test mixed language questions
    quick_tests = [
        (
            "What is an algorithm?",
            "A procedure or set of instructions for solving a problem",
            "english",
        ),
        (
            "অ্যালগরিদম কি?",
            "অ্যালগরিদম হল একটি সমস্যা সমাধানের জন্য নির্দিষ্ট নির্দেশাবলীর একটি ক্রম",
            "bangla",
        ),
        (
            "How does quicksort work?",
            "Divides array using pivot, recursively sorts partitions",
            "english",
        ),
        (
            "কুইকসর্টের সময় জটিলতা কত?",
            "গড় ক্ষেত্রে O(n log n), সবচেয়ে খারাপ ক্ষেত্রে O(n²)",
            "bangla",
        ),
        (
            "What is machine learning?",
            "No relevant information found",
            "english",
        ),  # Should fail
    ]

    results = []
    for question, expected, language in quick_tests:
        result = tester.test_single_question(question, expected, language=language)
        results.append(result)

    # Quick summary
    passed = sum(1 for r in results if r["equivalent"])
    english_passed = sum(
        1 for r in results if r["language"] == "english" and r["equivalent"]
    )
    bangla_passed = sum(
        1 for r in results if r["language"] == "bangla" and r["equivalent"]
    )

    print(f"\n📊 Quick Test Results:")
    print(f"   Overall: {passed}/{len(results)} passed")
    print(
        f"   English: {english_passed}/{sum(1 for r in results if r['language'] == 'english')} passed"
    )
    print(
        f"   Bangla: {bangla_passed}/{sum(1 for r in results if r['language'] == 'bangla')} passed"
    )

    return results


if __name__ == "__main__":
    print("🧪 BanglaRAG System Testing Framework")
    print("Choose test mode:")
    print("1. Quick test (5 mixed-language questions)")
    print("2. Full test suite (20+ mixed-language questions)")
    print("3. Original quick test (3 English questions)")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        run_bangla_quick_test()
    elif choice == "2":
        # Full test suite
        db = load_database()
        if not db:
            print("❌ Database not found. Please run create_database.py first.")
        else:
            tester = RAGTester(db)
            tester.run_test_suite()
            tester.generate_test_report()
    elif choice == "3":
        run_quick_test()
    else:
        print("Invalid choice. Running Bangla quick test...")
        run_bangla_quick_test()
