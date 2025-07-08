import json
import time
from datetime import datetime
from ollama_llm import run_rag_query, query_ollama, get_available_models
from query_database import load_database


class RAGTester:
    """Class to test and evaluate RAG system performance."""

    def __init__(self, db=None):
        """Initialize RAG tester."""
        self.db = db or load_database()
        self.test_results = []

    def evaluate_answer_similarity(self, expected, actual, judge_model="phi3"):
        """
        Use LLM to judge if two answers are equivalent in meaning.

        Args:
            expected: Expected answer
            actual: Actual RAG system answer
            judge_model: Model to use for evaluation

        Returns:
            Tuple of (is_equivalent: bool, confidence: str, explanation: str)
        """
        judge_prompt = f"""
You are an expert evaluator. Compare these two answers and determine if they convey the same core information and meaning.

Expected Answer: "{expected}"

Actual Answer: "{actual}"

Evaluation Criteria:
- Do both answers address the same key concepts?
- Is the factual information consistent?
- Minor differences in wording are acceptable
- Focus on semantic equivalence, not exact word matching

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

    def test_single_question(self, question, expected_answer, test_id=None):
        """
        Test a single question and evaluate the result.

        Args:
            question: Question to ask
            expected_answer: Expected answer for comparison
            test_id: Optional test identifier

        Returns:
            Dictionary with test results
        """
        print(f"\n🧪 Testing: {question}")

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
        """Run comprehensive test suite with various question types."""

        test_cases = [
            # Basic algorithmic concepts
            {
                "id": "algo_001",
                "question": "What is an algorithm?",
                "expected": "A finite sequence of instructions for solving a problem or computational procedure",
            },
            {
                "id": "sort_001",
                "question": "How does merge sort work?",
                "expected": "Merge sort divides array into halves, recursively sorts them, then merges sorted halves",
            },
            {
                "id": "complexity_001",
                "question": "What is the time complexity of quicksort?",
                "expected": "Average case O(n log n), worst case O(n²)",
            },
            {
                "id": "ds_001",
                "question": "What is a binary search tree?",
                "expected": "A binary tree where left subtree values are less than node, right subtree values are greater",
            },
            {
                "id": "dp_001",
                "question": "What is dynamic programming?",
                "expected": "An algorithmic technique that solves problems by breaking them into overlapping subproblems",
            },
            # Specific algorithmic details
            {
                "id": "heap_001",
                "question": "How do you maintain the heap property?",
                "expected": "Use heapify operations to ensure parent nodes satisfy heap property relative to children",
            },
            {
                "id": "graph_001",
                "question": "What is Dijkstra's algorithm used for?",
                "expected": "Finding shortest paths from a source vertex to all other vertices in weighted graphs",
            },
            # Negative test cases (should not find relevant info)
            {
                "id": "negative_001",
                "question": "How do I cook pasta?",
                "expected": "No relevant information found",
            },
            {
                "id": "negative_002",
                "question": "What is the weather today?",
                "expected": "No relevant information found",
            },
        ]

        print("🚀 Starting RAG System Test Suite")
        print(f"📊 Running {len(test_cases)} test cases...")
        print("=" * 60)

        results = []

        for test_case in test_cases:
            result = self.test_single_question(
                test_case["question"], test_case["expected"], test_case["id"]
            )
            results.append(result)
            time.sleep(1)  # Brief pause between tests

        self.test_results = results
        return results

    def generate_test_report(self, save_to_file=True):
        """Generate comprehensive test report."""
        if not self.test_results:
            print("No test results to report. Run test suite first.")
            return

        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["equivalent"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) * 100

        avg_response_time = (
            sum(r["response_time"] for r in self.test_results) / total_tests
        )

        # Generate report
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": round(pass_rate, 2),
                "average_response_time": round(avg_response_time, 2),
            },
            "detailed_results": self.test_results,
        }

        # Print summary
        print("\n" + "=" * 60)
        print("📋 RAG SYSTEM TEST REPORT")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Avg Response Time: {avg_response_time:.2f}s")

        # Show failed tests
        failed_results = [r for r in self.test_results if not r["equivalent"]]
        if failed_results:
            print(f"\n❌ Failed Tests ({len(failed_results)}):")
            for result in failed_results:
                print(f"  - {result['test_id']}: {result['question']}")
                print(f"    Reason: {result['explanation']}")

        # Show passed tests
        passed_results = [r for r in self.test_results if r["equivalent"]]
        if passed_results:
            print(f"\n✅ Passed Tests ({len(passed_results)}):")
            for result in passed_results:
                confidence_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(
                    result["confidence"], "⚪"
                )
                print(
                    f"  - {result['test_id']}: {result['question']} {confidence_emoji}"
                )

        # Save to file
        if save_to_file:
            filename = (
                f"rag_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
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
        print(
            "❌ No Ollama models available. Please install a model: ollama pull phi3"
        )
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


if __name__ == "__main__":
    print("🧪 RAG System Testing Framework")
    print("Choose test mode:")
    print("1. Quick test (3 questions)")
    print("2. Full test suite (9 questions)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        run_quick_test()
    elif choice == "2":
        # Full test suite
        db = load_database()
        if not db:
            print("❌ Database not found. Please run create_database.py first.")
        else:
            tester = RAGTester(db)
            tester.run_test_suite()
            tester.generate_test_report()
    else:
        print("Invalid choice. Running quick test...")
        run_quick_test()
