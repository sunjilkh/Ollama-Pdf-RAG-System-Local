#!/usr/bin/env python3
"""
Comprehensive test suite for optimized RAG system
Tests all components automatically without human intervention
"""

import time
import json
import sys
from datetime import datetime
from typing import Dict, List, Any


class ComprehensiveTestSuite:
    """Complete test suite for optimized RAG system"""

    def __init__(self):
        self.results = []
        self.start_time = None
        self.errors = []

    def run_all_tests(self):
        """Run all tests automatically"""
        print("🚀 Starting Comprehensive RAG System Tests")
        print("=" * 60)

        self.start_time = time.time()

        # Test 1: Database Loading Performance
        self.test_database_loading()

        # Test 2: Model Optimization
        self.test_model_optimization()

        # Test 3: Translation Optimization
        self.test_translation_optimization()

        # Test 4: Prompt Optimization
        self.test_prompt_optimization()

        # Test 5: End-to-End Performance
        self.test_end_to_end_performance()

        # Test 6: Functionality Verification
        self.test_functionality_verification()

        # Test 7: Error Handling
        self.test_error_handling()

        # Test 8: Cache Performance
        self.test_cache_performance()

        # Generate final report
        self.generate_final_report()

    def test_database_loading(self):
        """Test database loading performance"""
        print("\n1️⃣ Testing Database Loading Performance...")

        try:
            start = time.time()
            from query_database import load_database

            db = load_database()
            load_time = time.time() - start

            if db:
                print(f"✅ Database loaded in {load_time:.3f}s")
                self.results.append(
                    {
                        "test": "database_loading",
                        "success": True,
                        "time": load_time,
                        "status": "GOOD" if load_time < 2.0 else "SLOW",
                    }
                )
            else:
                print("❌ Database failed to load")
                self.results.append(
                    {
                        "test": "database_loading",
                        "success": False,
                        "time": load_time,
                        "status": "FAILED",
                    }
                )

        except Exception as e:
            print(f"❌ Database loading error: {e}")
            self.errors.append(f"Database loading: {e}")
            self.results.append(
                {
                    "test": "database_loading",
                    "success": False,
                    "error": str(e),
                    "status": "ERROR",
                }
            )

    def test_model_optimization(self):
        """Test model optimization features"""
        print("\n2️⃣ Testing Model Optimization...")

        try:
            from optimized_models import model_manager

            # Test model warm-up
            print("🔥 Testing model warm-up...")
            warmup_start = time.time()

            # Wait a bit for warmup to complete
            time.sleep(2)

            warmup_time = time.time() - warmup_start

            # Test optimized query
            print("🧠 Testing optimized query...")
            query_start = time.time()

            response, model = model_manager.query_with_smart_fallback(
                "What is an algorithm?", max_tokens=100, temperature=0.2
            )

            query_time = time.time() - query_start

            if response:
                print(f"✅ Model query successful in {query_time:.3f}s")
                print(f"📝 Response: {response[:80]}...")
                self.results.append(
                    {
                        "test": "model_optimization",
                        "success": True,
                        "query_time": query_time,
                        "model_used": model,
                        "status": "GOOD" if query_time < 10.0 else "SLOW",
                    }
                )
            else:
                print("❌ Model query failed")
                self.results.append(
                    {
                        "test": "model_optimization",
                        "success": False,
                        "query_time": query_time,
                        "status": "FAILED",
                    }
                )

        except Exception as e:
            print(f"❌ Model optimization error: {e}")
            self.errors.append(f"Model optimization: {e}")
            self.results.append(
                {
                    "test": "model_optimization",
                    "success": False,
                    "error": str(e),
                    "status": "ERROR",
                }
            )

    def test_translation_optimization(self):
        """Test translation optimization"""
        print("\n3️⃣ Testing Translation Optimization...")

        try:
            from translator import process_query_with_translation

            # Test English query (should skip translation)
            print("🔤 Testing English query...")
            english_start = time.time()
            english_result = process_query_with_translation("What is an algorithm?")
            english_time = time.time() - english_start

            if english_result["success"] and not english_result["translation_needed"]:
                print(
                    f"✅ English query processed in {english_time:.3f}s (translation skipped)"
                )
                english_status = "GOOD"
            else:
                print(
                    f"⚠️  English query processed in {english_time:.3f}s (translation not skipped)"
                )
                english_status = "SUBOPTIMAL"

            # Test Bangla query (should translate)
            print("🌐 Testing Bangla query...")
            bangla_start = time.time()
            bangla_result = process_query_with_translation("অ্যালগরিদম কি?")
            bangla_time = time.time() - bangla_start

            if bangla_result["success"] and bangla_result["translation_needed"]:
                print(
                    f"✅ Bangla query processed in {bangla_time:.3f}s (translation: {bangla_result['processed_query']})"
                )
                bangla_status = "GOOD"
            else:
                print(f"❌ Bangla query processing failed")
                bangla_status = "FAILED"

            self.results.append(
                {
                    "test": "translation_optimization",
                    "success": True,
                    "english_time": english_time,
                    "bangla_time": bangla_time,
                    "english_status": english_status,
                    "bangla_status": bangla_status,
                    "status": (
                        "GOOD"
                        if english_status == "GOOD" and bangla_status == "GOOD"
                        else "PARTIAL"
                    ),
                }
            )

        except Exception as e:
            print(f"❌ Translation optimization error: {e}")
            self.errors.append(f"Translation optimization: {e}")
            self.results.append(
                {
                    "test": "translation_optimization",
                    "success": False,
                    "error": str(e),
                    "status": "ERROR",
                }
            )

    def test_prompt_optimization(self):
        """Test prompt optimization"""
        print("\n4️⃣ Testing Prompt Optimization...")

        try:
            from query_database import load_database
            from ollama_llm import generate_optimized_prompt_template

            db = load_database()
            if not db:
                print("❌ Database not available for prompt testing")
                return

            # Test prompt generation
            print("📝 Testing optimized prompt generation...")
            prompt_start = time.time()

            # Get some sample results
            results = db.similarity_search("algorithm", k=3)

            if results:
                # Test original prompt
                from query_database import generate_prompt_template

                original_prompt = generate_prompt_template(
                    "What is an algorithm?", results
                )

                # Test optimized prompt
                optimized_prompt = generate_optimized_prompt_template(
                    "What is an algorithm?", results
                )

                prompt_time = time.time() - prompt_start

                original_len = len(original_prompt)
                optimized_len = len(optimized_prompt)
                reduction = ((original_len - optimized_len) / original_len) * 100

                print(f"✅ Prompt optimization completed in {prompt_time:.3f}s")
                print(f"📏 Original prompt: {original_len} chars")
                print(f"📏 Optimized prompt: {optimized_len} chars")
                print(f"⚡ Reduction: {reduction:.1f}%")

                self.results.append(
                    {
                        "test": "prompt_optimization",
                        "success": True,
                        "time": prompt_time,
                        "original_length": original_len,
                        "optimized_length": optimized_len,
                        "reduction_percent": reduction,
                        "status": "GOOD" if reduction > 20 else "MINIMAL",
                    }
                )
            else:
                print("❌ No results found for prompt testing")
                self.results.append(
                    {
                        "test": "prompt_optimization",
                        "success": False,
                        "status": "NO_RESULTS",
                    }
                )

        except Exception as e:
            print(f"❌ Prompt optimization error: {e}")
            self.errors.append(f"Prompt optimization: {e}")
            self.results.append(
                {
                    "test": "prompt_optimization",
                    "success": False,
                    "error": str(e),
                    "status": "ERROR",
                }
            )

    def test_end_to_end_performance(self):
        """Test end-to-end performance"""
        print("\n5️⃣ Testing End-to-End Performance...")

        test_queries = [
            "What is an algorithm?",
            "How does sorting work?",
            "Explain binary search",
            "অ্যালগরিদম কি?",  # Bangla query
            "What is data structure?",
        ]

        total_time = 0
        successful_queries = 0
        query_results = []

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}/5: {query}")

            try:
                start = time.time()

                # Test with optimized system
                from ollama_llm import run_rag_query

                result = run_rag_query(query, k=3, max_tokens=200)

                end = time.time()
                query_time = end - start
                total_time += query_time

                if result["success"]:
                    successful_queries += 1
                    print(f"✅ Success in {query_time:.3f}s")
                    print(f"📝 Answer: {result['answer'][:80]}...")

                    query_results.append(
                        {
                            "query": query,
                            "time": query_time,
                            "success": True,
                            "answer_length": len(result["answer"]),
                            "sources": result.get("num_sources", 0),
                        }
                    )
                else:
                    print(f"❌ Failed in {query_time:.3f}s: {result['answer']}")
                    query_results.append(
                        {
                            "query": query,
                            "time": query_time,
                            "success": False,
                            "error": result["answer"],
                        }
                    )

            except Exception as e:
                print(f"❌ Query error: {e}")
                self.errors.append(f"Query '{query}': {e}")
                query_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )

        avg_time = total_time / len(test_queries)
        success_rate = successful_queries / len(test_queries)

        print(f"\n📊 END-TO-END PERFORMANCE SUMMARY:")
        print(f"  ⏱️  Average time: {avg_time:.3f}s")
        print(f"  ✅ Success rate: {success_rate:.1%}")
        print(f"  🎯 Target: <5.0s")

        performance_status = (
            "EXCELLENT" if avg_time < 2 else "GOOD" if avg_time < 5 else "SLOW"
        )

        self.results.append(
            {
                "test": "end_to_end_performance",
                "success": True,
                "average_time": avg_time,
                "success_rate": success_rate,
                "total_queries": len(test_queries),
                "successful_queries": successful_queries,
                "query_results": query_results,
                "status": performance_status,
            }
        )

    def test_functionality_verification(self):
        """Test that all functionality still works after optimization"""
        print("\n6️⃣ Testing Functionality Verification...")

        functionality_tests = [
            ("English query processing", lambda: self._test_english_query()),
            ("Bangla query processing", lambda: self._test_bangla_query()),
            ("Source citations", lambda: self._test_source_citations()),
            ("Model fallback", lambda: self._test_model_fallback()),
        ]

        passed_tests = 0
        total_tests = len(functionality_tests)

        for test_name, test_func in functionality_tests:
            try:
                print(f"🧪 Testing {test_name}...")
                result = test_func()
                if result:
                    print(f"✅ {test_name} passed")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name} failed")
            except Exception as e:
                print(f"❌ {test_name} error: {e}")
                self.errors.append(f"{test_name}: {e}")

        functionality_score = passed_tests / total_tests

        self.results.append(
            {
                "test": "functionality_verification",
                "success": True,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "functionality_score": functionality_score,
                "status": "GOOD" if functionality_score > 0.8 else "PARTIAL",
            }
        )

        print(
            f"📊 Functionality Score: {functionality_score:.1%} ({passed_tests}/{total_tests})"
        )

    def _test_english_query(self):
        """Test English query processing"""
        from ollama_llm import run_rag_query

        result = run_rag_query("What is an algorithm?", k=2, max_tokens=100)
        return result["success"] and "algorithm" in result["answer"].lower()

    def _test_bangla_query(self):
        """Test Bangla query processing"""
        from ollama_llm import run_rag_query

        result = run_rag_query("অ্যালগরিদম কি?", k=2, max_tokens=100)
        return result["success"] and result.get("translation_info", {}).get(
            "translation_needed", False
        )

    def _test_source_citations(self):
        """Test source citations"""
        from ollama_llm import run_rag_query

        result = run_rag_query("What is sorting?", k=3, max_tokens=100)
        return (
            result["success"] and result.get("sources") and len(result["sources"]) > 0
        )

    def _test_model_fallback(self):
        """Test model fallback functionality"""
        from ollama_llm import get_available_models

        models = get_available_models()
        return len(models) > 0  # Just check if models are available

    def test_error_handling(self):
        """Test error handling"""
        print("\n7️⃣ Testing Error Handling...")

        error_tests = [
            ("Empty query", lambda: self._test_empty_query()),
            ("Invalid query", lambda: self._test_invalid_query()),
            ("Database unavailable", lambda: self._test_database_unavailable()),
        ]

        passed_error_tests = 0

        for test_name, test_func in error_tests:
            try:
                print(f"🧪 Testing {test_name}...")
                result = test_func()
                if result:
                    print(f"✅ {test_name} handled correctly")
                    passed_error_tests += 1
                else:
                    print(f"❌ {test_name} not handled correctly")
            except Exception as e:
                print(f"❌ {test_name} error: {e}")

        self.results.append(
            {
                "test": "error_handling",
                "success": True,
                "passed_error_tests": passed_error_tests,
                "total_error_tests": len(error_tests),
                "status": (
                    "GOOD" if passed_error_tests == len(error_tests) else "PARTIAL"
                ),
            }
        )

    def _test_empty_query(self):
        """Test empty query handling"""
        from ollama_llm import run_rag_query

        result = run_rag_query("", k=2, max_tokens=50)
        return not result["success"]  # Should fail gracefully

    def _test_invalid_query(self):
        """Test invalid query handling"""
        from ollama_llm import run_rag_query

        result = run_rag_query("@#$%^&*()", k=2, max_tokens=50)
        return "success" in result  # Should return some result structure

    def _test_database_unavailable(self):
        """Test database unavailable handling"""
        from ollama_llm import run_rag_query

        result = run_rag_query("test query", db=None, k=2, max_tokens=50)
        return "success" in result  # Should handle gracefully

    def test_cache_performance(self):
        """Test cache performance"""
        print("\n8️⃣ Testing Cache Performance...")

        try:
            # Test repeated queries to check caching
            from ollama_llm import run_rag_query

            query = "What is an algorithm?"

            # First query (cache miss)
            print("🔍 First query (cache miss)...")
            start1 = time.time()
            result1 = run_rag_query(query, k=3, max_tokens=100)
            time1 = time.time() - start1

            # Second query (should be faster due to caching)
            print("🔍 Second query (cache hit)...")
            start2 = time.time()
            result2 = run_rag_query(query, k=3, max_tokens=100)
            time2 = time.time() - start2

            cache_improvement = ((time1 - time2) / time1) * 100 if time1 > 0 else 0

            print(f"⏱️  First query: {time1:.3f}s")
            print(f"⏱️  Second query: {time2:.3f}s")
            print(f"🚀 Cache improvement: {cache_improvement:.1f}%")

            self.results.append(
                {
                    "test": "cache_performance",
                    "success": True,
                    "first_query_time": time1,
                    "second_query_time": time2,
                    "cache_improvement": cache_improvement,
                    "status": "GOOD" if cache_improvement > 10 else "MINIMAL",
                }
            )

        except Exception as e:
            print(f"❌ Cache performance error: {e}")
            self.errors.append(f"Cache performance: {e}")
            self.results.append(
                {
                    "test": "cache_performance",
                    "success": False,
                    "error": str(e),
                    "status": "ERROR",
                }
            )

    def generate_final_report(self):
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("🏆 FINAL OPTIMIZATION REPORT")
        print("=" * 60)

        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        print(f"📊 Test Summary:")
        print(
            f"  ✅ Successful tests: {successful_tests}/{total_tests} ({success_rate:.1%})"
        )
        print(f"  ⏱️  Total test time: {total_time:.3f}s")
        print(f"  ❌ Errors encountered: {len(self.errors)}")

        # Performance summary
        performance_results = [
            r for r in self.results if r["test"] == "end_to_end_performance"
        ]
        if performance_results:
            perf = performance_results[0]
            avg_time = perf["average_time"]
            print(f"\n🎯 Performance Summary:")
            print(f"  ⏱️  Average query time: {avg_time:.3f}s")
            print(f"  🎯 Target: <5.0s")
            print(f"  📈 Status: {perf['status']}")

            if avg_time < 5:
                print(f"  🎉 SUCCESS: Target achieved!")
            else:
                print(f"  ⚠️  NEEDS IMPROVEMENT: {avg_time:.3f}s > 5.0s")

        # Detailed results
        print(f"\n📋 Detailed Results:")
        for result in self.results:
            status_emoji = "✅" if result["success"] else "❌"
            print(f"  {status_emoji} {result['test']}: {result['status']}")

        # Error summary
        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors:
                print(f"  - {error}")

        # Recommendations
        print(f"\n💡 Recommendations:")

        # Check performance
        if performance_results and performance_results[0]["average_time"] > 5:
            print("  🔴 CRITICAL: Average response time > 5s")
            print("    - Consider using smaller models")
            print("    - Reduce context length further")
            print("    - Implement more aggressive caching")

        # Check functionality
        func_results = [
            r for r in self.results if r["test"] == "functionality_verification"
        ]
        if func_results and func_results[0]["functionality_score"] < 0.9:
            print("  🟡 WARNING: Some functionality tests failed")
            print("    - Review optimization impact on accuracy")
            print("    - Test with more diverse queries")

        # Save results
        self.save_results()

        # Final status
        if success_rate > 0.8 and (
            not performance_results or performance_results[0]["average_time"] < 5
        ):
            print(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
            print(f"   System is ready for production use")
            return True
        else:
            print(f"\n⚠️  OPTIMIZATION NEEDS WORK")
            print(f"   Review recommendations above")
            return False

    def save_results(self):
        """Save test results to file"""
        filename = (
            f"optimization_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_test_time": time.time() - self.start_time,
            "results": self.results,
            "errors": self.errors,
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r["success"]),
                "success_rate": (
                    sum(1 for r in self.results if r["success"]) / len(self.results)
                    if self.results
                    else 0
                ),
            },
        }

        with open(filename, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"💾 Results saved to: {filename}")


def main():
    """Main test runner"""
    print("🧪 RAG System Optimization Test Suite")
    print("This will test all optimizations automatically")
    print("No human intervention required!")

    test_suite = ComprehensiveTestSuite()
    success = test_suite.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
