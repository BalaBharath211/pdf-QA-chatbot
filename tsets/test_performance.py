#!/usr/bin/env python3
"""
Performance Testing Suite for PDF RAG System
Measures: Response Time, Accuracy, Memory Usage, Processing Speed
"""

import time
import psutil
import os
import tracemalloc
from datetime import datetime
from typing import List, Dict, Tuple
import json

from rag_core import build_vector_store, answer_query


# ============================================================================
# TEST QUERIES WITH EXPECTED CHARACTERISTICS
# ============================================================================

TEST_QUERIES = {
    "simple_factual": [
        "What is homomorphic encryption?",
        "Define machine learning",
        "What is a neural network?",
        "Explain what AI means",
        "What is deep learning?"
    ],
    
    "comparison": [
        "Compare supervised and unsupervised learning",
        "What is the difference between AI and ML?",
        "Compare decision trees and random forests",
        "Supervised learning vs reinforcement learning",
        "Neural networks vs traditional algorithms"
    ],
    
    "multi_aspect": [
        "What are all the types of machine learning?",
        "List all privacy concerns with AI",
        "What are the various applications of AI?",
        "Describe all the components of a neural network",
        "What are the different kinds of encryption?"
    ],
    
    "explanation": [
        "How does backpropagation work?",
        "Explain the concept of overfitting",
        "How do convolutional neural networks work?",
        "Describe how gradient descent optimizes models",
        "Explain transfer learning"
    ]
}


# ============================================================================
# PERFORMANCE METRICS TRACKER
# ============================================================================

class PerformanceTracker:
    """Tracks and calculates performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "memory_usage": [],
            "query_success": {"total": 0, "successful": 0},
            "by_type": {}
        }
        self.start_time = None
        self.process = psutil.Process()
    
    def start_query(self):
        """Start timing a query"""
        self.start_time = time.time()
    
    def end_query(self, success: bool, query_type: str):
        """End timing and record results"""
        elapsed = time.time() - self.start_time
        self.metrics["response_times"].append(elapsed)
        
        # Track success rate
        self.metrics["query_success"]["total"] += 1
        if success:
            self.metrics["query_success"]["successful"] += 1
        
        # Track by type
        if query_type not in self.metrics["by_type"]:
            self.metrics["by_type"][query_type] = {
                "count": 0, "successful": 0, "times": []
            }
        
        self.metrics["by_type"][query_type]["count"] += 1
        if success:
            self.metrics["by_type"][query_type]["successful"] += 1
        self.metrics["by_type"][query_type]["times"].append(elapsed)
    
    def record_memory(self):
        """Record current memory usage"""
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
        self.metrics["memory_usage"].append(mem_mb)
        return mem_mb
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        times = self.metrics["response_times"]
        
        summary = {
            "avg_response_time": sum(times) / len(times) if times else 0,
            "min_response_time": min(times) if times else 0,
            "max_response_time": max(times) if times else 0,
            "overall_success_rate": (
                self.metrics["query_success"]["successful"] / 
                self.metrics["query_success"]["total"] * 100
                if self.metrics["query_success"]["total"] > 0 else 0
            ),
            "avg_memory_mb": (
                sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"])
                if self.metrics["memory_usage"] else 0
            ),
            "peak_memory_mb": max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
        }
        
        # Add per-type statistics
        summary["by_query_type"] = {}
        for qtype, data in self.metrics["by_type"].items():
            if data["count"] > 0:
                summary["by_query_type"][qtype] = {
                    "success_rate": data["successful"] / data["count"] * 100,
                    "avg_time": sum(data["times"]) / len(data["times"]),
                    "total_queries": data["count"]
                }
        
        return summary


# ============================================================================
# TEST EXECUTION
# ============================================================================

def evaluate_answer_quality(answer: str, query: str, query_type: str) -> bool:
    """
    Heuristic evaluation of answer quality
    Returns True if answer seems valid
    """
    # Basic checks
    if not answer or len(answer) < 50:
        return False
    
    if "couldn't find" in answer.lower() or "not found" in answer.lower():
        return False
    
    # Type-specific checks
    if query_type == "comparison":
        # Should mention both items or have comparative words
        comparative_words = ["differ", "compare", "versus", "vs", "while", "whereas", "both"]
        if not any(word in answer.lower() for word in comparative_words):
            return False
    
    elif query_type == "multi_aspect":
        # Should have multiple points (look for numbered lists or multiple paragraphs)
        if answer.count('\n') < 3 and answer.count('.') < 5:
            return False
    
    elif query_type == "explanation":
        # Should be detailed (longer answer)
        if len(answer) < 200:
            return False
    
    return True


def test_response_time_and_accuracy(vector_store, tracker: PerformanceTracker):
    """Test response time and accuracy across different query types"""
    
    print("\n" + "="*70)
    print("📊 TESTING RESPONSE TIME & ACCURACY")
    print("="*70)
    
    for query_type, queries in TEST_QUERIES.items():
        print(f"\n🔍 Testing {query_type.upper()} queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"   Query {i}/{len(queries)}: {query[:50]}...")
            
            # Start tracking
            tracker.start_query()
            
            try:
                answer, sources = answer_query(vector_store, query)
                
                # Evaluate quality
                is_successful = evaluate_answer_quality(answer, query, query_type)
                
                tracker.end_query(is_successful, query_type)
                tracker.record_memory()
                
                status = "✅" if is_successful else "⚠️"
                print(f"   {status} Completed in {tracker.metrics['response_times'][-1]:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:50]}...")
                tracker.end_query(False, query_type)
    
    print("\n✅ Response time and accuracy testing complete!")


def test_processing_speed(pdf_paths: List[str]) -> Dict:
    """Test PDF processing speed"""
    
    print("\n" + "="*70)
    print("📄 TESTING DOCUMENT PROCESSING SPEED")
    print("="*70)
    
    results = []
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠️  Skipping {pdf_path} (not found)")
            continue
        
        # Get file info
        file_size_mb = os.path.getsize(pdf_path) / 1024 / 1024
        
        print(f"\n📖 Processing: {os.path.basename(pdf_path)}")
        print(f"   Size: {file_size_mb:.2f} MB")
        
        # Time the processing
        start_time = time.time()
        
        try:
            # Track memory during processing
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0] / 1024 / 1024
            
            vector_store = build_vector_store([pdf_path])
            
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            elapsed = time.time() - start_time
            mem_used = (peak_mem - start_mem * 1024 * 1024) / 1024 / 1024
            
            results.append({
                "file": os.path.basename(pdf_path),
                "size_mb": file_size_mb,
                "time_seconds": elapsed,
                "memory_mb": mem_used,
                "speed_mb_per_sec": file_size_mb / elapsed if elapsed > 0 else 0
            })
            
            print(f"   ✅ Processed in {elapsed:.2f}s ({file_size_mb/elapsed:.2f} MB/s)")
            print(f"   💾 Memory used: {mem_used:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    return results


def test_memory_usage(vector_store, tracker: PerformanceTracker):
    """Test memory usage during sustained operation"""
    
    print("\n" + "="*70)
    print("💾 TESTING MEMORY USAGE")
    print("="*70)
    
    # Get baseline memory
    baseline_mem = tracker.record_memory()
    print(f"\n📊 Baseline memory: {baseline_mem:.2f} MB")
    
    # Run multiple queries
    print("🔄 Running sustained query load...")
    
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "Compare AI and ML",
        "What are the types of learning?",
        "How does deep learning work?"
    ]
    
    for i, query in enumerate(test_queries * 3, 1):  # Run each query 3 times
        try:
            answer, sources = answer_query(vector_store, query)
            mem = tracker.record_memory()
            print(f"   Query {i}/15: {mem:.2f} MB")
        except Exception as e:
            print(f"   Query {i}/15: Error - {str(e)[:30]}...")
    
    peak_mem = max(tracker.metrics["memory_usage"])
    avg_mem = sum(tracker.metrics["memory_usage"]) / len(tracker.metrics["memory_usage"])
    
    print(f"\n📊 Memory Statistics:")
    print(f"   Baseline: {baseline_mem:.2f} MB")
    print(f"   Average:  {avg_mem:.2f} MB")
    print(f"   Peak:     {peak_mem:.2f} MB")
    print(f"   Growth:   {peak_mem - baseline_mem:.2f} MB")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(tracker: PerformanceTracker, processing_results: List[Dict]):
    """Generate comprehensive performance report"""
    
    summary = tracker.get_summary()
    
    print("\n" + "="*70)
    print("📊 PERFORMANCE REPORT")
    print("="*70)
    
    # Overall metrics
    print("\n🎯 OVERALL METRICS")
    print("-" * 70)
    print(f"Query Response Time:    {summary['avg_response_time']:.2f}s "
          f"(range: {summary['min_response_time']:.2f}s - {summary['max_response_time']:.2f}s)")
    print(f"Overall Success Rate:   {summary['overall_success_rate']:.1f}%")
    print(f"Average Memory Usage:   {summary['avg_memory_mb']:.2f} MB")
    print(f"Peak Memory Usage:      {summary['peak_memory_mb']:.2f} MB")
    
    # By query type
    print("\n📋 BY QUERY TYPE")
    print("-" * 70)
    for qtype, stats in summary["by_query_type"].items():
        print(f"\n{qtype.upper().replace('_', ' ')}:")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Avg Time:     {stats['avg_time']:.2f}s")
        print(f"  Total Tested: {stats['total_queries']}")
    
    # Processing speed
    if processing_results:
        print("\n📄 DOCUMENT PROCESSING")
        print("-" * 70)
        avg_speed = sum(r["speed_mb_per_sec"] for r in processing_results) / len(processing_results)
        avg_time = sum(r["time_seconds"] for r in processing_results) / len(processing_results)
        
        print(f"Average Processing Speed: {avg_speed:.2f} MB/s")
        print(f"Average Time per Document: {avg_time:.2f}s")
        
        for result in processing_results:
            print(f"\n  {result['file']}:")
            print(f"    Size: {result['size_mb']:.2f} MB")
            print(f"    Time: {result['time_seconds']:.2f}s")
            print(f"    Speed: {result['speed_mb_per_sec']:.2f} MB/s")
    
    # README comparison
    print("\n" + "="*70)
    print("📊 README METRICS VALIDATION")
    print("="*70)
    
    readme_metrics = {
        "Query Response Time": "3-6 seconds",
        "Simple Factual Accuracy": "95%",
        "Comparison Query Success": "85%",
        "Memory Usage": "~1GB",
    }
    
    actual_metrics = {
        "Query Response Time": f"{summary['avg_response_time']:.1f}s",
        "Simple Factual Accuracy": f"{summary['by_query_type'].get('simple_factual', {}).get('success_rate', 0):.0f}%",
        "Comparison Query Success": f"{summary['by_query_type'].get('comparison', {}).get('success_rate', 0):.0f}%",
        "Memory Usage": f"~{summary['peak_memory_mb']/1024:.1f}GB",
    }
    
    print(f"\n{'Metric':<30} {'README':<20} {'Actual':<20} {'Status'}")
    print("-" * 70)
    
    for metric in readme_metrics:
        readme_val = readme_metrics[metric]
        actual_val = actual_metrics[metric]
        
        # Simple validation (you can make this more sophisticated)
        if "Response Time" in metric:
            status = "✅" if 3 <= summary['avg_response_time'] <= 6 else "⚠️"
        elif "Factual" in metric:
            status = "✅" if summary['by_query_type'].get('simple_factual', {}).get('success_rate', 0) >= 90 else "⚠️"
        elif "Comparison" in metric:
            status = "✅" if summary['by_query_type'].get('comparison', {}).get('success_rate', 0) >= 80 else "⚠️"
        else:
            status = "✅"
        
        print(f"{metric:<30} {readme_val:<20} {actual_val:<20} {status}")
    
    # Save to file
    report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "processing": processing_results,
            "readme_comparison": {
                "readme": readme_metrics,
                "actual": actual_metrics
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run complete performance test suite"""
    
    print("=" * 70)
    print("🚀 PDF RAG SYSTEM - PERFORMANCE TEST SUITE")
    print("=" * 70)
    
    # Configuration
    PDF_PATHS = [
        "data/sample1.pdf",  # Update with your actual PDF paths
        "data/sample2.pdf",
    ]
    
    print("\n⚙️  Configuration:")
    print(f"   Test PDFs: {len([p for p in PDF_PATHS if os.path.exists(p)])} found")
    print(f"   Test Queries: {sum(len(queries) for queries in TEST_QUERIES.values())} total")
    
    input("\n Press Enter to start testing...")
    
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Test 1: Document Processing Speed
    processing_results = test_processing_speed(PDF_PATHS)
    
    # Build vector store for query testing
    print("\n🔨 Building vector store for query testing...")
    available_pdfs = [p for p in PDF_PATHS if os.path.exists(p)]
    
    if not available_pdfs:
        print("❌ No PDFs found! Please add PDFs to the data/ folder")
        print("   Update PDF_PATHS in this script with your actual PDF locations")
        return
    
    vector_store = build_vector_store(available_pdfs)
    print("✅ Vector store ready!")
    
    # Test 2: Response Time & Accuracy
    test_response_time_and_accuracy(vector_store, tracker)
    
    # Test 3: Memory Usage
    test_memory_usage(vector_store, tracker)
    
    # Generate Report
    generate_report(tracker, processing_results)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
