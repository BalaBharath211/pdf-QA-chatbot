#!/usr/bin/env python3
"""
Quick Performance Check - Get metrics in under 2 minutes
"""

import time
import psutil
import os
from rag_core import build_vector_store, answer_query


def quick_test():
    """Quick test to get basic performance numbers"""
    
    print("\n" + "="*60)
    print("⚡ QUICK PERFORMANCE CHECK")
    print("="*60)
    
    # Ask user for PDF
    print("\n📁 Looking for PDFs in data/ folder...")
    pdf_dir = "data"
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"❌ No PDFs found! Please add PDF files to the '{pdf_dir}/' folder")
        return
    
    pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdfs:
        print(f"❌ No PDFs found in '{pdf_dir}/' folder!")
        print("   Add at least one PDF file and run again.")
        return
    
    print(f"✅ Found {len(pdfs)} PDF(s):")
    for pdf in pdfs:
        size_mb = os.path.getsize(pdf) / 1024 / 1024
        print(f"   - {os.path.basename(pdf)} ({size_mb:.1f} MB)")
    
    # Test 1: Processing Speed
    print("\n" + "-"*60)
    print("📊 TEST 1: Document Processing Speed")
    print("-"*60)
    
    start = time.time()
    vector_store = build_vector_store(pdfs[:1])  # Test with first PDF only
    processing_time = time.time() - start
    
    file_size = os.path.getsize(pdfs[0]) / 1024 / 1024
    print(f"✅ Processed {os.path.basename(pdfs[0])}")
    print(f"   Time: {processing_time:.2f} seconds")
    print(f"   Speed: {file_size/processing_time:.2f} MB/s")
    
    # Test 2: Query Response Time
    print("\n" + "-"*60)
    print("📊 TEST 2: Query Response Time")
    print("-"*60)
    
    test_queries = [
        "What is the main topic of this document?",
        "Summarize the key points",
        "What are the important concepts mentioned?"
    ]
    
    times = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        start = time.time()
        
        try:
            answer, sources = answer_query(vector_store, query)
            elapsed = time.time() - start
            times.append(elapsed)
            
            print(f"✅ Response time: {elapsed:.2f}s")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Sources found: {len(sources)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average response time: {avg_time:.2f}s")
    
    # Test 3: Memory Usage
    print("\n" + "-"*60)
    print("📊 TEST 3: Memory Usage")
    print("-"*60)
    
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {mem_mb:.2f} MB ({mem_mb/1024:.2f} GB)")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY FOR README")
    print("="*60)
    
    print(f"""
Update your README with these actual numbers:

**Query Response Time:** {avg_time:.1f} seconds (avg)
**Processing Speed:** ~{file_size/processing_time:.1f}s per {file_size:.0f}MB document
**Memory Usage:** ~{mem_mb/1024:.1f}GB

You can also say:
- Response time range: {min(times):.1f}-{max(times):.1f} seconds
- Processing speed: {file_size/processing_time:.2f} MB/s
""")
    
    print("="*60)
    print("✅ Quick test complete!")
    print("="*60)
    print("\nFor detailed testing, run: python test_performance.py")


if __name__ == "__main__":
    quick_test()
