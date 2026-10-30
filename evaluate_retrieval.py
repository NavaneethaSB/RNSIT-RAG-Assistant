"""Evaluate retrieval against page URLs and supported facts, without calling Groq."""

from ask import load_retrieval_resources, retrieve_chunks


CASES = [
    {"name": "CSE intake 720", "question": "Which department has a current intake of 720 students?", "url": "/cse/", "fact": "720"},
    {"name": "ISE intake 240", "question": "Which department has a current intake of 240 students?", "url": "/ise/", "fact": "240"},
    {"name": "AI and ML intake", "question": "What is the approved annual intake of AI and ML?", "url": "/aiml/", "fact": "360"},
    {"name": "MCA duration", "question": "What is the duration of the MCA programme?", "url": "/mca/", "fact": "two-year"},
    {"name": "CSE overview", "question": "Give me an overview of the CSE department.", "url": "/cse/", "fact": "Computer Science"},
    {"name": "MCA duration follow-up", "question": "mca programme duration", "url": "/mca/", "fact": "two-year"},
    {"name": "Unsupported question", "question": "What is the principal's private phone password?", "unsupported": True},
]


def main():
    chunks, index, model, bm25 = load_retrieval_resources()
    passed = 0
    checked = 0
    for case in CASES:
        results = retrieve_chunks(case["question"], chunks, index, model, bm25)
        if case.get("unsupported"):
            print(f"N/A {case['name']}: retrieval has no abstention threshold; answer abstention is checked by evaluate_answers.py (candidates={len(results)})")
            continue
        checked += 1
        relevant_ranks = [
            rank for rank, result in enumerate(results, start=1)
            if case["url"] in result.get("url", "")
            and case["fact"].lower() in result.get("text", "").lower()
        ]
        rank = relevant_ranks[0] if relevant_ranks else None
        ok = rank is not None
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {case['name']}: evidence_rank={rank}")
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. {result.get('title')} | {result.get('url')} | chunk={result.get('chunk_id')}")
    print(f"Retrieval cases: {passed}/{checked} relevant-page cases passed; 1 unsupported case is N/A")


if __name__ == "__main__":
    main()
