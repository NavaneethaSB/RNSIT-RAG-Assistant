"""Evaluate answer content and answer-level sources; requires a working Groq API key."""

import unicodedata

from groq import RateLimitError

from api import AskResponse, ConversationTurn, _retrieval_query
from ask import _is_abstention, build_answer_sources, generate_answer, load_resources, retrieve_chunks, select_relevant_chunks


CASES = [
    {"name": "CSE intake", "question": "What is the current CSE intake?", "expected": "720", "url": "/cse/", "source_terms": ["cse", "720"]},
    {"name": "ISE intake", "question": "What is the current ISE intake?", "expected": "240", "url": "/ise/", "source_terms": ["ise", "240"]},
    {"name": "AI and ML intake", "question": "What is the AI and ML department's approved annual intake?", "expected": "360", "url": "/aiml/", "source_terms": ["360"]},
    {"name": "MCA duration", "question": "How long is the MCA programme?", "expected": "two", "url": "/mca/", "source_terms": ["mca", "year"]},
    {"name": "CSE overview", "question": "Give me a short overview of the CSE department.", "expected": "computer science", "url": "/cse/", "source_terms": ["computer", "science"]},
    {"name": "Unsupported", "question": "What is the principal's private phone password?", "url": None},
    {"name": "MCA follow-up", "question": "Tell me more.", "conversation": ["What is the MCA programme?"], "expected": "mca", "url": "/mca/", "source_terms": ["mca"]},
    {"name": "MCA duration follow-up", "question": "What is its duration?", "conversation": ["What is the MCA programme?"], "expected": "two-year", "url": "/mca/", "source_terms": ["mca", "year"]},
]


def _normalize_for_match(text: str) -> str:
    """Normalize typography and whitespace without changing expected words."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(char) == "Pd" or char == "\N{MINUS SIGN}" else char
        for char in normalized
    )
    return " ".join(normalized.split())


def _sources_match_case(case: dict, sources) -> bool:
    """Unsupported cases must not present retrieved pages as supporting evidence."""
    if case["url"] is None:
        return not sources
    terms = [_normalize_for_match(term) for term in case.get("source_terms", [])]
    return any(
        case["url"] in source.url
        and all(term in _normalize_for_match(source.excerpt) for term in terms)
        for source in sources
    )


def main():
    chunks, index, model, bm25, client = load_resources()
    passed = 0
    failed = 0
    skipped = 0
    for case in CASES:
        history = [ConversationTurn(role="user", content=question) for question in case.get("conversation", [])]
        resolved_question = _retrieval_query(case["question"], history)
        retrieved = retrieve_chunks(resolved_question, chunks, index, model, bm25)
        retrieved = select_relevant_chunks(resolved_question, retrieved, chunks)
        try:
            answer = generate_answer(case["question"], retrieved, client, chunks)
        except RateLimitError:
            skipped += 1
            print(f"SKIP {case['name']}: Groq rate limit reached")
            continue
        sources = build_answer_sources(resolved_question, answer, retrieved, chunks)
        payload = AskResponse(answer=answer, sources=sources)
        answer_normalized = _normalize_for_match(payload.answer)
        expected_answer = (
            _is_abstention(payload.answer)
            if case["url"] is None
            else _normalize_for_match(case["expected"]) in answer_normalized
        )
        expected_source = _sources_match_case(case, payload.sources)
        unique_sources = len({source.url for source in payload.sources}) == len(payload.sources)
        ok = expected_answer and expected_source and unique_sources
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{'PASS' if ok else 'FAIL'} {case['name']}: answer={expected_answer}, source={expected_source}, unique_sources={unique_sources}")
        print("  Answer:", payload.answer.replace("\n", " ")[:350])
        for source in payload.sources:
            print(f"  Source {source.id}: {source.title} | {source.url}")
    print(f"Answer cases: {passed} passed, {failed} failed, {skipped} skipped ({len(CASES)} total)")


if __name__ == "__main__":
    main()
