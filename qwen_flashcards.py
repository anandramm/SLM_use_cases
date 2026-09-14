"""Offline Qwen3 flashcards. Requires only your existing llama-cpp-python.

Put Qwen3-1.7B-Q8_0.gguf beside this script. Edit STUDY_NOTES below,
then run: python qwen_flashcards.py
One nonempty line = one self-contained fact. Answers are copied from notes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Qwen3-1.7B-Q8_0.gguf"

STUDY_NOTES = """
PROVIDE YOUR STUDY NOTES HERE 
"""
MAX_CARDS = 5                 
CONTEXT_TOKENS = 4_096
QUESTION_TOKENS = 64          


def read_facts(notes: str) -> list[tuple[int, str]]:
    """Keep original line numbers and reject empty or oversized study notes."""
    if MAX_CARDS < 1:
        raise ValueError("MAX_CARDS must be at least 1.")
    facts = [(number, line.strip())
             for number, line in enumerate(notes.splitlines(), start=1)
             if line.strip()]
    if not facts:
        raise ValueError("Add some facts to STUDY_NOTES first.")
    selected = facts[:MAX_CARDS]
    for number, fact in selected:
        if len(fact) > 1_500:
            raise ValueError(f"Note line {number} is too long; use a short fact.")
        if any(marker in fact for marker in ("<|", "<think>", "</think>")):
            raise ValueError(f"Remove model control markers from note line {number}.")
    print(f"Using {len(selected)} of {len(facts)} note lines.", flush=True)
    return selected


def load_model():
    """Load once for generation. Reviewing an existing deck skips this."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Put the model beside this script:\n{MODEL_PATH}")
    from llama_cpp import Llama

    threads = max(1, min(4, (os.cpu_count() or 2) - 1))
    print(f"Loading Qwen with {threads} CPU threads ...", flush=True)
    return Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=0,
        n_ctx=CONTEXT_TOKENS,
        n_batch=128,
        n_ubatch=128,
        n_threads=threads,
        n_threads_batch=threads,
        use_mmap=True,
        use_mlock=False,
        logits_all=False,
        embedding=False,
        verbose=False,
        seed=42,
    )


def build_prompt(fact: str) -> str:
    """Format a Qwen3 conversation with an already completed thinking block."""
    instructions = (
    "Write ONE short study question answered directly by the supplied fact. "
    "The answer must be explicitly stated in the fact. "
    "Do not infer or invent causes, purposes, locations, ownership, chronology, "
    "relationships, or meanings. Preserve the exact names and nouns from the fact. "
    "If the fact says something is unknown or gives no information, return SKIP. "
    "If the question cannot be answered using only the fact, return SKIP. "
    "Return only one question on one line, ending with a question mark. "
    "Do not include the answer."
    )
    return (
        f"<|im_start|>system\n{instructions}<|im_end|>\n"
        f"<|im_start|>user\nFACT:\n{fact}\n/no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def generate_question(model, fact: str) -> str | None:
    """Generate one question; reject malformed or truncated output."""
    prompt = build_prompt(fact)
    # Count with the same special-token handling used for inference.
    tokens = model.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    if len(tokens) + QUESTION_TOKENS + 16 > CONTEXT_TOKENS:
        raise ValueError("This fact exceeds the context budget; shorten it.")

    response = model.create_completion(
        prompt=tokens,
        max_tokens=QUESTION_TOKENS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"],
        echo=False,
    )
    choice = response["choices"][0]
    question = choice["text"].strip()
    if choice.get("finish_reason") == "length":
        return None
    if (question.upper() == "SKIP" or not question.endswith("?")
            or question.count("?") != 1 or len(question) > 300
            or "\n" in question or "<" in question):
        return None
    return question


def make_deck(model, facts: list[tuple[int, str]]) -> list[dict]:
    """Let Qwen write questions; Python supplies every answer verbatim."""
    cards = []
    seen_questions = set()
    for number, (source_line, fact) in enumerate(facts, start=1):
        print(f"Writing question {number}/{len(facts)} ...", flush=True)
        started = perf_counter()
        question = generate_question(model, fact)
        print(f"Finished in {perf_counter() - started:.1f} seconds.", flush=True)
        if question is None or question.casefold() in seen_questions:
            print("Skipped an invalid or duplicate question; no automatic retry.")
            continue
        seen_questions.add(question.casefold())
        cards.append({
            "question": question,
            "answer": fact,           
            "source_quote": fact,    
            "source_line": source_line,
            "question_reviewed": False,
        })
    return cards


def save_deck(cards: list[dict]) -> Path:
    """Save a separate timestamped deck; leave earlier study sessions intact."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = PROJECT_DIR / f"flashcards_{stamp}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(cards, handle, ensure_ascii=False, indent=2)
    return path


def read_deck(path: Path) -> list[dict]:
    """Validate a saved deck before displaying it."""
    cards = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cards, list) or not cards:
        raise ValueError("The saved deck must contain a nonempty list of cards.")
    for card in cards:
        if (not isinstance(card, dict)
                or not isinstance(card.get("question"), str)
                or not card["question"].strip()
                or not isinstance(card.get("answer"), str)
                or not card["answer"].strip()
                or card["answer"] != card.get("source_quote")
                or type(card.get("source_line")) is not int):
            raise ValueError("A saved card has missing or inconsistent fields.")
    return cards


def practise(cards: list[dict]) -> None:
    """Reveal answers after recall. The learner judges correctness."""
    print("\nTry answering aloud or mentally, then reveal the answer.")
    print("Questions are drafts: check that each actually matches its source.")
    for number, card in enumerate(cards, start=1):
        print(f"\nCard {number}/{len(cards)}: {card['question']}")
        if input("Enter: reveal answer | q: quit > ").strip().lower() == "q":
            break
        print(f"Answer (copied from note line {card['source_line']}):")
        print(card["answer"])
        if input("Enter: next card | q: quit > ").strip().lower() == "q":
            break


def main() -> None:
    existing = sorted(PROJECT_DIR.glob("flashcards_*.json"))
    if existing:
        print(f"Latest saved deck: {existing[-1].name}")
        action = input("Enter: practise saved deck | g: generate new | q: quit > ")
        if action.strip().lower() == "q":
            return
        if action.strip().lower() != "g":
            practise(read_deck(existing[-1]))
            return

    facts = read_facts(STUDY_NOTES)
    model = load_model()
    try:
        cards = make_deck(model, facts)
    finally:
        model.close() 
    if not cards:
        print("No usable questions were generated. Try clearer, shorter facts.")
        return
    path = save_deck(cards)
    print(f"Saved {len(cards)} draft flashcards to {path}")
    practise(cards)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStudy session ended.")
