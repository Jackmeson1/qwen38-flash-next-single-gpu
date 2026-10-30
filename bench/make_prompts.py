"""Build the prompt files from public-domain Project Gutenberg books, cut to exact token counts with the model's
tokenizer. Every prompt starts with its own passage, so no two requests share a prefix beyond the chat template.

  python make_prompts.py --tokenizer models/w4a16 --out prompts
"""
import argparse
import json
import os
import random
import re
import urllib.request

BOOKS = [2600, 135, 1184, 2701, 996, 145, 1023, 766, 1399, 28054, 1342, 98, 1400, 4300, 2554, 1260, 768, 2610, 1257,
         84, 345, 174, 44, 158, 120, 36, 6130, 3207, 8800, 1727]
INSTRUCTIONS = [
    "Write a long, detailed literary analysis of the passage above: its themes, characters, style and what it suggests about the rest of the book.",
    "Retell the passage above in your own words, then continue the story for several more pages in the same style.",
    "Write a detailed study guide for the passage above, with a summary, a list of characters, key quotes explained, and ten discussion questions with model answers.",
    "Explain the passage above to a modern reader in depth: historical background, difficult words, and how each paragraph fits the plot.",
    "Write a long essay comparing the passage above with modern fiction, quoting the passage often.",
    "Rewrite the passage above as a screenplay scene with detailed stage directions, then explain your choices at length.",
]
QUESTIONS = [
    "Who are the people named in the document, and what does each of them do?",
    "Summarize the document in ten bullet points.",
    "What places are mentioned in the document, and what happens in each?",
    "List every number or date that appears in the document and explain its context.",
    "What is the mood of the document, and which sentences create it?",
    "Write five quiz questions about the document, with answers.",
    "Which character changes the most in the document, and how?",
    "Explain three words from the document a modern reader might not know.",
]


def book_text(cache, bid):
    path = os.path.join(cache, f"pg{bid}.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt", path)
    t = open(path, encoding="utf-8-sig", errors="replace").read()
    a = re.search(r"\*\*\* ?START OF.*?\*\*\*", t)
    b = re.search(r"\*\*\* ?END OF", t)
    t = t[a.end() if a else 0:b.start() if b else len(t)]
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="gutenberg", help="where the downloaded books are kept")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    ids = []
    for bid in BOOKS:
        ids.extend(tok.encode(book_text(args.cache, bid) + "\n\n", add_special_tokens=False))
    print("corpus tokens", len(ids))
    pos = 0

    def take(n):
        # Walk the corpus; on wrap-around shift the start so passages cut differently. Every prompt also gets a
        # unique first line, so no two requests share a prefix beyond the chat template.
        nonlocal pos
        if pos + n > len(ids):
            pos = (pos + n) % 997
        piece = tok.decode(ids[pos:pos + n])
        pos += n
        return piece

    rng = random.Random(0)
    # (name, passage tokens, count, max_tokens). Passage + instruction ~= the target prompt length.
    specs = [("w1_1k", 950, 60, 512), ("w1_8k", 8090, 60, 512), ("w1_32k", 32590, 60, 512), ("w2", 950, 9000, 512)]
    for name, n, count, mt in specs:
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for i in range(count):
                msg = f"Excerpt {name}-{i}:\n\n" + take(n) + "\n\n" + rng.choice(INSTRUCTIONS)
                fh.write(json.dumps({"id": f"{name}-{i}", "messages": [{"role": "user", "content": msg}],
                                     "max_tokens": mt}) + "\n")
    doc = take(8100)
    with open(os.path.join(args.out, "w4.jsonl"), "w", encoding="utf-8") as fh:
        for i in range(48):
            q = QUESTIONS[i % len(QUESTIONS)] + f" (Answer {i // len(QUESTIONS) + 1}: be thorough.)"
            fh.write(json.dumps({"id": f"w4-{i}", "messages": [{"role": "user", "content": doc + "\n\n" + q}],
                                 "max_tokens": 256}) + "\n")
    print("used tokens", pos)


if __name__ == "__main__":
    main()
