import pandas as pd
import yaml
import ast
from collections import defaultdict
import random
import re

# Path to your QA CSV
qa_path = "labeled_data/hq_qa_pairs.csv"

# Load QA data
qa_df = pd.read_csv(qa_path)
qa_df["question"] = qa_df["question"].apply(lambda q: ast.literal_eval(q) if isinstance(q, str) else [])

# === Intent Buckets and Counters ===
intent_buckets = defaultdict(list)
intent_counters = defaultdict(int)

def label_intent(sentence):
    s = sentence.lower()

    if re.search(r"current page|page \d+|last page|service that is not offered", s):
        return "ask_general_question"

    if any(kw in s for kw in [
        "program", "degree", "curriculum", "course", "class", "syllabus", "credit hour",
        "undergraduate", "graduate", "phd", "masters", "bachelor",
        "introduction", "topic", "field", "area", "focus", "study", "subject", "application",
        "research", "project", "thesis", "dissertation", "capstone", "essay exhibit",
        "portfolio", "writing exhibit", "academic advising", "enrollment", "registration",
        "english 1101", "english 1102", "first-year writing", "placement exam", "service learning"
    ]):
        return "academic_info"

    elif any(kw in s for kw in [
        "professor", "faculty", "staff", "lecturer", "chair", "advisor", "mentor",
        "hired", "appointed", "retiring", "emeritus",
        "award", "honor", "recognized", "recipient", "winner", "achievement",
        "student award", "dean's list", "merit scholar", "top student",
        "project", "lab", "research group", "internship", "career center", "student club",
        "team", "chapter", "postings", "director",
        "magazine feature", "journal", "publication", "editor", "news coverage",
        "new tool", "innovation district", "student success office", "career advising"
    ]):
        return "faculty_and_awards"

    elif any(kw in s for kw in [
        "event", "workshop", "conference", "symposium", "seminar", "talk", "colloquium",
        "presentation", "guest speaker", "hackathon", "open house", "lecture", "commencement", "ceremony"
    ]):
        return "event_announcement"

    elif any(kw in s for kw in [
        "tuition", "fees", "financial aid", "scholarship", "loan", "payment plan", "donation", "gift", "fundraising", "grant", "funded"
    ]):
        return "financial_info"

    else:
        return "ask_general_question"

# === Main labeling loop ===
for _, row in qa_df.iterrows():
    for q in row["question"]:
        q_clean = q.strip()
        if len(q_clean) > 5:
            intent = label_intent(q_clean)
            intent_buckets[intent].append(q_clean)
            intent_counters[intent] += 1

# === Balance the dataset ===
TARGET_SAMPLES = 150  # Change this if you want a different target size

balanced_intent_buckets = {}

for intent, examples in intent_buckets.items():
    n = len(examples)
    if n > TARGET_SAMPLES:
        balanced_intent_buckets[intent] = random.sample(examples, TARGET_SAMPLES)
    elif n < TARGET_SAMPLES:
        multiplied = examples * (TARGET_SAMPLES // n) + random.sample(examples, TARGET_SAMPLES % n)
        balanced_intent_buckets[intent] = multiplied
    else:
        balanced_intent_buckets[intent] = examples

# === Save Rasa NLU YAML ===
rasa_nlu = {"version": "3.1", "nlu": []}

for intent, examples in balanced_intent_buckets.items():
    formatted = "\n".join(f"- {q}" for q in examples)
    rasa_nlu["nlu"].append({
        "intent": intent,
        "examples": formatted
    })

out_path = "nlu_hq_qa_balanced.yml"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(rasa_nlu, f, allow_unicode=True, sort_keys=False)


# === Print intent counts ===
print("\nBalanced Intent Distribution:")
total_balanced = sum(len(v) for v in balanced_intent_buckets.values())
for intent, examples in sorted(balanced_intent_buckets.items(), key=lambda x: -len(x[1])):
    percent = (len(examples) / total_balanced) * 100
    print(f"{intent:30s}: {len(examples)} ({percent:.2f}%)")

print(f"\nTotal balanced sentences: {total_balanced}")

