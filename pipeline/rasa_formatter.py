import pandas as pd
import yaml
import ast
from collections import defaultdict

# Path to your QA CSV
qa_path = "labeled_data/hq_qa_pairs.csv"

# Load QA data
qa_df = pd.read_csv(qa_path)
qa_df["question"] = qa_df["question"].apply(lambda q: ast.literal_eval(q) if isinstance(q, str) else [])

def label_intent(sentence):
    s = sentence.lower()

    if any(kw in s for kw in [
        "hackathon", "competition", "team", "best hack", "challenge", "placed", "winners"
    ]):
        return "highlight_achievement"

    elif any(kw in s for kw in [
        "award", "medal", "recognized", "honored", "congratulations", "recipient", "winner", "achievement"
    ]):
        return "recognize_award"

    elif any(kw in s for kw in [
        "grant", "funded", "nsf grant", "research funding", "endowment", "fellowship", "sponsored by"
    ]):
        return "announce_grant"

    elif any(kw in s for kw in [
        "lecture", "colloquium", "seminar", "keynote", "talk", "presentation", "guest speaker"
    ]):
        return "announce_lecture"

    elif any(kw in s for kw in [
        "event", "commencement", "ceremony", "makeathon", "symposium", "conference", "workshop", "open house", "gathering"
    ]):
        return "announce_event"

    elif any(kw in s for kw in [
        "professor", "faculty", "chair", "lecturer", "hired", "joins faculty", "appointed", "retiring", "emeritus"
    ]):
        return "faculty_news"

    elif any(kw in s for kw in [
        "application", "admissions", "apply online", "how to apply", "submission deadline", "accepting applications"
    ]):
        return "ask_admissions"

    elif any(kw in s for kw in [
        "graduate program", "phd", "masters", "m.s.", "program details", "doctoral", "graduate studies", "terminal degree"
    ]):
        return "ask_graduate_program"

    elif any(kw in s for kw in [
        "undergraduate program", "bachelor", "major in", "minor in", "intro to", "bs in", "ba in", "core curriculum"
    ]):
        return "ask_undergraduate_program"

    elif any(kw in s for kw in [
        "internship", "teaching assistant", "research assistant", "library", "museum", "career center", "job board",
        "campus job", "student worker", "apply for position", "employment opportunity"
    ]):
        return "student_opportunities"

    elif any(kw in s for kw in [
        "support us", "donate", "gift", "giving", "fundraising", "alumni support", "development office", "campaign"
    ]):
        return "donation_appeal"

    elif any(kw in s for kw in [
        "contact", "location", "email", "address", "reach out", "visit us", "phone", "office", "directory"
    ]):
        return "ask_contact_info"

    elif any(kw in s for kw in [
        "newsletter", "news update", "press release", "recent news", "media release", "announcements"
    ]):
        return "news_update"

    elif any(kw in s for kw in [
        "course", "curriculum", "major", "minor", "degree", "credit hour", "syllabus", "prerequisite", "enrollment"
    ]):
        return "ask_academic_program"

    elif any(kw in s for kw in [
        "deadline", "due date", "last day", "timeline", "calendar", "schedule", "important dates"
    ]):
        return "ask_timeline"

    elif any(kw in s for kw in [
        "research paper", "publication", "journal", "article", "presented at", "published in"
    ]):
        return "research_publication"

    elif any(kw in s for kw in [
        "student award", "dean's list", "honors student", "merit scholar"
    ]):
        return "student_recognition"

    elif any(kw in s for kw in [
        "tuition", "fees", "cost of attendance", "financial aid", "scholarship", "loan", "payment plan"
    ]):
        return "ask_financial_info"

    elif any(kw in s for kw in [
        "housing", "residence hall", "dorm", "meal plan", "room assignment"
    ]):
        return "ask_campus_life"

    else:
        return "ask_general_question"

intent_buckets = defaultdict(list)

for _, row in qa_df.iterrows():
    for q in row["question"]:
        q_clean = q.strip()
        if len(q_clean) > 5:
            intent = label_intent(q_clean)
            intent_buckets[intent].append(q_clean)

# === (Optional) Step 2: Oversample HQ Examples ===
# You can uncomment and modify this section to oversample HQ pairs
for _ in range(2):  # repeat 2x (for 3x total including original)
    for _, row in qa_df.iterrows():
        for q in row["question"]:
            q_clean = q.strip()
            if len(q_clean) > 5:
                intent = label_intent(q_clean)
                intent_buckets[intent].append(q_clean)

rasa_nlu = {"version": "3.1", "nlu": []}

for intent, examples in intent_buckets.items():
    formatted = "\n".join(f"- {q}" for q in examples)
    rasa_nlu["nlu"].append({
        "intent": intent,
        "examples": formatted
    })

out_path = "nlu_hq_qa.yml"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(rasa_nlu, f, allow_unicode=True, sort_keys=False)

print(f"Rasa NLU file saved to: {out_path}")
