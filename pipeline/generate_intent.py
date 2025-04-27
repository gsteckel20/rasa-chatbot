from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import when
import yaml
import pandas as pd
from nltk.tokenize import sent_tokenize

spark = SparkSession.builder.appName("IntentLabeling").getOrCreate()

def chunk_text(text):
    return sent_tokenize(text)

# Load the cleaned CSVs
df = spark.read.option("header", True).csv("cleaned_data_csv/*.csv")
df = df.filter(col("cleaned_text").isNotNull())

data_pd = df.select("cleaned_text").toPandas()

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
        return "unknown"

    
split_rows = []
for _, row in data_pd.iterrows():
    sentences = sent_tokenize(row["cleaned_text"])
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:
            split_rows.append({
                "sentence": sentence,
                "intent": label_intent(sentence)
            })

df_split = pd.DataFrame(split_rows)
print(df_split.head())

unknown_samples = df_split[df_split["intent"] == "unknown"]
labeled_samples = df_split[df_split["intent"] != "unknown"]

unknown_samples.to_csv("labeled_data/labeled_training_data_unknowns.csv", index=False)
labeled_samples.to_csv("labeled_data/labeled_training_data_known.csv", index=False)

# Format for Rasa
rasa_format = {
    "version": "3.1",
    "nlu": []
}

for intent in labeled_samples["intent"].unique():
    examples = labeled_samples[labeled_samples["intent"] == intent]["sentence"].tolist()
    rasa_format["nlu"].append({
        "intent": intent,
        "examples": "\n".join([f"- {ex.strip()}" for ex in examples if ex.strip()])
    })

# Save to YAML
with open("rasa/general_nlu.yml", "w", encoding="utf-8") as file:
    yaml.dump(rasa_format, file, sort_keys=False, allow_unicode=True)

