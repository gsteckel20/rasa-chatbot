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
    
    if any(kw in s for kw in ["hackathon", "competition", "team", "best hack"]):
        return "highlight_achievement"
    elif any(kw in s for kw in ["award", "medal", "recognized", "honored", "congratulations"]):
        return "recognize_award"
    elif any(kw in s for kw in ["grant", "funded", "nsf grant", "research funding"]):
        return "announce_grant"
    elif any(kw in s for kw in ["lecture", "colloquium", "seminar", "keynote", "talk", "presentation"]):
        return "announce_lecture"
    elif any(kw in s for kw in ["event", "commencement", "ceremony", "makeathon", "symposium"]):
        return "announce_event"
    elif any(kw in s for kw in ["professor", "faculty", "chair", "lecturer", "hired", "joins faculty"]):
        return "faculty_news"
    elif any(kw in s for kw in ["application", "admissions", "apply online", "how to apply"]):
        return "ask_admissions"
    elif any(kw in s for kw in ["graduate program", "phd", "masters", "m.s.", "program details"]):
        return "ask_graduate_program"
    elif any(kw in s for kw in ["internship", "teaching assistant", "library", "museum", "career center"]):
        return "student_opportunities"
    elif any(kw in s for kw in ["support us", "donate", "gift", "giving", "fundraising"]):
        return "donation_appeal"
    elif any(kw in s for kw in ["contact", "location", "email", "address", "reach out"]):
        return "ask_contact_info"
    elif any(kw in s for kw in ["newsletter", "news update", "press release"]):
        return "news_update"
    elif any(kw in s for kw in ["course", "curriculum", "major", "minor", "degree", "credit hour"]):
        return "ask_academic_program"
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

unknown_samples.to_csv("labeled_data/labeled_training_data_unknowns.csv", index=False)
df_split.to_csv("labeled_data/labeled_training_data.csv", index=False)

# Format for Rasa
rasa_format = {
    "version": "3.1",
    "nlu": []
}

for intent in df_split["intent"].unique():
    examples = df_split[df_split["intent"] == intent]["sentence"].tolist()
    rasa_format["nlu"].append({
        "intent": intent,
        "examples": "\n".join([f"- {ex.strip()}" for ex in examples if ex.strip()])
    })

# Save to YAML
#with open("rasa/nlu.yml", "w", encoding="utf-8") as file:
    #yaml.dump(rasa_format, file, sort_keys=False, allow_unicode=True)

