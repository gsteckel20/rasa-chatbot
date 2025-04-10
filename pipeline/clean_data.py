from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, coalesce, lit
from pyspark.sql.types import StringType
from bs4 import BeautifulSoup
import re

spark = SparkSession.builder.appName("CleanTextPipeline").getOrCreate()


df = spark.read.json("./scrapy/output/*.json")

# Use the first non-null value among possible text fields
df = df.withColumn("raw_text", coalesce(
    col("body"),
    col("description"),
    col("title"),
    lit("")
))

def clean_text(text):
    if not text:
        return ""

    # Lowercase the text
    text = text.lower()

    # List of common noise phrases to remove
    noise_phrases = [
        "skip to main content", "skip to main menu", "search this site", "give now", 
        "main menu", "submit search", "close", "school's twitter feed", 
        "school's youtube channel", "school's linkedin page", "© university of georgia",
        "facebook", "twitter", "instagram", "snapchat", "youtube", "linked in",
        "human trafficking notice", "reporting hotline", "privacy policy", "login for faculty",
        "skip to spotlight region", "skip to secondary region", "skip to uga region", "skip to tertiary region",
        "skip to quaternary region", "skip to unit footer"
    ]

    for phrase in noise_phrases:
        text = text.replace(phrase, "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    text = re.split(r"school of computing 415 boyd research and education center university of georgia athens, ga 30602-7404 , staff and students", text)[0]
    return text

clean_udf = udf(clean_text, StringType())

# Apply initial cleaning
df_cleaned = df.withColumn("cleaned_text", clean_udf(col("raw_text")))

#Detect repeated prefix and suffix across multiple pages
sample_texts = [row.cleaned_text for row in df_cleaned.select("cleaned_text").limit(50).collect() if row.cleaned_text]
def find_common_prefix(texts):
    if not texts:
        return ""
    first = texts[0]
    for i in range(len(first)):
        prefix = first[:i+1]
        if not all(text.startswith(prefix) for text in texts):
            return first[:i] if i > 20 else ""  # Only if it's meaningfully long
    return first

common_prefix = find_common_prefix(sample_texts)

#Remove repeated prefix and suffix
def remove_common_prefix(text):
    if text and common_prefix and text.startswith(common_prefix):
        return text[len(common_prefix):].strip()
    return text or ""

remove_prefix_udf = udf(remove_common_prefix, StringType())

df_cleaned = df_cleaned.withColumn("cleaned_text", remove_prefix_udf(col("cleaned_text")))

#Select and save
df_cleaned = df_cleaned.select(
    coalesce(col("url"), lit("unknown")).alias("source"),
    col("cleaned_text")
)

df_cleaned.write.mode("overwrite").option("header", True).option("quoteAll", True).csv("cleaned_data_csv")