from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import glob
import nltk
from nltk.tokenize import sent_tokenize

model_name = "allenai/t5-small-squad2-question-generation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def chunk_text(text):
    return sent_tokenize(text)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output

csv_files = glob.glob("cleaned_data_csv/*.csv")

print(csv_files)

qa_pairs = []

for f in csv_files:
    fdf = pd.read_csv(f, escapechar='\\')
    #fdf.to_csv("labeled_data/whole_file.csv", index=False)
    fdf = fdf.dropna(subset=['cleaned_text'])
    #fdf.to_csv("labeled_data/whole_file_cleaned.csv", index=False)
    for _, row in fdf.iterrows():
        sentences = chunk_text(row["cleaned_text"])
        for sentence in sentences:
            try:
                question = run_model(sentence)
                qa_pairs.append({
                    "question": question,
                    "answer": sentence
                })
            except Exception as e:
                print(f"Error generating question: {e}")
                continue


qa_df = pd.DataFrame(qa_pairs)

# Save to CSV
qa_df.to_csv("labeled_data/qa_pairs_2.csv", index=False)