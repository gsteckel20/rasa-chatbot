import pandas as pd
import yaml
import ast

qa_df = pd.read_csv("labeled_data/Chatbot_DataSet_formated.csv")

qa_df["question"] = qa_df["question"].apply(lambda q: ast.literal_eval(q) if isinstance(q, str) else [])

qa_dict = {}
for _, row in qa_df.iterrows():
    answer = row["answer"].strip()
    for q in row["question"]:
        qa_dict.setdefault(answer, []).append(q.strip())

# Load existing nlu.yml
nlu_file = "rasa/nlu.yml"
with open(nlu_file, "r") as f:
    nlu_data = yaml.safe_load(f)

if "nlu" not in nlu_data:
    nlu_data["nlu"] = []

# Add QA as a new intent block
for idx, (answer, questions) in enumerate(qa_dict.items()):
    intent_name = f"inform_qa_{idx+1}"
    examples = "\n    - " + "\n    - ".join(questions)
    intent_block = {
        "intent": intent_name,
        "examples": examples
    }
    nlu_data["nlu"].append(intent_block)

# Save back to nlu.yml (append style)
with open(nlu_file, "w") as f:
    yaml.dump(nlu_data, f, allow_unicode=True, sort_keys=False)

print(f"✅ Appended {len(qa_dict)} QA intents to {nlu_file}")