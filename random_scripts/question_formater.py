import csv

input_file = "labeled_data/Chatbot_DataSet.csv"   # Your original CSV file
output_file = "labeled_data/Chatbot_DataSet_formated.csv"  # Output with modified question formatting

with open(input_file, mode="r", newline="", encoding="utf-8") as infile, \
     open(output_file, mode="w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        if len(row) >= 2:
            question = row[0].strip()
            answer = row[1].strip()
            formatted_question = f"['{question}']"
            writer.writerow([formatted_question, answer])

print(f"File saved as '{output_file}'")