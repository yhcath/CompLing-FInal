import csv
import random

rows = []

with open('sentiment_nonrandomized.tsv') as tsv_file:
    for row in csv.reader(tsv_file, delimiter="\t"):
        rows.append(row)

random.shuffle(rows)
clean_csv = "id\tsentiment\ttweet\n"
for row in rows:
    clean_csv += f"{row[0]}\t{row[1]}\t{row[2]}\n"

with open("sentiment_randomized.csv", "w") as result_file:
    result_file.write(clean_csv)

print("Done!")