"""Practical 2

Greatly inspiread by Stanford CS224 2019 class.
"""
import csv
from datasets import load_dataset
from nltk.tokenize import word_tokenize

_MIN_LENGTH = 10
_PATH = "datasets/open_subtitles_pl_georgia/"

data = load_dataset("open_subtitles", lang1="ka", lang2="pl")
new_dataset = []
tokens = set()

for idx, example in enumerate(data["train"]):
    if len(example["translation"]["pl"].split()) > _MIN_LENGTH:
        tokenized = word_tokenize(
            example["translation"]["pl"], language="polish")
        tokens.update(tokenized)
        new_dataset.append(" ".join(tokenized))


print("Number of sentences: ", len(new_dataset))
print("Number of tokens: ", len(tokens))


with open(_PATH + "datasetSentences.txt", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["sentence_index", "sentence"])
    for index, row in enumerate(new_dataset):
        writer.writerow([index,row])


with open(_PATH + "dictionary.txt", 'w') as f:
    writer = csv.writer(f)
    for index, row in enumerate(tokens):
        writer.writerow([row + "|" + str(index)])
