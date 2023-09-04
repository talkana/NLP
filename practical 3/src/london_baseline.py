# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
EVAL_CORPUS_PATH = "birth_dev.tsv"
with open(EVAL_CORPUS_PATH) as fin:
    lines = [x.strip().split('\t') for x in fin]
    london_count = 0
    for x in lines:
        if x[1] == "London":
            london_count += 1
    accuracy = london_count / len(lines)
    print(f"Accuracy of baseline: {accuracy}")
