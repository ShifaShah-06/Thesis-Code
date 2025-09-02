import spacy
import os
import csv
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define dependency labels that indicate clausal complexity
CLAUSAL_LABELS = {"acl", "conj", "advcl", "ccomp", "csubj", "discourse", "parataxis"}


# Count number of clausal structures in a sentence
def count_clausal_density(sentence):
    return sum(1 for token in sentence if token.dep_ in CLAUSAL_LABELS)


# Recursively calculate dependency tree depth
def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


# Count nominalisations in a sentence
def count_nominalisations(sentence):
    nominalisation_pattern = re.compile(r"\b\w+(tion|ment|ance|ence|ion|ity|ness|ship)(s|es)?\b", re.IGNORECASE)
    nouns = [token.lemma_ for token in sentence if token.pos_ == "NOUN"]
    nominalisations = [noun for noun in nouns if nominalisation_pattern.match(noun)]
    return len(nominalisations)


# Process each sentence in a text: return sentence, clausal density, syntactic depth, nominalisations, and filename
def process_sentences(text, filename):
    results = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        doc = nlp(line)
        for sentence in doc.sents:
            clausal_density = count_clausal_density(sentence)
            syntactic_depth = walk_tree(sentence.root, 0)
            nominalisation_count = count_nominalisations(sentence)
            results.append([
                sentence.text.strip(),
                clausal_density,
                syntactic_depth,
                nominalisation_count,
                filename
            ])
    return results


# === File Handling ===

# Define the input directory on Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
input_dir = os.path.join(r'C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\matchedGLOBALcorpus-4o-mini\matchedGLOBALcorpus-4o-mini copy')

# Output CSV file path
output_csv = os.path.join(r'C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\global_gpt_redone.csv')

# Collect results from all files
all_results = []

# Iterate over all .txt files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt"):
        print(f"Processing: {file_name}")
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            file_results = process_sentences(text, file_name)
            all_results.extend(file_results)

# Write results to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Sentence", "Clausal Density", "Syntactic Depth", "Nominalisations", "Filename"])
    writer.writerows(all_results)

print(f"Results saved to: {output_csv}")



