# AI-Rewriting and Reasoning Diversity in Student Essays: Analysis Pipeline

This repository contains the core analysis code for the thesis:  
**"How does AI rewriting shape syntactic style and reasoning diversity in student essays?"**  
Author: Shifa Shah  
MSc Behavioural and Data Science, University of Warwick

---

## Overview

This project analyses a large matched corpus of undergraduate psychology essays (2015–2025) and their AI-rewritten counterparts. The goal is to quantify how AI rewriting alters both the syntactic complexity and the diversity of explicit reasoning types in student academic writing.

---

## Repository structure

- `scripts/`: Python scripts for all main analyses
- `data/`: Example data schema, not real student data (see below)
- `results/`: (Optional) Key figures and tables exported from scripts

---

## Main scripts

| Script                           | Description                                                  |
|-----------------------------------|--------------------------------------------------------------|
| extract_reasoning_by_prompt.py    | Annotates essays for four reasoning types using Azure OpenAI  |
| precision_recall_plots.py         | Plots classifier F1, precision, recall for reasoning types    |
| reasoning_eda.py                  | Exploratory data analysis for reasoning corpus                |
| reasoning_inferential_stats.py    | Paired t-test, Wilcoxon, effect sizes for reasoning markers   |
| reasoning_shannon_stats.py        | Shannon diversity, richness, and overall density metrics      |
| syntactic_analysis.py             | Extract and process syntactic metrics (nominalisations, etc.) |
| syntactic_eda_script.py           | EDA for syntactic features (plots, stats)                     |

**Note:**  
*Some minor preprocessing scripts (e.g., for joining multiple CSVs, or reformatting wide/long tables) are not included in this public repo to avoid clutter and dependency issues. If you need these helper scripts for full reproducibility, please contact me directly at [your.email@warwick.ac.uk](mailto:your.email@warwick.ac.uk) and I am happy to share.*

---

## Reproducing the analyses

- Clone the repo and set up your Python environment:
  ```bash
  git clone https://github.com/your-user/ai-student-essays-thesis.git
  cd ai-student-essays-thesis
  pip install -r requirements.txt
