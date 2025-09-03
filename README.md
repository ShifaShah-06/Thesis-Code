# The Impact of AI on Academic Writing: A Paired Analysis of Syntactic Complexity and Reasoning Diversity

This repository contains the core analysis code for the thesis:  
**"The Impact of AI on Academic Writing: A Paired Analysis of Syntactic Complexity and Reasoning Diversity"**  
Author: Shifa Shah  
MSc Behavioural and Data Science, University of Warwick

---

## Overview

This project analyses a large matched corpus of undergraduate psychology essays (2016–2022) and their AI-rewritten counterparts. The goal is to quantify how AI rewriting alters both the syntactic complexity and the diversity of explicit reasoning types in student academic writing.

---

## Repository structure

- `scripts/`: Python scripts for all main analyses
- `data/`: Templates of the wide files I used, not the actual data has been uploaded

---

## Main scripts

| Script                           | Description                                                  |
|-----------------------------------|--------------------------------------------------------------|
| extract_reasoning_by_prompt.py    | Annotates essays for four reasoning types using Azure OpenAI  |
| precision_recall_plots.py         | Plots classifier F1, precision, recall for reasoning types    |
| reasoning_eda.py                  | Exploratory data analysis for reasoning corpus                |
| reasoning_inferential_stats.py    | Paired t-test, Wilcoxon, effect sizes for reasoning markers   |
| reasoning_shannon_stats.py        | Shannon diversity, richness, and overall density metrics      |
| syntactic_metrics.py             | Extract and process syntactic metrics (nominalisations, etc.) |
| syntactic_eda_script.py           | EDA for syntactic features (plots, stats)                     |

**Note:**  
*Some minor preprocessing scripts (e.g., for joining multiple CSVs, or reformatting wide/long tables) are not included in this public repo to avoid clutter and dependency issues. If you need these helper scripts for full reproducibility, please contact me directly at shifa.shah@warwick.ac.uk and I will be happy to share.*

---
