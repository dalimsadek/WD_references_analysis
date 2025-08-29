# Reference Quality Predictor

This folder reproduces the **2017 Wikidata references quality analysis**, originally developed in [Aliossandro/WD_references_analysis](https://github.com/Aliossandro/WD_references_analysis).  
Work reproduced and maintained by **Mohamed Ali Msadek, EURECOM**.

---

## Overview

The goal of this component is to evaluate the quality of external references in Wikidata using **machine learning models**.  
The reproduction is faithful to the 2017 paper, with only **minor fixes** applied (path alignment, file format consistency, Python compatibility, and automation of evaluation). These changes are minimal and do not alter the methodology.

---

## Contents

- **Expriments.ipynb** — Interactive Jupyter notebook that reproduces the experiments step by step.  
- **run_all_models.py** — Script to automatically train and evaluate all models and save results in one file.  
- **results/** — Directory where all outputs are stored (notably `all_model_results.txt`).

---

## Usage

### Option 1 — Run the notebook
Open and run:
** Expriments.ipynb ** 

### Option 2 — Run the script
Execute all models at once:
```bash
python run_all_models.py

# Results

All outputs will be saved into:

- `results/all_model_results.txt`


# Requirements

- Python 3.10+ (tested)  
- Install dependencies:
  ```bash
  pip install pandas scikit-learn matplotlib numpy


# Notes

This reproduction validates the **baseline results** of the 2017 work.

It also serves as the foundation before moving to the **LLM Agreement Project**,  
where crowdsourcing is replaced by local LLMs.
