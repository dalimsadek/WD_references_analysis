# Reference Quality Predictor

This folder reproduces the **2017 Wikidata references quality analysis**, originally developed in [Aliossandro/WD_references_analysis](https://github.com/Aliossandro/WD_references_analysis).  
Work reproduced and maintained by **Mohamed Ali Msadek, EURECOM**.

---

## Overview

The goal of this component is to evaluate the quality of external references in Wikidata using **machine learning models**.  
The reproduction is faithful to the 2017 paper, with only **minor fixes** applied (path alignment, file format consistency, Python compatibility, and evaluation automation). These changes are minimal and do not alter the methodology.

---

## Usage

You have two options to run the experiments:

### Option 1 — Run models one by one
Use:
```bash
python model_trainer.py 
```
You must specify the model and the task each time.

### Option 2 — Run all models at once
Execute:
```bash
python run_all_models.py
```

Results will be saved into:
```
results/all_model_results.txt
```

---

## Results

All outputs will be saved into:

- `results/all_model_results.txt`

---

## Requirements

- Python 3.10+ (tested)  
- Install dependencies:
  ```bash
  pip install pandas scikit-learn matplotlib numpy
  ```

---

## Notes

This reproduction validates the **baseline results** of the 2017 work.  

It also serves as the foundation before moving to the **LLM Agreement Project**,  
where crowdsourcing is replaced by local LLMs.
