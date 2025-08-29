# Experiments — Wikidata References Analysis

This folder contains the experimental work on Wikidata references quality.

The work was carried out by **Mohamed Ali Msadek, EURECOM**, starting from a fork of the original 2017 repository by [Aliossandro](https://github.com/Aliossandro/WD_references_analysis).  
Minor adjustments were made to the 2017 codebase (path alignment, file format fixes, variable naming, Python compatibility, and automation of evaluations) in order to reproduce the results reliably. These changes are small and do not affect the methodology.

---

# Structure

- **expriments/**
  - **reference_quality_predictor/** — Reproduction of the 2017 machine learning models  
  - **llm_agreement_project/** — Extension with LLMs replacing/augmenting crowdsourcing



- **reference_quality_predictor/**  
  Contains the reproduction of the 2017 baseline experiments.  
  Models are run either through the notebook (`Expriments.ipynb`) or the automated script (`run_all_models.py`), with results stored in `results/all_model_results.txt`.

- **llm_agreement_project/**  
  A new pipeline built on top of the baseline.  
  It introduces **local LLMs via Ollama** to replace or complement crowdsourced annotations, covering tasks such as relevance, author type, publisher type, and publisher verification.  
  Agreement with human labels is evaluated using metrics like Cohen’s Kappa.

---

## Usage

- Start with `reference_quality_predictor/` to reproduce the original machine learning results.  
- Move to `llm_agreement_project/` to explore the new LLM-based experiments.

Each subfolder contains its own README with setup and detailed instructions.
