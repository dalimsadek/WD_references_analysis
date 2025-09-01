# Experiments — Wikidata References Analysis

This folder contains the experimental work on Wikidata references quality.

The work was carried out by **Mohamed Ali Msadek, EURECOM**, starting from a fork of the original 2017 repository by [Aliossandro](https://github.com/Aliossandro/WD_references_analysis).  
Minor adjustments were made to the 2017 codebase (path alignment, file format fixes, variable naming, Python compatibility, and automation of evaluations) in order to reproduce the results reliably. These changes are small and do not affect the methodology.

---

# Structure

- **reference_quality_predictor/**  
This folder contains our reproduction of the results of the original 2017 baseline experiments. Models are run either through the notebook [`Experiments.ipynb`](reference_quality_predictor/Expriments.ipynb) or the automated script [`run_all_models.py`](reference_quality_predictor/run_all_models.py), with results stored in `results/all_model_results.txt`.

- **llm_agreement_project/**  
This [folder](llm_agreement_project) presents a new pipeline built on top of the reference quality predictor where the goal is to replace the crowd by a LLM. It uses **local LLMs via a Ollama server** to replace or complement crowdsourced annotations, covering tasks such as relevance, author type, publisher type, and publisher verification. Agreement with human labels is evaluated using metrics like Cohen’s Kappa.

- **Wikidata_analysis/**  
This [notebook](`Wikidata_analysis_v3.ipynb`) analyze the **July 2025 Wikidata dump**, focusing primarily on external reference properties such as:
   - **P248**: *stated in*  
   - **P854**: *reference URL*

You can find the full analysis in this GitHub repository:  
👉 [gabrielmaia7/wikidata-reference-analysis](https://github.com/gabrielmaia7/wikidata-reference-analysis)
---

## Usage

- Start with `reference_quality_predictor/` to reproduce the original machine learning results.  
- Move to `llm_agreement_project/` to explore the new LLM-based experiments.

Each subfolder contains its own README with setup and detailed instructions.
