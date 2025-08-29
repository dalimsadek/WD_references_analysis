# LLM Agreement Project

Local-LLM pipeline to replace/augment crowdsourcing for Wikidata reference evaluation.  
Developed by **Mohamed Ali Msadek, EURECOM**. This module lives under `expriments/llm_agreement_project`.

The pipeline predicts labels with local LLMs (via **Ollama**) and measures agreement against crowd annotations (Cohen’s κ, annotator stats, plots).

---

## Tasks Covered

1. **Relevance** — does a reference support a Wikidata statement?  
2. **Author Type** — `individual`, `organisation`, `collective`, `nw`, `ne`, `dn`  
3. **Publisher Type** — `news`, `company`, `sp_source`, `academia`, `govt`, `other`, `nw`  
4. **Publisher Verification** — `yes`, `no`, `vendor`, `no_profit`, `political`, `cultural`, `trad_news`, `non_trad_news`, `academia_uni`, `academia_pub`, `academia_other`, `nw`, `ne`, `dn`

---

## Structure

```
llm_agreement_project/
  main.py                    # Relevance predictions (entry point)
  predict.py                 # Inference routines + saving helpers
  evaluation.py              # Agreement: Cohen’s κ, annotator bins/plots, logging
  llm_wrapper.py             # Ollama client wrapper
  extract_example.py         # Few/one-shot sampler from crowd data
  prompt.py                  # Prompt builders + lightweight page enrichment
```

**Data & Results (expected layout):**
```
data/
  crowdsource/
    relevance/llm_prompt_input_relevance_with_labels.csv
      # columns: ref_value,item_label,property_label,value_label,item_id,stat_property,stat_value
    author_type.csv
    publisher_type.csv
    publisher_verification_type.csv

  llm_crowdsource/<prompt_id>/<model>/
    author_results.csv
    publisher_results.csv
    verification_results.csv

results/
  results_<model>_prompt_<prompt_id>/
    relevance_predictions.csv
    unreachable_references.csv

  results_<model>_<prompt_id>/<task>/
    merged.csv
    annotator_stats.csv
    annotator_agreement_bar_chart.png

evaluation_log.csv
```

---

## Requirements

- **Python** 3.10+ (tested)
- **Ollama** running locally (default host: `http://localhost:11434`)
  ```bash
  # examples
  ollama pull mistral:latest
  ollama pull llama3.2
  ```
- Python packages
  ```bash
  pip install pandas scikit-learn matplotlib requests beautifulsoup4 ollama
  ```

---

## Quickstart

### 1) Relevance Prediction (URL + statement)
Edit config at the top of `main.py`:
```python
model = "mistral:latest"
prompt_id = "relevance_eval"
```

Ensure input exists:
```
data/crowdsource/relevance/llm_prompt_input_relevance_with_labels.csv
# must contain:
# ref_value,item_label,property_label,value_label,item_id,stat_property,stat_value
```

Run:
```bash
python main.py
```

Outputs:
```
results/results_<model>_prompt_<prompt_id>/
  relevance_predictions.csv
  unreachable_references.csv
```

---

### 2) Author/Publisher/Verification (optional)
Uncomment the block at the bottom of `main.py` or call functions directly from `predict.py`:

```python
from predict import (
  predict_author_type, predict_publisher_type, predict_verification_type,
  save_all_results
)
from extract_example import get_diverse_examples_from_crowd
```

Optionally sample few/one-shot examples from crowd data:
```python
examples = get_diverse_examples_from_crowd(
  crowd_df, label_column="author_type", num_per_label=1
)
```

Results are saved under:
```
data/llm_crowdsource/<prompt_id>/<model>/
  author_results.csv
  publisher_results.csv
  verification_results.csv
```

> **Note:** If your model prints explanations before JSON, switch the `mode` arg in `predict_*` to `"extract"`.

---

### 3) Evaluate Agreement (LLM vs Crowd)
Configure at the bottom of `evaluation.py`:
```python
model = "llama3.2"
task = "publisher_verification"   # or "author", "publisher"
prompt_id = "prompt_5_metadata"
```

Run:
```bash
python evaluation.py
```

This computes majority vote per `ref_value` from the crowd data, compares with LLM predictions, prints Cohen’s κ and a classification report, generates annotator-level bins/plots, logs runs to `evaluation_log.csv`, and saves CSVs/plots to:
```
results/results_<model>_<prompt_id>/<task>/
  merged.csv
  annotator_stats.csv
  annotator_agreement_bar_chart.png
```

---

## Configuration Notes

- **Model name**: `llm_wrapper.py` uses the exact Ollama tag (`mistral:latest`, `llama3.2`, etc.). To keep filesystem names simple, downstream code may replace `:` with `_` when forming directory names.
- **Unreachable references**: `predict.py` and `prompt.py` mark HTTP failures and store them as `unreachable_references.csv` (or `unreachable_*.csv`).
- **Strict JSON**: Prompts require JSON-only responses: `{ "label": "<...>" }`.  
  - Use `mode="json"` when your model outputs clean JSON.  
  - Use `mode="extract"` to parse the last JSON block when models add extra text.
- **Timeouts and headers**: `prompt.py` uses a 10s timeout and a basic `User-Agent` for fetching pages.

---

## Troubleshooting

- **Ollama connection error** → ensure `ollama serve` is running and the host in `llm_wrapper.py` is correct.  
- **Model not found** → `ollama pull <model>` first.  
- **Empty/garbled JSON** → switch to `mode="extract"` in `predict_*`.  
- **Missing files/paths** → verify the `data/` and `results/` structure shown above.

---

## Provenance

This LLM module extends the 2017 baseline reproduction by introducing prompt-based local inference and agreement evaluation to reduce or replace crowdsourcing.
