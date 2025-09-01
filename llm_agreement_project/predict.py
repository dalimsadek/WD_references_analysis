import json 
from prompt import author_type_prompt, publisher_type_prompt, publisher_verification_prompt, build_enriched_prompt, build_relevance_prompt_from_ref
from llm_wrapper import LLMWrapper
import pandas as pd
import os


import json

def predict_author_type(llm, unique_authors, examples=None, mode="json"):
    """
    Predicts author type using a language model, with optional support for different JSON extraction modes.

    Args:
        llm: The LLM wrapper (must have run_prompt method).
        unique_authors: List of reference values.
        examples: Few-shot examples [(ref, label), ...]
        mode: 'json' (default) if model outputs clean JSON, 'extract' if output includes explanations before JSON.
    """
    author_results = []
    unreachable_refs = []
    for i, ref in enumerate(unique_authors, start=1):
        prompt = build_enriched_prompt(ref,"author_type")
        if "The reference is unreachable" in prompt:
            unreachable_refs.append((ref, "unreachable"))
            print(f"⛔ Unreachable reference: {ref}")
            continue
        response = llm.run_prompt(prompt)

        try:
            if mode == "json":
                label = json.loads(response)["label"]
            elif mode == "extract":
                start_idx = response.rfind('{')
                json_str = response[start_idx:]
                label = json.loads(json_str)["label"]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            author_results.append((ref, label))

        except Exception as e:
            print(f"⚠️ Skipped {ref} at index {i} due to error: {e}")
            continue

        if i % 10 == 0 or i == len(unique_authors):
            print(f"✅ Processed {i}/{len(unique_authors)} references")

    return author_results, unreachable_refs





import json

def predict_publisher_type(llm, unique_publishers, examples_publisher=None, mode="json"):
    publisher_results = []
    unreachable_refs = []
    for i, ref in enumerate(unique_publishers, start=1):
        prompt = build_enriched_prompt(ref, "publisher_type")
        if "The reference is unreachable" in prompt:
            unreachable_refs.append((ref, "unreachable"))
            print(f"⛔ Unreachable reference: {ref}")
            continue
        response = llm.run_prompt(prompt)
        
        try:
            if mode == "json":
                label = json.loads(response)["label"]
            elif mode == "extract":
                start_idx = response.rfind('{')
                json_str = response[start_idx:]
                label = json.loads(json_str)["label"]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            publisher_results.append((ref, label))
        except Exception as e:
            print(f"⚠️ Skipped {ref} at index {i} due to error: {e}")
            continue

        if i % 10 == 0 or i == len(unique_publishers):
            print(f"✅ Processed {i}/{len(unique_publishers)} references")

    return publisher_results, unreachable_refs




def predict_verification_type(llm, unique_verification, examples_verification=None, mode="json"):
    verification_results = []
    unreachable_refs = []
    for i, ref in enumerate(unique_verification, start=1):
        prompt = build_enriched_prompt(ref, "publisher_verification")
        if "The reference is unreachable" in prompt:
            unreachable_refs.append((ref, "unreachable"))
            print(f"⛔ Unreachable reference: {ref}")
            continue
        response = llm.run_prompt(prompt)
        
        try:
            if mode == "json":
                label = json.loads(response)["label"]
            elif mode == "extract":
                start_idx = response.rfind('{')
                json_str = response[start_idx:]
                label = json.loads(json_str)["label"]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            verification_results.append((ref, label))
        except Exception as e:
            print(f"⚠️ Skipped {ref} at index {i} due to error: {e}")
            continue

        if i % 10 == 0 or i == len(unique_verification):
            print(f"✅ Processed {i}/{len(unique_verification)} references")

    return verification_results, unreachable_refs



def save_all_results(model, prompt_id, author_results=None, publisher_results=None, verification_results=None,
                     unreachable_author=None, unreachable_publisher=None, unreachable_verification=None):
    output_dir = f"data/llm_crowdsource/prompt_{prompt_id}/{model}"
    os.makedirs(output_dir, exist_ok=True)

    if author_results:
        df_author = pd.DataFrame(author_results, columns=["ref_value", "author_type"])
        df_author.to_csv(os.path.join(output_dir, "author_results.csv"), index=False)
        print("💾 Saved author results.")

    if unreachable_author:
        df_bad_author = pd.DataFrame(unreachable_author, columns=["ref_value", "status"])
        df_bad_author.to_csv(os.path.join(output_dir, "unreachable_author.csv"), index=False)
        print("❌ Saved unreachable author references.")

    if publisher_results:
        df_publisher = pd.DataFrame(publisher_results, columns=["ref_value", "publisher_type"])
        df_publisher.to_csv(os.path.join(output_dir, "publisher_results.csv"), index=False)
        print("💾 Saved publisher results.")

    if unreachable_publisher:
        df_bad_publisher = pd.DataFrame(unreachable_publisher, columns=["ref_value", "status"])
        df_bad_publisher.to_csv(os.path.join(output_dir, "unreachable_publisher.csv"), index=False)
        print("❌ Saved unreachable publisher references.")

    if verification_results:
        df_verification = pd.DataFrame(verification_results, columns=["ref_value", "publisher_verification"])
        df_verification.to_csv(os.path.join(output_dir, "verification_results.csv"), index=False)
        print("💾 Saved verification results.")

    if unreachable_verification:
        df_bad_verification = pd.DataFrame(unreachable_verification, columns=["ref_value", "status"])
        df_bad_verification.to_csv(os.path.join(output_dir, "unreachable_verification.csv"), index=False)
        print("❌ Saved unreachable verification references.")



def predict_relevance(llm, reference_triples, examples=None, mode="json"):
    """
    Predicts relevance of references using an LLM.

    Args:
        llm: The LLM wrapper (must have run_prompt method).
        reference_triples: List of (ref_url, item_label, property_label, value_label, item_id, stat_property, stat_value).
        examples: Unused for now, placeholder for few-shot.
        mode: 'json' if model returns clean JSON, 'extract' if response includes explanations before JSON.
    """
    relevance_results = []
    unreachable_refs = []

    for i, (ref_url, item_label, property_label, value_label, item_id, stat_property, stat_value) in enumerate(reference_triples, start=1):
        prompt = build_relevance_prompt_from_ref(ref_url, item_label, property_label, value_label)

        if "The reference is unreachable" in prompt:
            unreachable_refs.append((ref_url, "unreachable"))
            print(f"⛔ Unreachable reference: {ref_url}")
            continue

        response = llm.run_prompt(prompt)

        try:
            if mode == "json":
                label = json.loads(response)["label"]
            elif mode == "extract":
                start_idx = response.rfind('{')
                json_str = response[start_idx:]
                label = json.loads(json_str)["label"]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            relevance_results.append((
                ref_url, item_label, property_label, value_label,
                item_id, stat_property, stat_value, label
            ))

        except Exception as e:
            print(f"⚠️ Skipped {ref_url} at index {i} due to error: {e}")
            continue

        if i % 10 == 0 or i == len(reference_triples):
            print(f"✅ Processed {i}/{len(reference_triples)} references")

    return relevance_results, unreachable_refs



import os
import pandas as pd

def save_relevance_results(model, prompt_id, results, unreachable, save_dir="results"):
    """
    Saves relevance predictions and unreachable references to CSV.

    Args:
        model (str): Model name (e.g., "llama3.2")
        prompt_id (str): Prompt version used (e.g., "relevance_eval")
        results (list of tuples): (ref_value, item_label, property_label, value_label, item_id, stat_property, stat_value, label)
        unreachable (list of tuples): (ref_value, "unreachable")
    """

    # Build result path
    output_path = os.path.join(save_dir, f"results_{model}_prompt_{prompt_id}")
    os.makedirs(output_path, exist_ok=True)

    # Save relevance predictions
    if results:
        df_pred = pd.DataFrame(results, columns=[
            "ref_value", "item_label", "property_label", "value_label",
            "item_id", "stat_property", "stat_value", "label"
        ])
        df_pred.to_csv(os.path.join(output_path, "relevance_predictions.csv"), index=False)
        print(f"✅ Saved {len(results)} relevance predictions.")
    else:
        print("⚠️ No relevance predictions to save.")

    # Save unreachable references
    if unreachable:
        df_unreachable = pd.DataFrame(unreachable, columns=["ref_value", "status"])
        df_unreachable.to_csv(os.path.join(output_path, "unreachable_references.csv"), index=False)
        print(f"⛔ Saved {len(unreachable)} unreachable references.")
    else:
        print("✅ No unreachable references found.")
