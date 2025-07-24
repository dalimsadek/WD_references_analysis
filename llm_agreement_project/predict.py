import json 
from prompt import author_type_prompt, publisher_type_prompt, publisher_verification_prompt
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

    for i, ref in enumerate(unique_authors, start=1):
        prompt = author_type_prompt(ref, examples)
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

    return author_results





import json

def predict_publisher_type(llm, unique_publishers, examples_publisher=None, mode="json"):
    publisher_results = []

    for i, ref in enumerate(unique_publishers, start=1):
        prompt = publisher_type_prompt(ref, examples_publisher)
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

    return publisher_results




def predict_verification_type(llm, unique_verification, examples_verification=None, mode="json"):
    verification_results = []

    for i, ref in enumerate(unique_verification, start=1):
        prompt = publisher_verification_prompt(ref, examples_verification)
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

    return verification_results



def save_all_results( model, prompt_id, author_results = None, publisher_results = None, verification_results = None):
    """
    Saves the results of author, publisher, and verification predictions into CSV files.

    Args:
        author_results (list of tuples): Each tuple is (ref_value, author_type)
        publisher_results (list of tuples): Each tuple is (ref_value, publisher_type)
        verification_results (list of tuples): Each tuple is (ref_value, publisher_verification)
        model (str): The model name (e.g., "llama_3.2")
        prompt_id (str): The prompt identifier (e.g., "2_onesl")
    """
    # Create target directory
    output_dir = f"data/llm_crowdsource/prompt_{prompt_id}/{model}"
    os.makedirs(output_dir, exist_ok=True)

    # Save author results
    if author_results:
        df_author = pd.DataFrame(author_results, columns=["ref_value", "author_type"])
        author_path = os.path.join(output_dir, "author_results.csv")
        df_author.to_csv(author_path, index=False)
        print(f"💾 Saved author results to: {author_path}")

    # Save publisher results
    if publisher_results:
        df_publisher = pd.DataFrame(publisher_results, columns=["ref_value", "publisher_type"])
        publisher_path = os.path.join(output_dir, "publisher_results.csv")
        df_publisher.to_csv(publisher_path, index=False)
        print(f"💾 Saved publisher results to: {publisher_path}")

    # Save verification results
    if verification_results:
        df_verification = pd.DataFrame(verification_results, columns=["ref_value", "publisher_verification"])
        verification_path = os.path.join(output_dir, "verification_results.csv")
        df_verification.to_csv(verification_path, index=False)
        print(f"💾 Saved verification results to: {verification_path}")
