import pandas as pd
from predict import predict_relevance, save_relevance_results
from llm_wrapper import LLMWrapper

# === Configuration ===
model = "mistral:latest"
prompt_id = "relevance_eval"

# === Load relevance dataset ===
# Must contain: ref_value, item_label, property_label, value_label, item_id, stat_property, stat_value
df = pd.read_csv('data/crowdsource/relevance/llm_prompt_input_relevance_with_labels.csv')
df = df.dropna(subset=['ref_value', 'item_label', 'property_label', 'value_label', 'item_id', 'stat_property', 'stat_value'])

# Prepare input tuples (including IDs)
reference_tuples = list(df[['ref_value', 'item_label', 'property_label', 'value_label',
                            'item_id', 'stat_property', 'stat_value']].itertuples(index=False, name=None))

# === Init LLM ===
llm = LLMWrapper(model_name=model)

# === Run relevance prediction ===
relevance_results, unreachable_refs = predict_relevance(
    llm, reference_tuples, mode="json"
)

# === Save results with IDs included ===
save_relevance_results(
    model=model,
    prompt_id=prompt_id,
    results=relevance_results,
    unreachable=unreachable_refs
)

print("✅ Relevance predictions and unreachable references saved.")



# === COMMENTED: author / publisher / verification ===
# from predict import predict_author_type, predict_publisher_type, predict_verification_type, save_all_results
# from prompt import author_type_prompt, publisher_type_prompt, publisher_verification_prompt
# from extract_example import get_diverse_examples_from_crowd

# pv = pd.read_csv('data/crowdsource/publisher_verification_type.csv')
# unique_verification = pv['ref_value'].dropna().unique()

# pt = pd.read_csv('data/crowdsource/publisher_type.csv')
# unique_publisher = pt['ref_value'].dropna().unique()

# at = pd.read_csv('data/crowdsource/author_type.csv')
# unique_author = at['ref_value'].dropna().unique()

# author_results, unreachable_author = predict_author_type(llm, unique_author, examples=None, mode="json")
# publisher_results, unreachable_publisher = predict_publisher_type(llm, unique_publisher, examples=None, mode="json")
# verification_results, unreachable_verification = predict_verification_type(llm, unique_verification, examples=None, mode="json")

# save_all_results(
#     model=model,
#     prompt_id=prompt_id,
#     author_results=author_results,
#     publisher_results=publisher_results,
#     verification_results=verification_results,
#     unreachable_author=unreachable_author,
#     unreachable_publisher=unreachable_publisher,
#     unreachable_verification=unreachable_verification
# )
