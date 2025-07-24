import pandas as pd
from predict import predict_author_type, predict_publisher_type, predict_verification_type, save_all_results
from prompt import author_type_prompt, publisher_type_prompt, publisher_verification_prompt
from llm_wrapper import LLMWrapper
from extract_example import get_diverse_examples_from_crowd

# Configuration
model = "deepseek-r1:1.5b"
prompt_id = "1"

# Load unique values from crowdsource
pv = pd.read_csv('data/crowdsource/publisher_verification_type.csv')
unique_verification = pv['ref_value'].dropna().unique()

pt = pd.read_csv('data/crowdsource/publisher_type.csv')
unique_publisher = pt['ref_value'].dropna().unique()

at = pd.read_csv('data/crowdsource/author_type.csv')
unique_author = at['ref_value'].dropna().unique()

# # Get few-shot examples (from crowd agreement only)
# examples_author = get_agreeing_examples_by_label(at, label_column="author_type")
# examples_publisher = get_agreeing_examples_by_label(pt, label_column="publisher_type")
# examples_verification = get_agreeing_examples_by_label(pv, label_column="publisher_verification")

# Init LLM
llm = LLMWrapper(model_name=model)

# Run predictions
#author_results = predict_author_type(llm, unique_author,examples=None,mode="extract")
publisher_results = predict_publisher_type(llm, unique_publisher, examples_publisher = None , mode ="extract")
# verification_results = predict_verification_type(llm, unique_verification, examples_verification)

# Save to CSV
save_all_results( model , prompt_id , author_results = [] , verification_results = [] , publisher_results = publisher_results)
print("✅ All predictions completed and saved.")
