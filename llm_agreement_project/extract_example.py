import pandas as pd
import random

def get_diverse_examples_from_crowd(
    crowd_df: pd.DataFrame,
    label_column: str,
    num_per_label: int = 1,
    ref_column: str = "ref_value",
    allowed_labels: list = None,
    seed: int = 42
):
    """
    Extracts diverse examples from the crowd dataset, ensuring coverage over different labels.

    Args:
        crowd_df (pd.DataFrame): The DataFrame with crowd annotations.
        label_column (str): The name of the label column (e.g., "author_type").
        num_per_label (int): Number of examples to sample per label.
        ref_column (str): Column that contains the reference (e.g., URL or domain).
        allowed_labels (list, optional): If specified, only use these labels.
        seed (int): Random seed for reproducibility.

    Returns:
        list of tuples: Each tuple is (ref_value, label)
    """
    random.seed(seed)
    examples = []

    # Take majority vote per ref_value
    majority_vote = crowd_df.groupby(ref_column)[label_column].agg(lambda x: x.mode()[0]).reset_index()

    # Filter by allowed labels if given
    if allowed_labels:
        majority_vote = majority_vote[majority_vote[label_column].isin(allowed_labels)]

    for label in majority_vote[label_column].unique():
        subset = majority_vote[majority_vote[label_column] == label]
        sample_n = min(num_per_label, len(subset))
        if sample_n > 0:
            sampled = subset.sample(n=sample_n, random_state=seed)
            examples.extend(list(zip(sampled[ref_column], sampled[label_column])))

    return examples