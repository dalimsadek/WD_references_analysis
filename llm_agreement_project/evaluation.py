# evaluation.py
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, classification_report
from typing import Tuple

def load_data(llm_model_name: str, prompt_id : str, task: str = "author") -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Loads LLM and crowdsource data based on the task.

    Returns:
        df_llm: DataFrame with ['ref_value', <task_label_column>]
        df_crowd: DataFrame with ['ref_value', 'X_worker_id', <task_label_column>]
        label_column: name of the label column (e.g., 'author_type')
    """
    # Determine label column
    if task == "author":
        label_column = "author_type"
    elif task == "publisher":
        label_column = "publisher_type"
    else:
        label_column = "publisher_verification"

    # Paths
    llm_path = f"data/llm_crowdsource/{prompt_id}/{llm_model_name}/{task}_results.csv"
    crowd_path = f"data/crowdsource/{task}_type.csv"

    # Load
    df_llm = pd.read_csv(llm_path)
    df_crowd = pd.read_csv(crowd_path)

    return df_llm, df_crowd, label_column


def cohen_kappa_against_majority(df_llm, df_crowd, label_column: str):
    """
    Computes Cohen’s Kappa between LLM predictions and human majority vote.
    """
    # Get majority vote
    majority = df_crowd.groupby('ref_value')[label_column].agg(lambda x: x.mode()[0]).reset_index()
    majority.columns = ['ref_value', 'majority_label']

    # Ensure LLM column is named 'label'
    df_llm = df_llm.rename(columns={label_column: 'label'})

    # Merge
    merged = pd.merge(majority, df_llm, on='ref_value')

    # Kappa
    kappa = cohen_kappa_score(merged['majority_label'], merged['label'])
    print(f"✅ Cohen's Kappa (LLM vs Majority): {kappa:.3f}\n")

    # Classification report
    print("📊 Classification Report:")
    print(classification_report(merged['majority_label'], merged['label'], zero_division=0))

    return merged , kappa


def plot_kappa_and_flag_low_performance(
    crowd: pd.DataFrame,
    llm: pd.DataFrame,
    label_column: str,
    kappa_thresh=0.4,
    top_k=5,
    model: str = "model",
    prompt_id: str = "1"
):
    """
    Plot agreement counts per kappa bin and print annotators with poor agreement.
    """
    results = []
    users = crowd['X_worker_id'].unique()

    for user in users:
        df_user = crowd[crowd['X_worker_id'] == user]
        merged = pd.merge(df_user, llm, on='ref_value')

        if len(merged) > 0:
            merged = merged.rename(columns={
                label_column: "label_human",
                "label": "label_llm"
            })

            try:
                kappa = cohen_kappa_score(merged["label_human"], merged["label_llm"])
                agree_count = (merged["label_human"] == merged["label_llm"]).sum()

                results.append({'user': user, 'kappa': kappa, 'agree_count': agree_count})
            except Exception as e:
                print(f"⚠️ Skipped user {user} due to error: {e}")
                continue

    df_results = pd.DataFrame(results)

    # Bin by kappa
    bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['<0', '0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    df_results['kappa_bin'] = pd.cut(df_results['kappa'], bins=bins, labels=labels)

    # Plot
    plt.figure(figsize=(10, 6))
    df_results.groupby('kappa_bin')['agree_count'].sum().plot(
        kind='bar', color='steelblue', edgecolor='black'
    )
    plt.title(f"LLM Agreement Counts by Annotator Kappa Bin ({label_column})")
    plt.xlabel("Cohen's Kappa Bin")
    plt.ylabel("Total Agreements with LLM")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print worst annotators
    print("\n🛑 Annotators with lowest agreement:\n")
    low_kappa = df_results[df_results['kappa'] < kappa_thresh].sort_values(by='kappa')
    print(low_kappa[['user', 'kappa', 'agree_count']].head(top_k).to_string(index=False))

        # Save the plot
    folder = f"results_{model}_{prompt_id}/{task}"
    os.makedirs(folder, exist_ok=True)
    plot_path = os.path.join(folder, "annotator_agreement_bar_chart.png")
    plt.savefig(plot_path)
    print(f"📊 Bar chart saved to: {plot_path}")
    return df_results
def save_results(
    merged_df: pd.DataFrame,
    user_kappa_df: pd.DataFrame,
    model: str,
    prompt_id: str = "1",
    task = str
):
    """
    Saves the agreement evaluation results to a structured folder.

    Args:
        merged_df: Cohen agreement per ref_value
        user_kappa_df: Per-user agreement results
        model: Model name (e.g., llama3.2)
        prompt_id: Prompt variant number (as string or int)
    """
    folder = f"results_{model}_{prompt_id}/{task}"
    os.makedirs(folder, exist_ok=True)

    merged_df.to_csv(os.path.join(folder, "merged.csv"), index=False)
    user_kappa_df.to_csv(os.path.join(folder, "annotator_stats.csv"), index=False)
    print(f"💾 Results saved to: {folder}/")
import csv
from datetime import datetime

def log_cohen_kappa(model: str, prompt_id: str, task: str, kappa: float, logfile: str = "evaluation_log.csv"):
    """
    Appends the current experiment's metadata and Cohen's Kappa score to a log file.
    """
    fieldnames = ["timestamp", "model", "prompt_id", "task", "cohen_kappa"]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "prompt_id": prompt_id,
        "task": task,
        "cohen_kappa": round(kappa, 4)
    }

    # Append to file
    file_exists = os.path.isfile(logfile)

    with open(logfile, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📌 Logged Cohen's Kappa to {logfile}")    



# Entry point
if __name__ == "__main__":
    model = "deepseek-r1:1.5b"   # set model folder
    task = "publisher"      # or "publisher", or "verification"
    prompt_id = 'prompt_1'
    df_llm, df_crowd, label_column = load_data(model,prompt_id, task)

    # Ensure LLM predictions use standardized column name
    df_llm.columns = ['ref_value', label_column]

    # 1. Cohen's Kappa
    merged , kappa = cohen_kappa_against_majority(df_llm, df_crowd, label_column)

    # 2. Annotator analysis
    user_stats = plot_kappa_and_flag_low_performance(
        crowd=df_crowd,
        llm=df_llm.rename(columns={label_column: "label"}),
        label_column=label_column,
        kappa_thresh=0.4,
        top_k=10,
        model = model,
        prompt_id = prompt_id
    )   
    log_cohen_kappa(model=model, prompt_id=prompt_id, task=task, kappa=kappa)
    save_results(merged, user_stats, model=model, prompt_id=prompt_id ,task = task)

