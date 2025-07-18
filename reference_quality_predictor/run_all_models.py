# -*- coding: utf-8 -*-
import os
import subprocess

# Models to run
models = [
    'baseline',
    'svm_model',
    'svm_model_cv',
    'linear_svm',
    'linear_svm_cv',
    'rf_model',
    'rf_model_cv',
    'nb_model',
    'nb_model_cv'
]

# Prediction tasks
targets = {
    'authoritativeness': '0',
    'relevance': '1'
}

# Output text file (not CSV)
output_file = 'all_model_results.txt'

# Clear the file before writing
with open(output_file, 'w') as f:
    f.write("All Model Results\n==================\n\n")

for task_name in targets:
    task_id = targets[task_name]

    for model in models:
        print("\n>>> Running model: {}, task: {}".format(model, task_name))

        process = subprocess.Popen(
            ['python', 'model_trainer.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        input_data = model + '\n' + task_id + '\n'
        stdout, stderr = process.communicate(input=input_data)

        lines = stdout.splitlines()
        result_line = ''
        for line in lines:
            if 'precision:' in line and ';' in line:
                result_line = line
                break

        with open(output_file, 'a') as f:
            f.write("Model: {}\nTask: {}\nResult: {}\n\n".format(model, task_name, result_line))
            if not result_line:
                f.write("Could not parse output.\n\n")
