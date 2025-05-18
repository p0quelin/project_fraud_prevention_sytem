import json

# Load the notebook
with open('payment_fraud_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix each code cell by adding empty outputs if missing
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        if 'outputs' not in cell:
            cell['outputs'] = []
        if 'execution_count' not in cell:
            cell['execution_count'] = None

# Save the fixed notebook
with open('payment_fraud_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully!") 