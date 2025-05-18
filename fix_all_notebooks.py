import json
import os
import glob

def fix_notebook(notebook_path):
    """Fix a Jupyter notebook to ensure it has valid structure"""
    try:
        with open(notebook_path, 'r') as f:
            try:
                notebook = json.load(f)
                
                # Ensure all code cells have required properties
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        if 'outputs' not in cell:
                            cell['outputs'] = []
                        if 'execution_count' not in cell or cell['execution_count'] is None:
                            cell['execution_count'] = None
                
                # Write back the fixed notebook
                with open(notebook_path, 'w') as f_out:
                    json.dump(notebook, f_out, indent=1)
                print(f"✓ Fixed notebook: {notebook_path}")
                return True
            except json.JSONDecodeError:
                print(f"✗ Error: {notebook_path} is not valid JSON")
                return False
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {str(e)}")
        return False

# Find all notebook files
notebook_files = glob.glob("*.ipynb")
fixed_count = 0

print(f"Found {len(notebook_files)} notebook files")
for notebook_path in notebook_files:
    if fix_notebook(notebook_path):
        fixed_count += 1

print(f"\nFixed {fixed_count} of {len(notebook_files)} notebooks") 