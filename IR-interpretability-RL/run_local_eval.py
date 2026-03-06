import json

with open("kaggle_meta_run/run_meta_experiments_kaggle.ipynb", "r") as f:
    nb = json.load(f)

script = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.startswith("!"):
            # Fix the naming error directly in the python string
            source = source.replace("SparseAutoencoder(768)", "BipolarSAE(768)")
            script += source + "\n\n"

with open("meta_eval_script.py", "w") as f:
    f.write("import torch\nimport numpy as np\nimport time\nimport torch.nn.functional as F\nfrom torch import nn\n")
    f.write(script)
