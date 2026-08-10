# BERT Text Classifier

A PyTorch-based text classification project leveraging pre-trained BERT (Bidirectional Encoder Representations from Transformers) models from Hugging Face.

---

## 📁 Project Structure

Here is the current structure of the project and the files created:

* **`Data/`**: Directory reserved for storing raw and processed datasets.
* **`Notebooks/`**: Directory for exploratory data analysis (EDA) and experimental Jupyter notebooks.
* **`src/`**: Source directory containing the modularized Python codebase:
  * `dataset.py` - Data loading, tokenization, and PyTorch `Dataset` definition.
  * `model.py` - BERT-based classification model definition.
  * `train.py` - Training loop, optimization, and checkpointing logic.
  * `evaluate.py` - Model evaluation script to compute metrics (accuracy, precision, recall, F1-score).
* **`requirements.md`**: Markdown file documenting the required libraries and specific versions.
* **`Readme.md`**: Project documentation and setup guide (this file).

---

## ⚙️ Setup & Installation

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to isolate project dependencies.

#### **On Windows (PowerShell/CMD):**
```powershell
# Create the virtual environment named '.venv'
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

#### **On macOS / Linux:**
```bash
# Create the virtual environment named '.venv'
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

*When activated, your terminal prompt will show `(.venv)` at the beginning.*

---

### 2. Install Dependencies

The required libraries are listed in `requirements.md`:
* `torch==2.11.0`
* `transformers==5.13.0`
* `datasets==5.0.1`
* `scikit-learn==1.8.0`

You can install these packages in your active virtual environment by running:

```bash
pip install torch==2.11.0 transformers==5.13.0 datasets==5.0.1 scikit-learn==1.8.0
```

Alternatively, if you prefer using a standard `requirements.txt` file, you can create one with the contents above and run:
```bash
pip install -r requirements.txt
```
