# Classifying Implicit Meaning

This project focuses on **automatically classifying whether text alterations meaningfully affect reader comprehension**. Given two versions of text, an original and a revised version with inserted explanations, models predict whether the insertion changes how most readers understand the text. Long-term, the goal is to automatically extract instances of so-called **explicitations** where implicit meaning has merely been verbalized.

## 📋 Task Definition

**Binary Classification Problem:** Does inserting explanatory text (in angle brackets) meaningfully change how readers understand the sentence?

**Example:**
- **Original:** "Keep your valuables in a safe."
- **Revised:** "Keep your valuables in a safe *to prevent theft*."
- **Label:** No (the insertion is already implicitly conveyed in the original)

**Labels:**
- `Yes` - The insertion meaningfully changes reader understanding
- `No` - The insertion is merely an explicitation of implicit meaning
---

## 📁 Project Structure

### Core Directories

**`classification/`** - Main classification pipeline

**`analysis/`** - Data analysis and visualization

**`core/`** - Shared utilities

**`framework_parsers/`** - Linguistic framework analysis

  **Purpose:** Extract linguistic structural patterns (syntactic trees, semantic frames, discourse relations) to understand what linguistic properties correlate with implicit vs. explicit information. Supports research into how different textual frameworks relate to whether information is implicit or explicit.

**`instructions/`** - Annotation guidelines for human annotators

---

## 🚀 Getting Started

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Format

Data is stored in JSON format with the following structure:
```json
{
  "sample_id": {
    "sentence_1": "original text",
    "sentence_2": "revised text with <insertion>",
    "context_before": "previous context or 'no context'",
    "context_after": "following context or 'no context'",
    "MACE Label": "Yes/No"
  }
}
```

---

## 🔬 Classification Approaches

### 1. **Fine-tuned LLMs** (Recommended)
Train large language models with LoRA adapters for efficient fine-tuning.

**Supported Models:**
- Meta-Llama-3-8B-Instruct
- Mistral-7B-Instruct
- Qwen2-7B-Instruct

**Fine-tuning with LoRA:**
```bash
python classification/classify_implicit_data.py \
  --data_split train \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --config classification/configs/llama/config.yaml \
  --lora_config classification/configs/llama/lora_config.yaml \
  -ft
```

**Inference on Test Set:**
```bash
python classification/classify_implicit_data.py \
  --data_split test \
  --model classification/fine_tuned_models/llama \
  -z
```

### 2. **NLI-based Classification**
Use Natural Language Inference models to determine contradictions between original and revised text.

**Supported Models:**
- facebook/bart-large-mnli
- roberta-large-mnli
- MoritzLaurer/deberta-v3-large-zeroshot-v2.0

```bash
python classification/run_nli.py \
    --data_split test
    --model_name facebook/bart-large-mnli \
    -c \
    -b
```

### 3. **GPT Models** (API-based)
Use OpenAI's GPT models with few-shot prompting.

Requires file with API key:
```bash
python classification/classify_implicit_data.py \
  --data_split test \
  --model gpt-4 \
  --api <api_file>
```

### 4. **Baseline**
- **Majority Baseline:** Always predict the majority class

---

## 📊 Classification on Test Sets

To run inference and get predictions on test data, use `classify_implicit_data.py`:

### In-Domain Test Set
```bash
python classification/classify_implicit_data.py \
  --data_split test \
  --model classification/fine_tuned_models/llama \
  -z
```

### Out-of-Domain Test Set
Test generalization on data from different domains:
```bash
python classification/classify_implicit_data.py \
  --data_split out_of_domain_test \
  --model classification/fine_tuned_models/llama \
  -z
```

This produces predictions saved in `predictions/` that can then be analyzed for:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrices
- Error analysis

---

## 🔍 Evaluating Predictions

After generating predictions, use the evaluation scripts to compute metrics and analyze model performance.

### Compute Classification Metrics

Example usage for `evaluate_predictions.py`:
```bash
python classification/evaluation/evaluate_predictions.py \
  --gold_labels ../implicit_data/gold_standard.json \
  --predictions predictions/model_predictions.json \
  --output results/metrics.json \
  --plot confusion_matrix.png
```

Run multiple evaluations at once with `evaluation.sh`:
```bash
bash classification/evaluation/evaluation.sh
```

This generates:
- `results/metrics.json` - Classification metrics
- Confusion matrix plots (`.png`)
- Error analysis reports

---

## 📈 Analysis & Interpretation

### Linguistic Analysis
Analyze linguistic properties of insertions and their relationship to labels:

```bash
python analysis/analyze_linguistics.py \
  --data_path ../implicit_data/gold_standard.json \
  --output_path analysis/results.json
```

**Features analyzed:**
- POS (Part-of-Speech) distribution
- Lexical density
- Revision length statistics
- Label distribution

### Significance Testing
Perform statistical significance tests on model predictions:

```bash
python analysis/significance_testing.py \
  --predictions predictions/model_predictions.json \
  --gold_standard ../implicit_data/gold_standard.json
```

### Visualization
Generate plots for linguistic analysis results:

```bash
bash analysis/analyze.sh
```

---

## 📝 Notes

- **GPU Requirements:** A40 or equivalent (24GB VRAM recommended)
- **Cache Management:** Model cache is stored in `.cache/` directory
- **Hugging Face Login:** Required for accessing gated models (Llama)
