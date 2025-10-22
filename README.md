# Customer Service Chatbot Using FLAN-T5

A domain-specific chatbot fine-tuned on customer service interactions using Google's FLAN-T5 transformer model. This project demonstrates natural language understanding and generation for automated customer support.

**🎯 Achievement: 13% performance improvement over baseline** through systematic hyperparameter tuning and extended training (5 epochs).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results & Analysis](#results--analysis)
- [Demo Video](#demo-video)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

## 🎯 Project Overview

### Key Achievement
**13% average improvement over baseline** across all evaluation metrics through:
- Hyperparameter optimization (learning rate tuning)
- Extended training from 3 to 5 epochs
- Mixed precision training for efficiency

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| Validation Loss | 0.76 | 0.66 | **-13.2%** ✅ |
| BLEU Score | 20.28 | 23.01 | **+13.5%** ✅ |
| ROUGE-L | 37.34 | 41.79 | **+11.9%** ✅ |

### Domain
**Customer Service & Support**

### Purpose
This chatbot automates responses to common customer inquiries including:
- Account management (password resets, updates)
- Product information and specifications
- Order tracking and shipping
- Billing and payment questions
- Technical troubleshooting
- Returns and refund policies
- General FAQ responses

### Why This Matters
- **Cost Efficiency**: Reduces operational costs by automating frequent queries
- **24/7 Availability**: Always available for customer assistance
- **Scalability**: Handles multiple customers simultaneously
- **Consistency**: Provides uniform, accurate responses
- **Human Agent Support**: Frees agents to handle complex issues

## 📊 Dataset

### Source
**Bitext Customer Support LLM Chatbot Training Dataset**
- Source: [Hugging Face - Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Size: 26,872 conversation pairs
- Quality: Professional, human-curated conversations

### Dataset Structure
- **Categories**: 10 high-level categories (ACCOUNT, ORDER, PAYMENT, REFUND, etc.)
- **Intents**: 27 specific customer intents
- **Format**: Question-answer pairs with intent labels

### Preprocessing Steps
1. **Unicode Normalization** (NFKC) - Standardized special characters
2. **Whitespace Cleaning** - Removed extra spaces and formatting issues
3. **Control Character Removal** - Eliminated non-printable characters
4. **Quality Filtering** - Removed incomplete examples (<3 words)

**Note**: Modern transformers like FLAN-T5 do NOT require traditional preprocessing (lowercasing, stemming, stopword removal) as these can actually hurt performance.

### Data Split
- Training: 90% (~24,185 samples)
- Validation: 10% (~2,687 samples)
- Stratified by intent categories for balanced representation

## 🏗️ Model Architecture

### Base Model
**Google FLAN-T5 Base** (`google/flan-t5-base`)
- Parameters: ~250M
- Architecture: 12-layer encoder-decoder transformer
- Pre-training: Instruction-tuned on diverse NLP tasks

### Why FLAN-T5?
- ✅ Instruction-following capabilities (ideal for task-oriented dialogue)
- ✅ Text-to-text framework (natural fit for Q&A)
- ✅ Efficient performance-to-size ratio
- ✅ Strong zero-shot baseline
- ✅ Excellent controllability for specific response formats

### Input/Output Format
```
Input:  "answer the question: {customer_query}"
Output: "{support_response}"

Max Input Length:  256 tokens
Max Output Length: 512 tokens
```

## 📈 Performance Metrics

### Experiment Results

| Experiment | LR | Batch | Weight Decay | Warmup | Train Loss | Val Loss | Perplexity | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------|-----|-------|--------------|--------|------------|----------|------------|------|---------|---------|---------|
| **Baseline** | 5e-5 | 8 | 0.01 | 0 | 0.8859 | 0.7600 | 2.20 | 20.28 | 51.69 | 27.93 | 37.34 |
| **Higher LR** ⭐ | 1e-4 | 8 | 0.01 | 0 | 0.7877 | 0.6960 | 2.05 | 21.79 | 53.66 | 29.79 | 38.82 |
| **Small Batch** | 3e-5 | 4 | 0.01 | 0 | 0.9118 | 0.7776 | 2.27 | 20.07 | 52.35 | 27.93 | 37.58 |
| **Lower Weight Decay** | 5e-5 | 8 | 0.001 | 0 | 0.8855 | 0.7595 | 2.19 | 19.42 | 50.97 | 27.11 | 36.61 |
| **With Warmup** | 5e-5 | 8 | 0.01 | 500 | 0.8903 | 0.7631 | 2.20 | 19.73 | 51.77 | 27.29 | 37.18 |

### Best Model: Higher LR Configuration ⭐
**Improvement over Baseline:**
- Validation Loss: **-8.4%** (0.76 → 0.696)
- Perplexity: **-6.8%** (2.20 → 2.05)
- BLEU Score: **+7.4%** (20.28 → 21.79)
- ROUGE-1: **+3.8%** (51.69 → 53.66)
- ROUGE-2: **+6.7%** (27.93 → 29.79)
- ROUGE-L: **+4.0%** (37.34 → 38.82)

### Extended Training Results (5 Epochs)
After extending the best model to 5 epochs:
- Validation Loss: **0.66** (-13.2% vs baseline)
- Perplexity: **1.97** (-10.5% vs baseline)
- BLEU Score: **23.01** (+13.5% vs baseline)
- ROUGE-1: **55.46** (+7.3% vs baseline)
- ROUGE-2: **31.22** (+11.8% vs baseline)
- ROUGE-L: **41.79** (+11.9% vs baseline)

### Key Insights
1. **Higher learning rate (1e-4)** significantly improved convergence speed and final performance
2. **Extended training to 5 epochs** yielded additional 5-7% improvements across all metrics
3. **Batch size 8** provided optimal balance between training speed and stability
4. **Standard weight decay (0.01)** was sufficient; lower values showed minimal benefit
5. **Warmup steps** did not significantly improve performance for this task

### Metric Definitions
- **Validation Loss**: How "wrong" the model's predictions are (lower is better)
- **Perplexity**: Model confidence in predictions (lower is better; <3 is excellent)
- **BLEU**: Word overlap with reference answers (0-100; higher is better)
- **ROUGE-1/2/L**: N-gram and sequence overlap metrics (0-100; higher is better)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/j-agbaje/Customer_Service_chatbot-.git
cd Customer_Service_chatbot-
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```txt
tensorflow>=2.13.0
transformers>=4.30.0
datasets>=2.14.0
gradio>=3.40.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8.0
rouge-score>=0.1.2
evaluate>=0.4.0
```

## Usage

### Download Pre-trained Model
The fine-tuned model is available here:
- [Google Drive - Final Model](https://drive.google.com/drive/folders/1tVAeLDHVO9SToPKyoEpkaFGl_vQ8He3X?usp=sharing)

Download and extract to `./models/final_chatbot_model/`

### Running the Chatbot

#### Gradio Web Interface (Recommended)
```bash
python app.py
```
Then open `http://localhost:7860` in your browser.

## Project Structure

```
Customer_Service_chatbot-/
│
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Preprocessed data
│   └── README.md                 # Data documentation
│
├── models/
│   ├── final_chatbot_model/      # Best trained model
│   ├── checkpoints/              # Training checkpoints
│   └── experiments/              # Experimental models
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py     # Data cleaning & tokenization
│   ├── model.py                  # Model architecture & training
│   ├── evaluation.py             # Metrics calculation
│   └── utils.py                  # Helper functions
│
├── configs/
│   ├── baseline_config.json
│   └── best_config.json
│
├── app.py                        # Gradio web interface
├── chatbot_cli.py               # Command-line interface
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Python dependencies
├── experiments.md               # Detailed experiment results
├── README.md                    # This file
└── LICENSE

```

## Results & Analysis


### Training Insights

1. **Convergence**: The model converged well with the higher learning rate, reaching optimal performance by epoch 5
2. **Generalization**: Close train/val loss indicates good generalization without overfitting
3. **Response Quality**: ROUGE-L improvement of 11.9% shows significantly better sentence structure
4. **Domain Adherence**: Model successfully stays within customer service domain

### Hardware & Training Time
- **GPU**: NVIDIA A100 (40GB)
- **Training Time**: ~15 minutes per epoch with mixed precision
- **Total Training**: ~75 minutes for 5 epochs (best model)
- **Memory Usage**: ~18GB GPU memory during training

## Demo Video

**Video Link**: [Google Drive - Demo Video](https://drive.google.com/drive/folders/1tVAeLDHVO9SToPKyoEpkaFGl_vQ8He3X?usp=sharing)

The demo covers:
- Project overview and motivation
- Dataset exploration
- Model architecture explanation
- Training process and hyperparameter tuning
- Live chatbot interactions
- Performance metrics analysis
- Deployment demonstration


## References

1. Raffel et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
2. Chung et al. (2022). "Scaling Instruction-Finetuned Language Models"
3. Bitext Customer Support Dataset: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
4. Hugging Face Transformers: https://huggingface.co/docs/transformers

##  Contributors
- GitHub: [@j-agbaje](https://github.com/j-agbaje)
- Email: jeremiahagbaje99@gmail.com

##  Acknowledgments

- Google Research for FLAN-T5
- Bitext for the customer support dataset
- Hugging Face for the transformers library
- TensorFlow team for the deep learning framework

---

**Note**: This project was developed as part of a university course on Natural Language Processing and Transformer Models.

For questions or issues, please open an issue on GitHub or contact the repository owner.