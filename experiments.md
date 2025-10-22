## Experiment Results

| Experiment | LR | Batch | Weight Decay | Warmup | Train Loss | Val Loss | Perplexity | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------|-----|-------|--------------|--------|------------|----------|------------|------|---------|---------|---------|
| **Baseline** | 5e-5 | 8 | 0.01 | 0 | 0.8859 | 0.7600 | 2.20 | 20.28 | 51.69 | 27.93 | 37.34 |
| **Higher LR**  | 1e-4 | 8 | 0.01 | 0 | 0.7877 | 0.6960 | 2.05 | 21.79 | 53.66 | 29.79 | 38.82 |
| **Small Batch** | 3e-5 | 4 | 0.01 | 0 | 0.9118 | 0.7776 | 2.27 | 20.07 | 52.35 | 27.93 | 37.58 |
| **Lower Weight Decay** | 5e-5 | 8 | 0.001 | 0 | 0.8855 | 0.7595 | 2.19 | 19.42 | 50.97 | 27.11 | 36.61 |
| **With Warmup** | 5e-5 | 8 | 0.01 | 500 | 0.8903 | 0.7631 | 2.20 | 19.73 | 51.77 | 27.29 | 37.18 |


## Extended Training Results (5 Epochs)
After extending training for 5 epochs using the best setup:
- **Validation Loss: **0.66** (↓13.2% from baseline)**
- Perplexity: **1.97** (↓10.5% from baseline)  
- BLEU Score: **23.01** (↑13.5% from baseline)  
- ROUGE-1: **55.46** (↑7.3% from baseline)  
- ROUGE-2: **31.22** (↑11.8% from baseline)  
- ROUGE-L: **41.79** (↑11.9% from baseline)  

## Training Insights
1. A **higher learning rate (1e-4)** helped the model learn faster and reach a lower loss.  
2. **Five epochs** allowed the model to stabilize and improved language generation quality.  
3. **Batch size of 8** gave a balance between speed and stability — smaller batches led to noisier gradients.  
4. **Weight decay of 0.01** worked best for regularization and avoided overfitting.  
5. **Warmup steps** had only a small effect, showing the model trained well even without warmup.  
6. The model **learned to produce fluent, relevant, and polite customer responses**, reducing repetition and improving coherence.  

## Evaluation and Performance Metrics
- **Validation Loss:** Measures how well the model predicts unseen data. Lower is better.  
- **Perplexity:** Tests how confident the model is when generating text. A lower number means better prediction accuracy.  
- **BLEU Score:** Checks how similar the model’s responses are to real answers, using word overlap.  
- **ROUGE Scores (1, 2, L):** Measure overlap in words, short phrases, and longest common sequences between predictions and references.  

The results show that fine-tuning the **FLAN-T5 Base** model produced strong and natural customer service responses. The model demonstrated improved accuracy, clarity, and coherence across all evaluation metrics, confirming that tuning the learning rate and training longer were effective strategies.
