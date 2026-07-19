# clinical-text-NER

# Does Biomedical Pretraining Matter for Clinical NER?
### A From-Scratch vs. Fine-Tuned Comparison

Clinical text, discharge summaries, progress notes, and case reports are critical for a patient's medical record. This data contains symptoms, diagnoses, medications, and procedures as unstructured free text, which makes it hard to search, aggregate, or feed into any downstream system. Named Entity Recognition (NER) is the general first step for turning this free text into structured data by tagging spans of text with what kind of clinical concept they represent. The applications of this vary, from pulling structured data out of patients' electronic health records, to building research cohorts from case reports at bigger scales.

In this notebook, I investigate: how much of a model's ability to do clinical NER actually comes from pretraining on clinical on biomedial text, as compared to just having a transformer architecture trained on the task itself?
In my attempt to answer this question, I built a transformer encoder from scratch, multi-head self attention, feedforward layers, pre-norm residual blocks with no pretrained weights, and trained it directly on the target task.
<br>
I then fine tuned Bio_ClinicalBERT, which is pretraind on clinical text specifically, on the exact same data, split and label set such that the only real variable between the models is the difference in pretraining.
<br>
The dataset is MACCROBAT ~200 annotated clinical case reports covering 41 BIO tagged entity types. This is too small of a dataset to train anything from scratch, making it a clean test of exactly the impact pretraining has in a low data clinical setting.

## Approach
- Built a transformer encoder from scratch (multi-head self-attention, 
  feedforward, pre-norm residual blocks), no pretrained weights
- Fine-tuned Bio_ClinicalBERT on the identical data split and label set
- Validated the pipeline using  a memorization overfit test before trusting 
  training results

## Results

| Entity Type           | Scratch F1 | Bio_ClinicalBERT F1 |
|------------------------|-----------|----------------------|
| Diagnostic_procedure   | 0.11      | 0.67                 |
| Lab_value              | 0.13      | 0.62                 |
| Sign_symptom           | 0.13      | 0.60                 |
| Biological_structure   | 0.10      | 0.64                 |
| Medication             | 0.02      | 0.67                 |
| Dosage                 | 0.00      | 0.72                 |
| Disease_disorder       | 0.01      | 0.34                 |
| Severity               | 0.58      | 0.00                 |

Micro F1: 0.12 → 0.54. Weighted F1: 0.13 → 0.53.

## Limitations
Based on the metrics for the scratch model, we can see that for high support tags like Diagnostic_procedure, Lab_value, and Biological_structure, precision is low as compared to recall, which is indicative of a model that is over predicting common tags by flagging a lot of false positives. Contrastingly, BERT's precision and recall scores are more balanced on the same tags showing that its purposely discriminating between them rather than guessing broadly.

Despite the expected results, there is one exception: For Severity, the scratch model beats BERT with respective f1 scores of 0.58 vs 0.00. This is probably because the BERT model has a truncation of 512, when the average document length is ~720, meaning context is lost due to loss of document length.

## Run it
Built in Colab (T4 GPU). Open `Clinical_Text_NER.ipynb`, run top to bottom.
