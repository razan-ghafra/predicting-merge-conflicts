# Predicting Merge Conflicts using Deep Active Learning

## Overview
This project predicts the likelihood of merge conflicts in software 
repositories by combining social and technical factors — merging 
merge-scenario-level data with aggregated developer-level data. 
A deep learning model classifies each merge scenario as conflict (1) 
or no-conflict (0).

## Methodology
1. **Data Preparation**: Merged scenario-level and developer-level datasets, 
   followed by exploratory data analysis.
2. **Baseline**: A dummy classifier (always predicts "no conflict") was used 
   as benchmark — showing 95% accuracy but 0% precision on conflicts, 
   highlighting the class imbalance problem.
3. **Imbalance Handling**: Compared Random Undersampling (RUS) vs. SMOTE 
   oversampling with an MLP model.
4. **Active Learning**: Applied deep active learning to improve prediction 
   efficiency with fewer labeled samples.

## Results
| Method | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Dummy Baseline | 95% | 0% | 0% | — |
| Random Undersampling + MLP | 93.67% | 90.51% | 97.44% | 93.85% |
| SMOTE + MLP | 93.2% | 42% | — | — |

**Key takeaway**: Random Undersampling outperformed SMOTE, achieving 
better balance between precision and recall while avoiding synthetic 
data artifacts — improving prediction accuracy by 25% over baseline.

## Tech Stack
Python, Deep Learning (Active Learning), Scikit-learn, Pandas

## Author
Razan Salem Bin Ghafrah — [LinkedIn](your-link)
