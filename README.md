# Job Recommender System: Skill-Gap Analysis with Fine-Tuned SBERT

Fine-tunes a sentence-transformers model on ESCO occupation-skill pairs and uses it to compare a user's skills against a target job role and report the gaps.

## How it works
1. `prepare_data.py` builds occupation-skill training pairs from the ESCO dataset
2. `train_sbert.py` fine-tunes SBERT on those pairs
3. `skill_gap.py` compares a user's skills to a target role using the fine-tuned model
4. `evaluate.py` produces the evaluation charts

## Setup
```bash
pip install -r requirements.txt
```
Download the ESCO dataset (v1.1.1, CSV, English) from https://esco.ec.europa.eu and place it in the project folder.

## Usage
```bash
python prepare_data.py
python train_sbert.py
python skill_gap.py
```

## Results
Evaluation charts (accuracy, precision, recall, F1, ROC curve, confusion matrix) are in the repo root.

## Notes
Trained model weights and the ESCO dataset are not included because of file size.

## Author
Ananya Srivastava
Shivam Mittal
