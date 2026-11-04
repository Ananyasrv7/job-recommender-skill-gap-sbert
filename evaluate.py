"""
evaluate.py — run this after training to measure real metrics and find
               the optimal similarity threshold for your use case.

Usage:
    python3 evaluate.py

Outputs:
    - Precision / Recall / F1 / Accuracy at every threshold
    - 6 separate charts: F1, Precision, Recall, Accuracy, Confusion Matrix, ROC Curve
    - AUC score
    - Best threshold value to use in match.py and skill_gap.py
"""

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    roc_curve,
    roc_auc_score,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer("esco-sbert")

# ── Load ESCO data ──────────────────────────────────────────────────────────
base = "ESCO dataset - v1.1.1 - classification - en - csv"

occupations = pd.read_csv(f"{base}/occupations_en.csv", low_memory=False)
occupations = occupations.dropna(subset=["preferredLabel", "description"])
occupations = occupations[occupations["description"].str.strip() != ""].reset_index(drop=True)

# ── Build evaluation pairs ──────────────────────────────────────────────────
random.seed(42)
occ_list = occupations.to_dict("records")
random.shuffle(occ_list)
eval_occ = occ_list[:1000]

sentences1, sentences2, labels = [], [], []

for i, row in enumerate(eval_occ):
    title    = str(row["preferredLabel"]).strip()
    desc     = str(row["description"]).strip()
    sentences1.append(title)
    sentences2.append(desc)
    labels.append(1)

    j        = random.choice([k for k in range(len(eval_occ)) if k != i])
    neg_desc = str(eval_occ[j]["description"]).strip()
    sentences1.append(title)
    sentences2.append(neg_desc)
    labels.append(0)

labels = np.array(labels)
print(f"\nEvaluation set: {len(labels)} pairs ({labels.sum()} positive, {(1-labels).sum()} negative)")

# Encode 
print("Encoding sentences...")
emb1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
emb2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
cos_scores = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()

# Threshold sweep 
thresholds = np.arange(0.20, 0.95, 0.025)
results = []

for t in thresholds:
    preds    = (cos_scores >= t).astype(int)
    f1       = f1_score(labels, preds, zero_division=0)
    accuracy = accuracy_score(labels, preds)
    prec     = (preds & labels).sum() / max(preds.sum(), 1)
    rec      = (preds & labels).sum() / max(labels.sum(), 1)
    results.append({
        "threshold": round(float(t), 3),
        "f1":        round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "accuracy":  round(float(accuracy), 4),
    })

results_df  = pd.DataFrame(results)
best_row    = results_df.loc[results_df["f1"].idxmax()]
best_thresh = best_row["threshold"]

print("\nThreshold sweep results:")
print(results_df.to_string(index=False))

print(f"\n{'='*55}")
print(f"Best threshold : {best_thresh}")
print(f"F1             : {best_row['f1']}")
print(f"Precision      : {best_row['precision']}")
print(f"Recall         : {best_row['recall']}")
print(f"Accuracy       : {best_row['accuracy']}")
print(f"{'='*55}")

# Classification report & confusion matrix 
best_preds = (cos_scores >= best_thresh).astype(int)

print("\nClassification report at best threshold:")
print(classification_report(labels, best_preds, target_names=["Not a match", "Match"]))

cm = confusion_matrix(labels, best_preds)
print("Confusion matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ROC Curve calculation 
# cos_scores are used directly as probability scores (no threshold needed here)
fpr, tpr, roc_thresholds = roc_curve(labels, cos_scores)
auc_score = roc_auc_score(labels, cos_scores)

print(f"\nAUC Score: {auc_score:.4f}")

# Helper: shared chart style 
def styled_chart(ax, x, y, color, title, ylabel, best_thresh, best_val):
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.axvline(best_thresh, color="red", linestyle="--", linewidth=1.5,
               label=f"Best threshold ({best_thresh})")
    ax.scatter([best_thresh], [best_val], color="red", zorder=5, s=60)
    ax.annotate(f"{best_val:.4f}", xy=(best_thresh, best_val),
                xytext=(best_thresh + 0.03, best_val - 0.06),
                fontsize=9, color="red")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0.18, 0.97)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

x = results_df["threshold"]

# Chart 1: F1 Score 
fig1, ax1 = plt.subplots(figsize=(7, 5))
styled_chart(ax1, x, results_df["f1"], "#2196F3",
             "F1 Score vs Threshold", "F1 Score",
             best_thresh, best_row["f1"])
plt.tight_layout()
plt.savefig("chart_f1.png", dpi=150)
print("\nSaved: chart_f1.png")
plt.show()

# Chart 2: Precision 
fig2, ax2 = plt.subplots(figsize=(7, 5))
styled_chart(ax2, x, results_df["precision"], "#4CAF50",
             "Precision vs Threshold", "Precision",
             best_thresh, best_row["precision"])
plt.tight_layout()
plt.savefig("chart_precision.png", dpi=150)
print("Saved: chart_precision.png")
plt.show()

# Chart 3: Recall
fig3, ax3 = plt.subplots(figsize=(7, 5))
styled_chart(ax3, x, results_df["recall"], "#FF9800",
             "Recall vs Threshold", "Recall",
             best_thresh, best_row["recall"])
plt.tight_layout()
plt.savefig("chart_recall.png", dpi=150)
print("Saved: chart_recall.png")
plt.show()

# Chart 4: Accuracy 
fig4, ax4 = plt.subplots(figsize=(7, 5))
styled_chart(ax4, x, results_df["accuracy"], "#9C27B0",
             "Accuracy vs Threshold", "Accuracy",
             best_thresh, best_row["accuracy"])
plt.tight_layout()
plt.savefig("chart_accuracy.png", dpi=150)
print("Saved: chart_accuracy.png")
plt.show()

# ── Chart 5: Confusion Matrix ────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not a match", "Match"])
disp.plot(ax=ax5, colorbar=False, cmap="Blues")
ax5.set_title(f"Confusion Matrix (threshold = {best_thresh})", fontsize=13,
              fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("chart_confusion_matrix.png", dpi=150)
print("Saved: chart_confusion_matrix.png")
plt.show()

# ── Chart 6: ROC Curve ───────────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(7, 5))
ax6.plot(fpr, tpr, color="#E53935", linewidth=2.5,
         label=f"ROC Curve (AUC = {auc_score:.4f})")
ax6.set_xlabel("False Positive Rate", fontsize=11)
ax6.set_ylabel("True Positive Rate (Recall)", fontsize=11)
ax6.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=10)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1.05)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart_roc_curve.png", dpi=150)
print("Saved: chart_roc_curve.png")
plt.show()

print(f"\nUse threshold = {best_thresh} in match.py and skill_gap.py")
print("All 6 charts saved separately.")