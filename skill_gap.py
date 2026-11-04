
import os

# Force CPU and avoid Apple's MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "true"

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------- Load model and data --------------------
model = SentenceTransformer(
    "esco-sbert-test/checkpoint-7302",
    device="cpu"
)
model.to("cpu")

base = "ESCO dataset - v1.1.1 - classification - en - csv"

skills_df = pd.read_csv(f"{base}/skills_en.csv")
relations = pd.read_csv(f"{base}/occupationSkillRelations_en.csv")
occupations = pd.read_csv(f"{base}/occupations_en.csv")


def get_job_skills(job_title, relation_type="essential"):
    """Return required skills for a job title from ESCO."""

    job = occupations[
        occupations["preferredLabel"].str.lower() == job_title.lower()
    ]

    if job.empty:
        print(f'Job "{job_title}" not found in dataset.')
        return []

    job_id = job.iloc[0]["conceptUri"]

    skill_links = relations[
        (relations["occupationUri"] == job_id) &
        (relations["relationType"] == relation_type)
    ]

    skill_ids = skill_links["skillUri"]

    job_skills = skills_df[
        skills_df["conceptUri"].isin(skill_ids)
    ]

    return job_skills["preferredLabel"].tolist()


def skill_gap(job_title, user_skills, threshold=0.60):
    """
    Identify missing skills for a target job.
    Batch-encodes all skills at once.
    """

    required_skills = get_job_skills(job_title)

    if not required_skills:
        return {
            "missing": [],
            "matched": [],
            "threshold": threshold
        }

    if not user_skills:
        return {
            "missing": [(s, 0.0) for s in required_skills],
            "matched": [],
            "threshold": threshold
        }

    # Explicitly encode on CPU
    required_embs = model.encode(
        required_skills,
        convert_to_tensor=True,
        show_progress_bar=False,
        device="cpu"
    )

    user_embs = model.encode(
        user_skills,
        convert_to_tensor=True,
        show_progress_bar=False,
        device="cpu"
    )

    # Similarity matrix: rows = required skills, cols = user skills
    sim_matrix = util.cos_sim(required_embs, user_embs)

    missing = []
    matched = []

    for i, skill in enumerate(required_skills):
        best_score = sim_matrix[i].max().item()

        if best_score >= threshold:
            matched.append((skill, round(best_score, 3)))

        elif best_score >= 0.25:
            missing.append((skill, round(best_score, 3)))

    return {
        "missing": missing,
        "matched": matched,
        "threshold": threshold
    }


# -------------------- Main --------------------
if __name__ == "__main__":

    job = input("Enter target job role: ").strip()

    skills_input = input(
        "Enter your skills (comma-separated): "
    ).strip()

    user_skills = [
        s.strip()
        for s in skills_input.split(",")
        if s.strip()
    ]

    print(f'\nAnalysing skill gap for: "{job}"')
    print(f"Your skills: {user_skills}\n")

    result = skill_gap(job, user_skills)

    print(f"Similarity threshold: {result['threshold']}")

    print(f"\nMatched skills ({len(result['matched'])}):")

    for skill, score in result["matched"]:
        print(f"  +  {skill}  (similarity: {score})")

    print(f"\nMissing skills ({len(result['missing'])}):")

    for skill, score in result["missing"]:
        print(f"  -  {skill}  (best match: {score})")

    if not result["missing"]:
        print("  None — you meet all essential skill requirements!")
