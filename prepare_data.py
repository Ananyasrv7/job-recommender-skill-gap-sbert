import pandas as pd

#Load ESCO files 
base = "ESCO dataset - v1.1.1 - classification - en - csv"

occupations = pd.read_csv(f"{base}/occupations_en.csv", low_memory=False)
skills      = pd.read_csv(f"{base}/skills_en.csv", low_memory=False)
relations   = pd.read_csv(f"{base}/occupationSkillRelations_en.csv", low_memory=False)

print("occupations:", occupations.shape)
print("skills:     ", skills.shape)
print("relations:  ", relations.shape)

#Keep useful columns 
occupations = occupations[["conceptUri", "preferredLabel", "description"]].copy()
skills      = skills[["conceptUri", "preferredLabel"]].copy()
relations   = relations[["occupationUri", "skillUri", "relationType"]].copy()

#Drop missing values 
occupations = occupations.dropna(subset=["preferredLabel", "description"])
occupations = occupations[occupations["description"].str.strip() != ""]
skills      = skills.dropna(subset=["preferredLabel"])

# Dataset 1: title ↔ description pairs 
# FIX: The original code produced only a single "text" column used as (text, text)
# which gave the model zero signal. We now create explicit (sentence1, sentence2)
# pairs that the training script can consume properly.
title_desc_pairs = pd.DataFrame({
    "sentence1": occupations["preferredLabel"].str.strip(),
    "sentence2": occupations["description"].str.strip(),
    "pair_type": "title_desc"
})

merged = relations.merge(
    occupations[["conceptUri", "preferredLabel"]],
    left_on="occupationUri", right_on="conceptUri"
).merge(
    skills[["conceptUri", "preferredLabel"]],
    left_on="skillUri", right_on="conceptUri",
    suffixes=("_occ", "_skill")
)

essential = merged[merged["relationType"] == "essential"].copy()
optional  = merged[merged["relationType"] == "optional"].copy()

essential["sentence1"] = essential["preferredLabel_occ"].str.strip()
essential["sentence2"] = "requires skill " + essential["preferredLabel_skill"].str.strip()
essential["pair_type"] = "essential_skill"

optional["sentence1"] = optional["preferredLabel_occ"].str.strip()
optional["sentence2"] = "may require skill " + optional["preferredLabel_skill"].str.strip()
optional["pair_type"] = "optional_skill"

skill_pairs = pd.concat([
    essential[["sentence1", "sentence2", "pair_type"]],
    optional[["sentence1", "sentence2", "pair_type"]]
], ignore_index=True)

all_pairs = pd.concat([title_desc_pairs, skill_pairs], ignore_index=True)

all_pairs = all_pairs.dropna(subset=["sentence1", "sentence2"])
all_pairs = all_pairs[
    (all_pairs["sentence1"].str.strip() != "") &
    (all_pairs["sentence2"].str.strip() != "")
]

all_pairs.to_csv("training_pairs.csv", index=False)

print(f"\nSaved {len(all_pairs)} training pairs to training_pairs.csv")
print(all_pairs["pair_type"].value_counts().to_string())
print("\nSample rows:")
print(all_pairs.sample(5)[["sentence1", "sentence2", "pair_type"]].to_string(index=False))