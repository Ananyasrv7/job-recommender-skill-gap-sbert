import pandas as pd
base = "ESCO dataset - v1.1.1 - classification - en - csv"

occupations = pd.read_csv(f"{base}/occupations_en.csv", low_memory=False)
skills = pd.read_csv(f"{base}/skills_en.csv", low_memory=False)
relations = pd.read_csv(f"{base}/occupationSkillRelations_en.csv", low_memory=False)


print("occupations:", occupations.shape)
print("skills:", skills.shape)
print("relations:", relations.shape)
