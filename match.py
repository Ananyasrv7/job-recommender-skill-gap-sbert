from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer("esco-sbert")   # Load the trained model

base = "ESCO dataset - v1.1.1 - classification - en - csv"   # Load ESCO occupations
occupations = pd.read_csv(f"{base}/occupations_en.csv", low_memory=False)   #contains all ESCO jobs

job_texts = occupations["preferredLabel"].fillna("").tolist()   #convert jobs into lists
job_embeddings = model.encode(job_texts, convert_to_tensor=True)   #converts job into embeddings which maks each job into a vector

print("\nModel loaded successfully!")
print("Total jobs:", len(job_texts))

resume = input("\nEnter resume / skills text:\n")   #user input

resume_embedding = model.encode(resume, convert_to_tensor=True)  #converting resume to embedding
scores = util.cos_sim(resume_embedding, job_embeddings)[0]   #computing similarity (1-identical, 0-unmatched)

# Change this:
top_results = scores.topk(5)

# To this — filters out weak matches below your tuned threshold:
THRESHOLD = 0.45
top_results = scores.topk(10)

print("\nTop matching jobs:\n")
for score, idx in zip(top_results.values, top_results.indices):
    if float(score) >= THRESHOLD:
        print(f"{job_texts[int(idx)]}  →  {float(score):.4f}")