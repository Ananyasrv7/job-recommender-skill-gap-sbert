import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Load fine-tuned model from checkpoints(containing trained weights)
model = SentenceTransformer("checkpoints")   #loading trained SBERT model to convert text into numerical embeddings
occupations_df = pd.read_csv("occupations_en.csv")  #loading ESCO occupations data
# Combining job title and description
job_texts = (
    occupations_df["preferredLabel"].fillna("") + " " +   #takes job title
    occupations_df["description"].fillna("")              #takes job desc
).tolist()             #Replaces missing values with empty strings; Concatenates title + description; Converts result into a Python list

job_embeddings = model.encode(job_texts, convert_to_numpy=True)  #creating job embeddings(Converts each job text into a vector)

#creating (sample) resume input
resume_text = """
I have experience in Python, machine learning, data analysis,
building predictive models, and working with large datasets.
"""
resume_embedding = model.encode([resume_text], convert_to_numpy=True)
similarity_scores = cosine_similarity(resume_embedding, job_embeddings)[0]   #cosine similarity calculation(0-not similar, 1-similar)
occupations_df["similarity"] = similarity_scores
#sort and display results
top_matches = occupations_df.sort_values(
    by="similarity", ascending=False
).head(10)

print("\nTop Job Matches:\n")
for _, row in top_matches.iterrows():
    print(f"{row['preferredLabel']}  →  Similarity: {row['similarity']:.4f}")