
import pandas as pd
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.title("AI Resume Screening System")


file_path = os.path.join("dataset", "resume_dataset.csv")

data = pd.read_csv(file_path)

vectorizer = TfidfVectorizer(stop_words="english")
resume_vectors = vectorizer.fit_transform(data["Resume"])

job_desc = st.text_area("Enter Job Description")

if st.button("Find Best Candidates"):
    if job_desc:
        job_vector = vectorizer.transform([job_desc])
        scores = cosine_similarity(job_vector, resume_vectors)[0]

        data["Match Score"] = scores
        result = data.sort_values(by="Match Score", ascending=False)

        st.dataframe(result.head(10))
    else:
        st.warning("Enter job description")
