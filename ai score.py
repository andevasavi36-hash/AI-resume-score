!pip install nltk scikit-learn PyPDF2
import nltk
import PyPDF2
import os
import re
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from google.colab import files

uploaded = files.upload()
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)
resume_texts = []
resume_names = []

for filename in uploaded.keys():
    with open(filename, 'rb') as f:
        text = extract_text_from_pdf(f)
        cleaned = clean_text(text)
        resume_texts.append(cleaned)
        resume_names.append(filename)
job_description = """
Looking for a Python developer with experience in machine learning,
data analysis, NLP, pandas, scikit-learn, and SQL.
"""

job_description = clean_text(job_description)
documents = [job_description] + resume_texts

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
results = pd.DataFrame({
    'Resume': resume_names,
    'Match Score': similarity_scores
})

results = results.sort_values(by='Match Score', ascending=False)
results
