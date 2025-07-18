import os
import re
import tempfile
import pandas as pd
import docx2txt
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MUST_HAVE_KEYWORDS = ["sql", "python", "etl", "gcp", "azure", "data modeling"]
NICE_TO_HAVE_KEYWORDS = ["airflow", "spark", "bigquery", "snowflake"]

# === UI Setup ===
st.set_page_config("Resume Shortlister", layout="wide")
st.title("📄 Resume Shortlister with OpenAI")

# === Utility Functions ===
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            return docx2txt.process(tmp.name)
    else:
        return ""

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:4000]
    )
    return response.data[0].embedding

def keyword_score(text):
    text_lower = text.lower()
    must = [kw for kw in MUST_HAVE_KEYWORDS if kw in text_lower]
    if len(must) < 3:
        return None
    nice = [kw for kw in NICE_TO_HAVE_KEYWORDS if kw in text_lower]
    must_score = len(must) / len(MUST_HAVE_KEYWORDS)
    nice_score = len(nice) / len(NICE_TO_HAVE_KEYWORDS)
    return 0.7 * must_score + 0.3 * nice_score

def extract_email_phone(text):
    email_match = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone_match = re.findall(r'(?:(?:\+92)|(?:92)|(?:0))[-\s]?3\d{2}[-\s]?\d{7}', text)

    email = email_match[0] if email_match else ""
    phone = phone_match[0].replace(" ", "").replace("-", "") if phone_match else ""

    if phone.startswith("0"):
        phone = "+92" + phone[1:]
    elif phone.startswith("92"):
        phone = "+92" + phone[2:]
    elif not phone.startswith("+92") and len(phone) == 10:
        phone = "+92" + phone
    return email, phone

# === Uploads ===
jd_file = st.file_uploader("📄 Upload Job Description (.txt)", type=["txt"])
resumes = st.file_uploader("📂 Upload Resumes (.pdf, .docx)", type=["pdf", "docx"], accept_multiple_files=True)

if jd_file and resumes:
    with st.spinner("Processing..."):
        jd_text = jd_file.read().decode("utf-8")
        jd_embedding = get_embedding(jd_text)

        results = []
        for resume in resumes:
            try:
                resume.seek(0)
                text = extract_text(resume)

                kw_score = keyword_score(text)
                if kw_score is None:
                    st.info(f"❌ Skipped {resume.name} (less than 3 must-have skills)")
                    continue

                embedding = get_embedding(text)
                sim = cosine_similarity([jd_embedding], [embedding])[0][0]
                final = 0.7 * sim + 0.3 * kw_score

                email, phone = extract_email_phone(text)

                results.append({
                    "Email": email,
                    "Phone": phone,
                    "Resume": resume.name,
                    "EmbeddingScore": round(sim, 4),
                    "KeywordScore": round(kw_score, 4),
                    "FinalScore": round(final, 4)
                })

            except Exception as e:
                st.error(f"⚠️ Failed to process {resume.name}: {e}")

        if results:
            df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
            st.success("✅ Done! Shortlisted resumes below.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "shortlisted_resumes.csv", "text/csv")
        else:
            st.warning("No resumes matched the minimum skills criteria.")
