import streamlit as st
import fitz  # PyMuPDF
import openai
import tiktoken
from datetime import datetime
import json
import os
import time
from openai import RateLimitError, APIError


# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Pitch Deck Classifier", layout="centered")
st.title("📊 Pitch Deck Classifier")
st.write("Upload a pitch deck PDF and get a VC-style evaluation with scoring and rationale.")

# Setup OpenAI client (v1 API)
client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "your-api-key-here"))

# ---------------------------
# 🔒 SIMPLE LOGIN SYSTEM
# ---------------------------
st.sidebar.title("🔒 Login")
password = st.sidebar.text_input("Enter password", type="password")
if password != st.secrets.get("APP_PASSWORD", "your-test-password"):
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# ---------------------------
# 🚫 DAILY USAGE LIMIT
# ---------------------------
def get_usage_today():
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = "usage_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            usage = json.load(f)
    else:
        usage = {}
    return usage.get(today, 0), usage, log_file

def record_usage(usage, log_file):
    today = datetime.now().strftime("%Y-%m-%d")
    usage[today] = usage.get(today, 0) + 1
    with open(log_file, "w") as f:
        json.dump(usage, f)


# ---------------------------
# TOKEN & CHUNKING UTILITIES
# ---------------------------
def estimate_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=3000, overlap=300):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(encoding.decode(chunk))
        i += max_tokens - overlap
    return chunks

# ---------------------------
# SUMMARIZATION & SCORING
# ---------------------------
def summarize_chunk(chunk, retries=3):
    prompt = f"""
You are a VC analyst. Read the following section from a startup pitch deck and summarize key information related to team, traction, and business model:

{chunk}
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a VC investment analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        except (RateLimitError, APIError) as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                st.warning(f"⚠️ Rate limit hit or temporary error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error("❌ OpenAI API rate limit exceeded. Please try again later.")
                raise e


def score_deck(summary, retries=3):
    rubric_prompt = f"""
You are a VC analyst. Score this startup based on the following rubric:

TEAM:
1. Relevant experience?
2. Worked together before?
3. Previous founder?

BUSINESS MODEL:
1. Scalable?
2. Upsell potential?
3. Resilient to external shocks?

TRACTION:
1. Initial customers?
2. Rapid growth?
3. Customer retention?

Give a score for each question between 0 and 1.
Return output in this format:
{{
    "team": {{"1": _, "2": _, "3": _}},
    "business_model": {{"1": _, "2": _, "3": _}},
    "traction": {{"1": _, "2": _, "3": _}},
    "total_score": _,
    "rationale": "Your explanation..."
}}

Startup deck summary:
{summary}
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a VC analyst."},
                    {"role": "user", "content": rubric_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        except (RateLimitError, APIError) as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                st.warning(f"⚠️ Rate limit hit or temporary error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error("❌ OpenAI API rate limit exceeded. Please try again later.")
                raise e


# ---------------------------
# MAIN APP FLOW
# ---------------------------
uploaded_file = st.file_uploader("Upload a pitch deck (PDF)", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded. Reading content...")

    # Check usage limit
    count_today, usage, log_file = get_usage_today()
    if count_today >= 5:
        st.error("🚫 Daily usage limit reached for this demo. Please try again tomorrow.")
        st.stop()

    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    token_count = estimate_tokens(text)
    st.write(f"📏 Estimated token count: **{token_count}**")

    if token_count < 6000:
        st.info("✅ Small file — processing in one go...")
        with st.spinner("Scoring deck..."):
            summary = summarize_chunk(text)
            final_score = score_deck(summary)
            st.success("Analysis complete!")
            st.json(final_score)
    else:
        st.warning("⚠️ Large file — processing in chunks...")
        with st.spinner("Summarizing deck in sections..."):
            chunks = chunk_text(text)
            summaries = [summarize_chunk(chunk) for chunk in chunks]
            combined_summary = "\n".join(summaries)
            final_score = score_deck(combined_summary)
            st.success("Analysis complete!")
            st.json(final_score)

    # ✅ Record usage
    record_usage(usage, log_file)
