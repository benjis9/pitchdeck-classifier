import streamlit as st
import fitz  # PyMuPDF
import openai
import tiktoken
from datetime import datetime
import json
import os
import time
import base64
from io import BytesIO
from PIL import Image
from openai import RateLimitError, APIError

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Pitch Deck Classifier", layout="centered")
st.title("📊 Pitch Deck Classifier")
st.write("Upload a pitch deck PDF and get a VC-style evaluation with scoring and rationale.")

# Setup OpenAI client (v1 API)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"))

# ---------------------------
# 🔒 SIMPLE LOGIN SYSTEM
# ---------------------------
st.sidebar.title("🔒 Login")
password = st.sidebar.text_input("Enter password", type="password")
if password != os.environ.get("APP_PASSWORD", "your-test-password"):
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

# ---------------------------
# IMAGE ENCODING
# ---------------------------
def encode_image_to_base64(pix):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ---------------------------
# SUMMARIZATION WITH IMAGE + TEXT
# ---------------------------
def summarize_slide(text, image_base64, retries=5):
    messages = [
        {"role": "system", "content": "You are a VC analyst skilled in evaluating startup pitch decks with both text and image inputs."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Please analyze this slide from a startup pitch deck. Summarize information related to the team, traction, and business model.\n\n{text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except (RateLimitError, APIError) as e:
            if attempt < retries - 1:
                wait_time = 5 * (attempt + 1)
                st.warning(f"⚠️ Rate limit hit or temporary error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error("❌ OpenAI API rate limit exceeded after retries.")
                raise e

# ---------------------------
# SCORING
# ---------------------------
def score_deck(summary, retries=5):
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
                wait_time = 5 * (attempt + 1)
                st.warning(f"⚠️ Rate limit hit or temporary error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error("❌ OpenAI API rate limit exceeded after retries.")
                raise e

# ---------------------------
# MAIN APP FLOW
# ---------------------------
uploaded_file = st.file_uploader("Upload a pitch deck (PDF)", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded. Reading content...")

    count_today, usage, log_file = get_usage_today()
    if count_today >= 5:
        st.error("🚫 Daily usage limit reached for this demo. Please try again tomorrow.")
        st.stop()

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page_summaries = []

    with st.spinner("Analyzing each slide..."):
        for i, page in enumerate(doc):
            text = page.get_text()
            pix = page.get_pixmap(dpi=150)
            image_b64 = encode_image_to_base64(pix)
            summary = summarize_slide(text, image_b64)
            page_summaries.append(summary)

    combined_summary = "\n".join(page_summaries)

    with st.spinner("Scoring full pitch deck..."):
        final_score = score_deck(combined_summary)
        st.success("Analysis complete!")
        st.json(final_score)

    record_usage(usage, log_file)
