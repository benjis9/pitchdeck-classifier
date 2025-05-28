import streamlit as st
import fitz  # PyMuPDF
import openai
import tiktoken
from datetime import datetime
import json
import os
import time
import base64
from openai import RateLimitError, APIError
from PIL import Image
from io import BytesIO

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Pitch Deck Classifier", layout="centered")
st.title("📊 Pitch Deck Classifier")
st.write("Upload a pitch deck PDF and get a VC-style evaluation with scoring and rationale.")

# Setup OpenAI client (v1 API)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------
# 🔒 SIMPLE LOGIN
# ---------------------------
st.sidebar.title("🔒 Login")

# Initialize login state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["login_failed"] = False

# Show login input only if not logged in
if not st.session_state["authenticated"]:
    password = st.sidebar.text_input("Enter password", type="password")
    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
        else:
            st.session_state["login_failed"] = True

    if st.session_state["login_failed"]:
        st.sidebar.error("❌ Incorrect password")

    st.stop()  # Don't render the app until logged in

# Hide login form and show a subtle message after login
st.sidebar.empty()
st.caption("✅ Logged in. You can close the sidebar.")


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
# PDF UTILITIES
# ---------------------------
def get_image_base64(page):
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------------------------
# SUMMARIZATION & SCORING
# ---------------------------
def summarize_slide(text, image_b64, retries=5):
    messages = [
        {"role": "system", "content": "You are a VC analyst. Analyze the pitch slide (text and image)."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize key information related to team, traction, and business model from the slide."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
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
Return only valid JSON using this format (without triple backticks):
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

    # Check usage limit
    count_today, usage, log_file = get_usage_today()
    if count_today >= 5:
        st.error("🚫 Daily usage limit reached for this demo. Please try again tomorrow.")
        st.stop()

    # Extract text and images from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    slides = []
    for page in doc:
        slide_text = page.get_text()
        slide_img_b64 = get_image_base64(page)
        slides.append((slide_text, slide_img_b64))

    # Process slides
    with st.spinner("Summarizing each slide with image + text..."):
        summaries = [summarize_slide(text, img_b64) for text, img_b64 in slides]
        combined_summary = "\n".join(summaries)
        final_score = score_deck(combined_summary)
        st.success("Analysis complete!")
        st.json(final_score)

    # ✅ Record usage
    record_usage(usage, log_file)
