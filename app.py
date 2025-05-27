import streamlit as st
import fitz  # PyMuPDF
import openai
import tiktoken
import math

# Use the new OpenAI v1.0+ client
client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "your-api-key-here"))

# Setup Streamlit page
st.set_page_config(page_title="Pitch Deck Classifier", layout="centered")
st.title("📊 Pitch Deck Classifier")
st.write("Upload a pitch deck PDF and get a VC-style evaluation with scoring and rationale.")

# Token encoder
def estimate_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Chunking utility
def chunk_text(text, max_tokens=3000, overlap=300):
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(encoding.decode(chunk))
        i += max_tokens - overlap
    return chunks

# Summarize a chunk
def summarize_chunk(chunk):
    prompt = f"""
You are a VC analyst. Read the following section from a startup pitch deck and summarize key information related to team, traction, and business model:

{chunk}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a VC investment analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Score the full summarized deck
def score_deck(summary):
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
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a VC analyst."},
            {"role": "user", "content": rubric_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Upload and process PDF
uploaded_file = st.file_uploader("Upload a pitch deck (PDF)", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded. Reading content...")

    # Extract text
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    token_count = estimate_tokens(text)
    st.write(f"📏 Estimated token count: **{token_count}**")

    # Decide to chunk or not
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
