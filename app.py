import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
import openai
import json
import os
import time
import base64
from openai import RateLimitError, APIError
from PIL import Image
from io import BytesIO
from datetime import datetime

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Pitch Deck Classifier", layout="centered")
st.title("📊 Pitch Deck Classifier")
st.write("Upload a pitch deck PDF and get a VC-style evaluation with scoring and rationale.")

# ---------------------------
# 🔒 SIMPLE LOGIN
# ---------------------------
st.sidebar.title("🔒 Login")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["login_failed"] = False

if not st.session_state["authenticated"]:
    password = st.sidebar.text_input("Enter password", type="password")
    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.session_state["login_failed"] = False
        else:
            st.session_state["login_failed"] = True

    if st.session_state["login_failed"]:
        st.sidebar.error("❌ Incorrect password")

if st.session_state["authenticated"]:
    st.sidebar.success("✅ Logged in. You can now close the sidebar.")

    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

    def get_image_base64(page):
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def load_prompt_from_file(filename):
        """
        Load the prompt content from a text file and return it as a string.
        """
        with open(filename, "r") as file:
            return file.read()
    
    def summarize_slide(text, image_b64, previous_summary="", retries=5):
        prompt = load_prompt_from_file("summary_prompt.txt")
        
        messages = [
            {"role": "system", "content": "You are a VC analyst. Analyze the pitch slide (text and image)."},
            {"role": "user", "content": prompt}
        ]

        if previous_summary:
            messages.append({"role": "assistant", "content": previous_summary})

        # Add text and image for current slide
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        })

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
        rubric_prompt = load_prompt_from_file("score_prompt.txt")

        rubric_prompt = rubric_prompt.format(summary=summary)

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
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
            except json.JSONDecodeError as je:
                st.error("❌ Failed to parse JSON from OpenAI response. Showing raw output:")
                raise je
            except (RateLimitError, APIError) as e:
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)
                    st.warning(f"⚠️ Rate limit hit or temporary error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.error("❌ OpenAI API rate limit exceeded after retries.")
                    raise e
    
    def render_html_table(data): 
        html = """
        <style>
        table {
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px;
            background-color: #f0f0f0; 
        }
        th, td {
            border: 1px solid #ccc; 
            padding: 8px; 
            text-align: center;
            color: #000; 
        }
        th {background-color: #e0e0e0;}
        td {background-color: #f0f0f0;}
        td.score-1 {background-color: #98fb98;} /* Light green for score 1 */
        td.score-0_5 {background-color: #ffffe0;} /* Light yellow for score 0.5 */
        td.score-0 {background-color: #f8d7da;} /* Light red for score 0 */
        </style>
        <table>
            <tr>
                <th rowspan="2">#</th>
                <th colspan="2">Team</th>
                <th colspan="2">Business Model</th>
                <th colspan="2">Traction</th>
            </tr>
            <tr>
                <th>Score</th><th>Rationale</th>
                <th>Score</th><th>Rationale</th>
                <th>Score</th><th>Rationale</th>
            </tr>
        """
        for i in range(1, 5):
            row = data[str(i)]
            def score_class(score):
                return f"score-{str(score).replace('.', '_')}"  # Generate class based on score value
    
            html += f"""
            <tr>
                <td>{i}</td>
                <td class='{score_class(row['Team']['score'])}'>{row['Team']['score']}</td>
                <td>{row['Team']['rationale']}</td>
                <td class='{score_class(row['Business Model']['score'])}'>{row['Business Model']['score']}</td>
                <td>{row['Business Model']['rationale']}</td>
                <td class='{score_class(row['Traction']['score'])}'>{row['Traction']['score']}</td>
                <td>{row['Traction']['rationale']}</td>
            </tr>
            """
        html += "</table>"
        components.html(html, height=1200, width=1600)
    
    uploaded_file = st.file_uploader("Upload a pitch deck (PDF)", type=["pdf"])

    st.text("Evaluation Criteria")
    html_table = load_prompt_from_file("criteria_table.txt")
    components.html(html_table, height=1200, width=1600)

    if uploaded_file:
        st.success("PDF uploaded. Reading content...")

        count_today, usage, log_file = get_usage_today()
        if count_today >= 25:
            st.error("🚫 Daily usage limit reached for this demo. Please try again tomorrow.")
            st.stop()

        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        slides = []
        for page in doc:
            slide_text = page.get_text()
            slide_img_b64 = get_image_base64(page)
            slides.append((slide_text, slide_img_b64))

        # Process slides in batches to ensure context is kept without hitting token limits
        context = ""
        batch_size = 10  # Process 10 slides at a time (adjust based on token limit)

        with st.spinner("Summarizing each slide with image + text..."):
            summaries = []
            for i in range(0, len(slides), batch_size):
                batch = slides[i:i + batch_size]
                batch_summary = ""
                for text, img_b64 in batch:
                    batch_summary += summarize_slide(text, img_b64, previous_summary=context)
                summaries.append(batch_summary)
                context = batch_summary  # Update context for next batch

            # Combine all the slide summaries
            combined_summary = "\n".join(summaries)
            score_data = score_deck(combined_summary)
            st.success("Analysis complete!")
            render_html_table(score_data)

        record_usage(usage, log_file)
