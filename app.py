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

    def summarize_slide(text, image_b64, previous_summary="", retries=5):
    messages = [
        {"role": "system", "content": "You are a VC analyst. Analyze the pitch slide (text and image)."},
        {"role": "user", "content": """
            Summarize the key information related to Team, Business Model, and Traction from the slide. 
            Consider the following questions for each slide, and use the examples below to guide your assessment:

            **Team Evaluation:**
            1. **Does the founding team appear to be complete?** 
                - Look for whether the team is a single founder or a group. If the team consists of only one founder, it may be seen as a **red flag** unless they have strong experience or advisors. Teams with multiple founders or a well-rounded advisory board are often seen as more robust. 
                - Example: "The team lacks co-founders, which may hinder collaboration and strategic decision-making."

            2. **Does the team have the strength and qualifications to compete in this particular industry or market?**
                - Evaluate whether the team’s **background and skills** align with the industry. A strong team should have relevant experience and expertise.
                - Example: "The founder has some experience in the tech space but lacks expertise in the fashion industry."

            3. **Does the founding team have relevant industry experience?**
                - The more **direct experience** the team has in the industry, the higher the evaluation. If their experience is more **general** or from a **different industry**, their chances of success might be lower.
                - Example: "The team has industry-relevant experience and the founder has built a successful startup before."

            4. **Have the founders previously worked together?**
                - Teams that have worked together in the past tend to perform better, especially if they’ve succeeded together before. Lack of prior collaboration might be a **concern**.
                - Example: "The founder is working solo, which could be a **challenge** in building a strong, collaborative team."

            **Business Model Evaluation:**
            1. **Is the business model scalable?**
                - Evaluate if the business can grow quickly without **substantial increases in costs**. Scalable models like **SaaS** or **digital-first businesses** tend to score higher, while **hardware or capital-intensive models** score lower.
                - Example: "The business model is a digital marketplace, which is scalable."

            2. **Does the business have the potential to add new product lines, services, or upsell to existing customers?**
                - A **scalable business model** that allows for product/service **expansion** and **upselling** is more likely to succeed. Look for opportunities for **growth** beyond the initial offering.
                - Example: "The business has the potential to diversify its product line and cross-sell additional services."

            3. **Is the business model resilient to external shocks?**
                - Consider whether the business is **immune** to risks like **regulatory changes** or **geopolitical factors**. Resilient business models like **software-only** businesses or those not reliant on global supply chains are less vulnerable.
                - Example: "The business model is highly vulnerable to changes in international shipping regulations."

            4. **Does the business create a new market or unlock a 'shadow market'?**
                - If the business is **innovative** and disrupts an existing market or creates a **new category**, it’s more likely to succeed. If it’s entering an already saturated market, it may face **more competition**.
                - Example: "The business targets a niche market in sustainable fashion, which is currently underserved."

            **Traction Evaluation:**
            1. **Does the business have initial customers or users?**
                - Businesses with **early customers** are generally more likely to succeed. Look for **proof of market fit** (e.g., paying customers, partnerships).
                - Example: "The business has initial customers, and it’s clear that there is demand for the product."

            2. **Is the business demonstrating rapid growth?**
                - Rapid growth is a strong indicator that the business is **scaling successfully**. Look for metrics like **user growth**, **revenue growth**, and **expansion into new markets**.
                - Example: "The company is growing steadily, but the growth rate is not impressive yet."

            3. **Is there an indication of good customer retention?**
                - High retention rates or repeat customers indicate that the product or service is **meeting customer needs**. If retention is low, the business might face **sustainability issues**.
                - Example: "Customer retention rates are low, which could indicate dissatisfaction with the product."

            4. **What metrics or KPIs can demonstrate the business’s growth trajectory?**
                - Look for **growth metrics** such as **revenue growth**, **user acquisition rates**, or **increased market share**.
                - Example: "The business has shown strong growth in its customer base, increasing GMV by 30% month-over-month."

        """}
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
        rubric_prompt = f"""
        You are a VC analyst. Based on the summary below, return a JSON dictionary with the following exact structure (only use 0, 0.5 or 1 for scores):

        {{
          "1": {{
            "Team": {{"score": 0, "rationale": "..."}} ,
            "Business Model": {{"score": 0, "rationale": "..."}} ,
            "Traction": {{"score": 0, "rationale": "..."}}
          }},
          "2": {{
            "Team": {{"score": 0, "rationale": "..."}} ,
            "Business Model": {{"score": 0, "rationale": "..."}} ,
            "Traction": {{"score": 0, "rationale": "..."}}
          }},
          "3": {{
            "Team": {{"score": 0, "rationale": "..."}} ,
            "Business Model": {{"score": 0, "rationale": "..."}} ,
            "Traction": {{"score": 0, "rationale": "..."}}
          }},
          "4": {{
            "Team": {{"score": 0, "rationale": "..."}} ,
            "Business Model": {{"score": 0, "rationale": "..."}} ,
            "Traction": {{"score": 0, "rationale": "..."}}
          }}
        }}

        Use the criteria below to guide your evaluations:

        **Team Evaluation:**
        1. **Does the founding team appear to be complete?** 
           - Score **1** if the team consists of multiple co-founders with complementary skill sets and/or has a strong **advisory board**. This is crucial for well-rounded decision-making and execution.
           - Score **0.5** if the team has some experience but is missing key members or lacks a strong advisory board.
           - Score **0** if the team has a single founder with limited expertise or no strong advisor network.
        
        2. **Does the team have the strength and qualifications to compete in this industry or market?**
           - Score **1** if the team has deep industry experience or has successfully built similar companies before. Strong founders should have **domain expertise** and know the challenges of the market.
           - Score **0.5** if the team has some relevant experience but lacks extensive or direct industry knowledge.
           - Score **0** if the team has minimal experience or lacks knowledge in the market they are targeting.
        
        3. **Does the founding team have relevant industry experience?**
           - Score **1** if the team has **direct experience** in the relevant industry, with clear examples of successful prior ventures or expertise.
           - Score **0.5** if the experience is somewhat relevant but not directly applicable.
           - Score **0** if the team lacks relevant industry experience and it is clear that they will need significant learning to succeed in the market.
        
        4. **Have the founders previously worked together?**
           - Score **1** if the founders have worked together in the past, particularly if they have **succeeded** in building a previous startup or project.
           - Score **0.5** if they have limited or **unproven collaboration history**.
           - Score **0** if the founders have **never worked together** or have little evidence of working cohesively as a team.
        
        **Business Model Evaluation:**
        1. **Is the business model scalable?**
           - Score **1** if the business model is **highly scalable**, such as a **B2B SaaS** model, which can grow without significantly increasing costs.
           - Score **0.5** if the business model has scalability potential but faces inherent limitations, such as high capital expenditure or operational complexity.
           - Score **0** if the business model is **difficult to scale**, such as a highly **capex-intensive** hardware business.
        
        2. **Does the business have the potential to add new product lines, services, or upsell to existing customers?**
           - Score **1** if the business has **clear potential for growth** by adding new products, expanding services, or creating upsell opportunities for existing customers (e.g., **cross-selling** or **expanding market reach**).
           - Score **0.5** if there’s some opportunity for upselling or adding products, but it’s not fully explored or there are challenges in expanding the offering.
           - Score **0** if the business lacks the ability to diversify or expand its product line and seems confined to a single offering.
        
        3. **Is the business model resilient to external shocks?**
           - Score **1** if the business model is **resilient** to external risks like **economic downturns**, **regulatory changes**, or **geopolitical instability**. Businesses that are **software-only** or have strong IP protection are generally more resilient.
           - Score **0.5** if the business has some degree of vulnerability to external factors but has mitigating strategies.
           - Score **0** if the business is **highly vulnerable** to external risks, such as reliance on global supply chains or markets prone to sudden regulation changes.
        
        4. **Does the business create a new market or unlock a 'shadow market'?**
           - Score **1** if the business model creates a **new market** or significantly **disrupts an existing market**, unlocking new opportunities (e.g., **disruptive innovation** or **addressing an unmet need**).
           - Score **0.5** if the business model operates in an **existing market** but may have some level of differentiation or market opportunity.
           - Score **0** if the business is operating in an **over-saturated market** or has little to no differentiation from existing competitors.
        
        **Traction Evaluation:**
        1. **Does the business have initial customers or users?**
           - Score **1** if the business has **early customers** or users, especially paying customers, or has clear evidence of **product-market fit**.
           - Score **0.5** if the business has some customers, but it’s unclear whether they are early adopters or if the product is truly resonating with the market.
           - Score **0** if the business has **no initial customers** or users, indicating a lack of product-market fit.
        
        2. **Is the business demonstrating rapid growth?**
           - Score **1** if the business is showing **rapid growth** in key metrics such as **revenue**, **users**, or **market share**.
           - Score **0.5** if the business is growing steadily but not at an **accelerated rate**.
           - Score **0** if the business is **not growing quickly** and there’s limited indication of significant expansion.
        
        3. **Is there an indication of good customer retention?**
           - Score **1** if the business has **high customer retention rates**, repeat customers, or **strong lifetime value (LTV)**.
           - Score **0.5** if retention data is mixed or unclear but the business shows potential to retain customers with future iterations.
           - Score **0** if retention is **low** or unclear, or if the business has no evidence of **customer loyalty**.
        
        4. **What metrics or KPIs can demonstrate the business’s growth trajectory?**
           - Score **1** if the business shows clear **metrics** of success, such as **revenue growth**, **customer acquisition rates**, **GMV**, or **user engagement**.
           - Score **0.5** if there are **some metrics** available, but they are insufficient to demonstrate long-term growth.
           - Score **0** if there are **no key metrics** or KPIs available to demonstrate growth.

        IMPORTANT: Return ONLY valid JSON. No markdown, no extra text, no formatting like ```json.

        Startup summary:
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
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        th { background-color: #f8f8f8; }
        td.score-1 { background-color: #d4edda; }
        td.score-0_5 { background-color: #fff3cd; }
        td.score-0 { background-color: #f8d7da; }
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
            st.text("Is this even working?")
            render_html_table(score_data)

        record_usage(usage, log_file)
