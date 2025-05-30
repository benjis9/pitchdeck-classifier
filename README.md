# Pitch Deck Classifier for Venture Capitalists
## Overview
The Pitch Deck Classifier is a tool designed to assist Venture Capital (VC) investors in evaluating and scoring startup pitch decks efficiently. By leveraging OpenAI’s GPT-4 model and custom prompt engineering, the tool processes pitch decks, extracting key information such as VC stage, region/country, industry, and grading based on a custom rubric. This application aims to save time and reduce manual work for VC investors by automatically analyzing pitch decks and presenting clear evaluations and rationales.

## Problem Statement
Venture capital firms typically receive thousands of pitch decks every year, often with dense and complex information spread across 15-30 pages. Investors are required to manually sift through these decks to find relevant details about the startup's market, business model, and stage, which can be inefficient and time-consuming. Moreover, crucial information, such as the company's incorporation country or stage of development, might be hidden in later pages.

## The Pitch Deck Classifier solves this problem by:

- Automatically analyzing and summarizing the content of the pitch deck, including images, graphs, and text.
- Extracting key details such as VC stage, region, and industry from the deck.
- Grading the pitch deck based on a set of predefined criteria, allowing investors to make faster, more informed decisions.

## Features
- **Pitch Deck Upload**: VC investors can upload a PDF pitch deck through the Streamlit interface.
- **Pitch Deck Analysis**: The tool processes the pitch deck, extracting the VC stage, region/country, and industry.
- **Grading and Evaluation**: The pitch deck is graded using a custom rubric that evaluates the startup’s team, business model, and traction.

## Scoring System: 
Each pitch deck is evaluated on:

- **Team**: Assessing the strength of the founding team.
- **Business Model**: Evaluating scalability, resilience, and innovation.
- **Traction**: Reviewing customer base, growth, and retention.

## Tech Stack
- **Frontend**: Streamlit (for the user interface)
- **Backend**: Python
- **AI Model**: OpenAI GPT-4 API for analyzing text and scoring the pitch deck.
- **PDF Processing**: PyMuPDF for extracting text and images from PDF files.
- **Image Processing**: PIL (Python Imaging Library) for handling images extracted from pitch decks.

## How It Works
**Upload a Pitch Deck**: Users upload a pitch deck in PDF format via the Streamlit interface.

**Pitch Deck Processing**:

The deck is processed page-by-page using PyMuPDF to extract text and images.
OpenAI GPT-4 is then used to analyze the extracted text and images, summarizing key details and assigning scores based on the criteria in the custom rubric.

**Scoring and Grading**: Each pitch deck is scored on three main categories:
- **Team**: Strength and experience of the founding team.
- **Business Model**: Scalability, resilience, and innovation of the business model.
- **Traction**: Early customer base, growth metrics, and retention.

Display Results: The VC stage, region, industry, and final score are displayed to the user along with the grading rationale.

## Grading Rubric
The pitch decks are evaluated based on the following criteria:

### Team Evaluation:
1. Does the founding team appear to be complete?
2. Have one or more of the founders built a business before?
3. Does the founding team have relevant industry experience?
4. Have the founders previously worked together?

### Business Model Evaluation:
1. Is the business model easily scalable?
2. Does the business have the potential to add new product lines, services, or upsell to existing customers?
3. Is the business model resilient to external shocks?
4. Does the business create a new market or unlock a 'shadow market'?

### Traction Evaluation:
1. Does the business have initial customers or users?
2. Is the business demonstrating rapid growth?
3. Is there an indication of good customer retention?
4. What metrics or KPIs can demonstrate the business’s growth trajectory?

Each category is scored on a 0-1 scale:

- 1 indicates a positive evaluation.
- 0.5 indicates a mixed evaluation.
- 0 indicates a negative evaluation.

## Example Workflow
The VC investor uploads a pitch deck.

The tool processes the deck, extracting key information (VC stage, region, industry) and summarizing the text and images.

The pitch deck is scored based on the rubric.

The final results are presented, including:

**VC Stage**: (e.g., Pre-Seed, Seed, Series A)
**Region**: (e.g., Europe, Israel)
**Industry**: (e.g., Climate-Tech, SaaS)
**Final Score**: (Total points out of maximum score)

## Usage Instructions
Launch the Streamlit app:

bash
Copy
streamlit run app.py
Log in: Use the password defined in the .streamlit/secrets.toml file.

Upload a Pitch Deck: Once logged in, upload a PDF pitch deck using the interface.

View Results: After processing, the results will display the VC stage, region, industry, and the final score for the pitch deck.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Contrarian Ventures for providing pitch decks for testing and validation.
OpenAI for the GPT-4 API to analyze and score pitch decks.
PyMuPDF and PIL for processing PDF files and images.
