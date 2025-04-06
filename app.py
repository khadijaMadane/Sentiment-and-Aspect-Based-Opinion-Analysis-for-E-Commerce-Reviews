import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import re
import base64

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("C:/Users/khadija/Documents/sentiment_proj/Sentiment-and-Aspect-Based-Opinion-Analysis-for-E-Commerce-Reviews/backg.png")

# Load trained model and tokenizer
model_path = "sentiment_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to classify sentiment
def classify_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if predictions[0][0] > 0.5 else "Negative"
    return sentiment

# Function to extract aspects
def extract_aspects(text):
    doc = nlp(text)
    aspects = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return aspects if aspects else ["General"]

# Function to analyze text
def analyze_review(review):
    sentences = re.split(r'[,.]', review)
    analysis_results = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentiment = classify_sentiment(sentence)
            aspects = extract_aspects(sentence)
            analysis_results.append((sentence, sentiment, aspects))

    return analysis_results

st.markdown(
    """
    <style>
        /* Keep the title in green */
        h1 {
            color: #228B22 !important;
        }

        /* Make subtitles and labels black */
        h2, h3, h4, p, label {
            color: #000000 !important;
        }

        /* Make buttons green with black text */
        div.stButton > button {
            background-color: #228B22 !important;
            color: #000000 !important;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }

        /* Force alignment to left */
        .block-container {
            width: 50vw !important;
            margin: 0 auto 0 10px !important;
            padding-left: 10px;
            background-color: rgba(255, 255, 255, 0.0008);
            border-radius: 10px;
            padding: 20px;
        }

        /* Adjust text area */
        .stTextArea textarea {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "text_area" not in st.session_state:
    st.session_state.text_area = ""

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []  
def clear_text():
    st.session_state.text_area = ""  
    st.session_state.analysis_results = []  

st.title("Sentiment Analysis Application")
st.write("Analyze sentiment and extract aspects from product reviews.")

review = st.text_area("Enter a product review:", value=st.session_state.text_area, key="text_area")

if st.button("Analyze Review", key="analyze_button"):
    if review:
        st.session_state.analysis_results = analyze_review(review)  

for idx, (sentence, sentiment, aspects) in enumerate(st.session_state.analysis_results):
    st.subheader(f"Sentence {idx+1}:")
    st.write(f"**Text:** {sentence}")
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Extracted Aspects:** {', '.join(aspects)}")

st.button("Reset", key="reset_button", on_click=clear_text)  
