import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
import json, re
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import google.generativeai as gen_ai

def clean_text(text):
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def get_sentiment_rating(data):
    for index, _ in data.iterrows():
        token = tokenizer.encode(data.loc[index]['Post Content'], padding=True, truncation=True, max_length=50, add_special_tokens=True, return_tensors='pt')
        results = model(token)
        rating = int(torch.argmax(results.logits))+1
        
        if rating <= 2:
            data.at[index, 'BERT_pred_label'] = 'Negative'
        elif rating == 3:
            data.at[index, 'BERT_pred_label'] = 'Neutral'
        else:
            data.at[index, 'BERT_pred_label'] = 'Positive'
    return data

def get_GPT_rating(data):
    GPT_prompt = f'''
    You are an expert in social media sentiment analysis. Classify the posts into: Positive, Neutral, or Negative.
    In case the post is blank, classify it as Neutral.
    Only return the json code as output - update predicted labels under 'GPT_pred_label'.
    Don't modify the json format.
    Note that:
    If post asks questions -> likely Neutral
    If post shows enthusiasm/excitement -> Positive 
    If post shows confusion/frustration -> Negative

    ```
    {data}
    ```
    '''
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": GPT_prompt}
        ],
    )
    return response.choices[0].message.content.strip()

def get_Gemini_rating(data):
    Gemini_prompt = f'''
    You are an expert linguist and a sentiment analysis bot which yields 3 types of sentiments - positive, neutral or negative. 
    Help me classify the reviews into: Positive, Neutral and Negative. 
    In case that the statement is blank, then return it as a negative sentiment.
    In your output, only return the json code  back as output - which is provided between three backticks. 
    Your task is to update predicted labels under 'Gemini_pred_label' in the json code.  
    Don't make any changes to the json code format, please.
    do not add the word 'json' in your response. just return the updated data.
    Note that:
    If post asks questions -> likely Neutral
    If post shows enthusiasm/excitement -> Positive 
    If post shows confusion/frustration -> Negative


    ```
    {data}
    ```

    '''
    model = gen_ai.GenerativeModel('models/gemini-1.5-flash')
    response_Gemini = model.generate_content(Gemini_prompt)
    c_r = response_Gemini.candidates[0].content.parts[0].text.strip()
    
    return c_r

st.set_page_config(layout='centered', page_title='Social Media Behavior Dataset')

# Configure API keys
working_dir = 'D:/Projects/GPT-streamlit website'
config_data = json.load(open(f"{working_dir}/config.json"))
openai.api_key = config_data["OPENAI_API_KEY"]
gen_ai.configure(api_key=config_data["GEMINI_API_KEY"])

# Load BERT model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load CSS
with open("assets/style2.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.image('Home_header.png')
st.title("Tech Social Media Sentiment Analysis")

# Load and process data
df_data = pd.read_csv('social_media_behavior_dataset.csv')
st.write('Raw Data Sample:')
AgGrid(df_data.head(), fit_columns_on_grid_load=True)

# Sample size selection
# sample_size = st.slider('Select sample size for analysis:', 5, 30, 5)
analysis_sample = df_data.sample(5)

# Add prediction columns
analysis_sample['BERT_pred_label'] = ''
analysis_sample['GPT_pred_label'] = ''
analysis_sample['Gemini_pred_label'] = ''

# Run predictions
analysis_sample = get_sentiment_rating(analysis_sample)
st.write('BERT Predictions:')
AgGrid(analysis_sample)

# GPT predictions
json_data_GPT = analysis_sample[['Sentiment', 'Post Content', 'BERT_pred_label', 'GPT_pred_label', 'Gemini_pred_label']].to_json(orient='records')
response_GPT = get_GPT_rating(json_data_GPT)
gpt_data = json.loads(response_GPT.strip("`"))
df_gpt = pd.DataFrame(gpt_data)
st.write('GPT Predictions:')
AgGrid(df_gpt)


# Gemini predictions
json_data_Gemini = df_gpt[['Sentiment', 'Post Content', 'BERT_pred_label', 'GPT_pred_label', 'Gemini_pred_label']].to_json(orient='records')
response_gemini = get_Gemini_rating(json_data_Gemini)
gemini_data = json.loads(response_gemini.strip("`"))
df_final = pd.DataFrame(gemini_data)
st.write('Final Predictions (Including Gemini):')
AgGrid(df_final)

# Calculate agreement metrics for given sample
def calculate_agreement(df):
    total = len(df)
    bert_agreement = (df['BERT_pred_label'] == df['Sentiment']).mean() * 100
    gpt_agreement = (df['GPT_pred_label'] == df['Sentiment']).mean() * 100
    gemini_agreement = (df['Gemini_pred_label'] == df['Sentiment']).mean() * 100
    
    st.write(f"Model Agreement with Original Sentiment:")
    st.write(f"BERT: {bert_agreement:.2f}%")
    st.write(f"GPT: {gpt_agreement:.2f}%")
    st.write(f"Gemini: {gemini_agreement:.2f}%")

# calculate_agreement(df_final)