import streamlit as st
import pandas as pd 
import altair as alt


st.set_page_config(
    layout='centered',
    page_title='Dashboard'
)


with open( "assets/style2.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.image('Home_header.png')



st.header("Overview")
st.markdown(
    '''
<b>Sentiment Scholar </b> is a <i>Sentiment Analysis Bot</i> that allows the user to detect sentiments using NLP Models or GPT. 

Additionally, a comparison is made to view the performance between GPT, Gemini LLMs against a normal NLP Model. 
The project is divided into several Modules for better optimization and functioning. It can be noted that this Bot undertakes a  role to help people understand the various forms in which Sentiment Analysis can be made.
''', unsafe_allow_html=True
)
tab1, tab2, tab3 = st.tabs(['Direct Module', 'ChatBot Module', 'Webscrapping Module'])
with tab1:
    st.subheader('Direct Module')
    st.image('DirectModule.png', use_column_width=True)

with tab2:
    st.subheader('ChatBot Module')
    st.image('ChatbotModule.png', use_column_width='auto')

with tab3:
    st.subheader('Webscrapping Module')
    st.image('Web-WebsiteModule.png',use_column_width=True)


