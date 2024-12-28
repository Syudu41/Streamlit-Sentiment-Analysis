import streamlit as st
import json
import os
from typing import Literal, Optional
from dataclasses import dataclass
from openai import OpenAI
import google.generativeai as gen_ai

# Configuration
@dataclass
class Config:
    openai_key: str
    gemini_key: str

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        try:
            with open(config_path) as f:
                data = json.load(f)
                return cls(
                    openai_key=data["OPENAI_API_KEY"],
                    gemini_key=data["GEMINI_API_KEY"]
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            st.error(f"Error loading configuration: {str(e)}")
            st.stop()

# Constants
PROMPT = """You are an expert linguist and a sentiment analysis bot which yields 3 types of sentiments - positive, negative or neutral. 
Please analyse the following statements that the user inputs and classify each one by sentiment (Positive, Negative, Neutral). 
Note that if there are any spelling errors, it may be corrected. In case of swear words, except for the first and last letters, 
censor the other letters of the word with '*' symbol. Then, output the results in a table with two columns: 'Statement' and 'Sentiment'. 
Note that at any point of time, other than text/table, no other form of output must be provided."""

ModelChoice = Literal['None', 'ChatGPT', 'Gemini']

class SentimentAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.setup_apis()

    def setup_apis(self):
        """Initialize API clients"""
        self.openai_client = OpenAI(api_key=self.config.openai_key)
        gen_ai.configure(api_key=self.config.gemini_key)
        self.gemini_model = gen_ai.GenerativeModel(
            'models/gemini-1.5-flash',
            system_instruction=PROMPT
        )

    def analyze_with_chatgpt(self, message: str) -> str:
        """Get sentiment analysis from ChatGPT"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error with ChatGPT API: {str(e)}")
            return None

    def analyze_with_gemini(self, message: str) -> Optional[str]:
        """Get sentiment analysis from Gemini"""
        try:
            response = self.gemini_model.generate_content(message)
            return response.text
        except Exception as e:
            st.error(f"Error with Gemini API: {str(e)}")
            return None

class Dashboard:
    def __init__(self):
        self.setup_page()
        self.load_config()
        self.initialize_state()

    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(layout='centered', page_title='Sentiment Analysis Dashboard')
        self.load_custom_css()
        st.image('Home_header.png')
        st.subheader("Welcome to the Chatbot Module!")

    def load_custom_css(self):
        """Load custom CSS if available"""
        try:
            with open("assets/style2.css") as css:
                st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            pass

    def load_config(self):
        """Load configuration and initialize analyzer"""
        working_dir = os.path.dirname(os.path.abspath(__file__))
        config = Config.load(f"{working_dir}/config.json")
        self.analyzer = SentimentAnalyzer(config)

    def initialize_state(self):
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def run(self):
        """Run the dashboard"""
        choice = st.selectbox(
             'Choose Generative Model:',
            ['None', 'ChatGPT']
        )

        if choice == 'None':
            return

        st.subheader(f'You are now using {choice}:')
        
        # Display chat history
        self.display_chat_history(choice)
        
        # Handle user input
        user_input = st.chat_input('Enter any statement to analyze the sentiment...')
        if user_input:
            self.process_user_input(user_input, choice)

    def display_chat_history(self, model: ModelChoice):
        """Display chat history based on selected model"""
        if model == 'ChatGPT':
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])

    def process_user_input(self, user_input: str, model: ModelChoice):
        """Process user input and get model response"""
        st.chat_message("user").markdown(user_input)

        if model == 'ChatGPT':
            self.handle_chatgpt_response(user_input)


    def handle_chatgpt_response(self, user_input: str):
        """Handle ChatGPT response"""
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = self.analyzer.analyze_with_chatgpt(user_input)
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()