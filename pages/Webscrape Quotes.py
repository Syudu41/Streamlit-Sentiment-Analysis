import streamlit as st
import pandas as pd
import json
import requests
from typing import Optional, List, Dict
from dataclasses import dataclass
from bs4 import BeautifulSoup
from openai import OpenAI

@dataclass
class Config:
    """Configuration class for API key"""
    openai_key: str

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        try:
            with open(config_path) as f:
                data = json.load(f)
                return cls(openai_key=data["OPENAI_API_KEY"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            st.error(f"Error loading configuration: {str(e)}")
            st.stop()

class QuoteScraper:
    """Handle website scraping and quote analysis"""
    def __init__(self, config: Config):
        self.openai_client = OpenAI(api_key=config.openai_key)
        self.sites = {
            'Quotes-to-Scrape': {
                'url': "https://quotes.toscrape.com/tag/{}",
                'tags': ['simile', 'love', 'humor', 
                        'books', 'reading', 'friends', 'truth']
            },
            'Lit Quotes': {
                'url': "https://www.litquotes.com/{}.php",
                'tags': ['random-words-of-wisdom', 'Random-Quote']
            }
        }

    def scrape_quotes_to_scrape(self, tag: str) -> List[str]:
        """Scrape quotes from quotes.toscrape.com"""
        try:
            response = requests.get(self.sites['Quotes-to-Scrape']['url'].format(tag))
            soup = BeautifulSoup(response.content, 'html.parser')
            quotes = soup.find_all('div', class_='quote')
            return [quote.find('span', class_='text').text for quote in quotes]
        except Exception as e:
            st.error(f"Error scraping Quotes to Scrape: {str(e)}")
            return []

    def scrape_lit_quotes(self, tag: str) -> List[str]:
        """Scrape quotes from litquotes.com"""
        try:
            response = requests.get(self.sites['Lit Quotes']['url'].format(tag))
            soup = BeautifulSoup(response.content, 'html.parser')
            quotes = soup.find_all('div', class_='purple')
            
            if tag == 'random-words-of-wisdom':
                return [quote.find('div').find('span').text for quote in quotes]
            else:  # Random-Quote
                return [quote.find('p').find('b').text for quote in quotes]
        except Exception as e:
            st.error(f"Error scraping Lit Quotes: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis bot. Classify the sentiment as Positive, Negative, or Neutral."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"

class Dashboard:
    """Main dashboard for quote scraping and analysis"""
    def __init__(self):
        self.setup_page()
        self.config = Config.load("config.json")
        self.scraper = QuoteScraper(self.config)

    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(layout='centered', page_title='Website Scraper')
        self.load_custom_css()
        st.image('Home_header.png')
        st.subheader("Welcome to my Web Scraper!")

    def load_custom_css(self):
        """Load custom CSS"""
        try:
            with open("assets/style2.css") as css:
                st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            pass

    def run(self):
        """Run the dashboard"""
        website = st.selectbox(
            'Select a sample website for Web Scraping:',
            ['None'] + list(self.scraper.sites.keys())
        )

        if website == 'None':
            return

        st.write(f'You are now scraping: {website}')
        tag = st.selectbox(
            'Select a topic to scrape:',
            ['None'] + self.scraper.sites[website]['tags']
        )

        if tag == 'None':
            return

        # Scrape and analyze quotes
        quotes = (self.scraper.scrape_quotes_to_scrape(tag) 
                 if website == 'Quotes-to-Scrape' 
                 else self.scraper.scrape_lit_quotes(tag))

        for quote in quotes:
            st.success(quote)
            sentiment = self.scraper.analyze_sentiment(quote)
            with st.chat_message("assistant"):
                st.markdown(sentiment)


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()