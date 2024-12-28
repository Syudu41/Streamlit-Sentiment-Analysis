# social_media_scraper.py
import streamlit as st
import pandas as pd
import json
import os
import re
from typing import Optional, Dict, List
from dataclasses import dataclass
from googleapiclient.discovery import build
from openai import OpenAI
from st_aggrid import AgGrid
import random, requests
@dataclass
class Config:
    """Configuration class for API keys"""
    openai_key: str
    google_yt_key: str

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        try:
            with open(config_path) as f:
                data = json.load(f)
                return cls(
                    openai_key=data["OPENAI_API_KEY"],
                    google_yt_key=data["GOOGLE_YT_API_KEY"]
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            st.error(f"Error loading configuration: {str(e)}")
            st.stop()

class YouTubeAnalyzer:
    """Handle YouTube API interactions and comment analysis"""
    def __init__(self, config: Config):
        self.youtube = build('youtube', 'v3', developerKey=config.google_yt_key)
        self.openai_client = OpenAI(api_key=config.openai_key)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\b[a-zA-Z]\b", " ", text)
        text = re.sub(r"<[^>]*>", " ", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|$)'
        match = re.search(pattern, url)
        return match.group(1) if match else None

    def get_total_comments(self, video_id: str) -> int:
        """Get total number of comments for a video"""
        try:
            request = self.youtube.videos().list(
                part="statistics",
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                return int(response['items'][0]['statistics']['commentCount'])
            return 0
            
        except Exception as e:
            st.error(f"Error fetching comment count: {str(e)}")
            return 0

    def get_video_comments(self, video_id: str, max_comments: int) -> pd.DataFrame:
        """Fetch comments for a YouTube video"""
        try:
            comments_data = []
            next_page_token = None
            
            while len(comments_data) < max_comments:
                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=min(100, max_comments - len(comments_data)),
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments_data.append({
                        'comment': comment['textDisplay'],
                        'user_name': comment['authorDisplayName'],
                        'date': comment['publishedAt'],
                        'replies': [reply['snippet']['textDisplay'] 
                                  for reply in item.get('replies', {}).get('comments', [])]
                    })
                    
                    if len(comments_data) >= max_comments:
                        break
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return pd.DataFrame(comments_data[:max_comments])
        
        except Exception as e:
            st.error(f"Error fetching comments: {str(e)}")
            return pd.DataFrame()

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": """You are an expert linguist and a sentiment analysis bot. 
                    Classify the sentiment as Positive, Negative, or Neutral. 
                    Format your response as: 'Sentiment: [classification]'"""},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"

class Dashboard:
    """Main dashboard for YouTube comment analysis"""
    def __init__(self):
        self.setup_page()
        self.config = Config.load("config.json")
        self.analyzer = YouTubeAnalyzer(self.config)

    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(layout='centered', page_title='Social Media Analysis')
        self.load_custom_css()
        st.image('Home_header.png')
        st.subheader("YouTube Comment Analyzer")

    def load_custom_css(self):
        """Load custom CSS"""
        try:
            with open("assets/style2.css") as css:
                st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            pass

    def run(self):
        """Run the dashboard"""
        yt_link = st.text_input('Enter a YouTube Link')
        
        if not yt_link:
            return

        video_id = self.analyzer.extract_video_id(yt_link)
        if not video_id:
            st.error('Invalid YouTube link')
            return

        # Get total comments available
        total_comments = self.analyzer.get_total_comments(video_id)
        
        if total_comments == 0:
            st.warning("No comments available for this video")
            return

        # Add number of comments selector with max comments info
        num_comments = st.number_input(
            f'Number of comments to analyze (Max available: {total_comments:,})',
            min_value=1,
            max_value=min(100, total_comments),  # Limit to either 100 or total comments, whichever is smaller
            value=min(5, total_comments),  # Default to 5 or total comments, whichever is smaller
            step=1
        )

        # Add analyze button
        if st.button('Analyze Comments'):
            with st.spinner('Fetching and analyzing comments...'):
                # Clear previous results if they exist
                if 'comments_df' in st.session_state:
                    del st.session_state.comments_df
                if 'sentiments' in st.session_state:
                    del st.session_state.sentiments

                # Fetch new comments
                comments_df = self.analyzer.get_video_comments(video_id, num_comments)
                st.session_state.comments_df = comments_df
                
                if comments_df.empty:
                    st.warning('No comments found')
                    return

                # Display comments grid
                st.write(f'Showing {len(comments_df)} comments:')
                AgGrid(comments_df, fit_columns_on_grid_load=True)

                # Analyze sentiments
                st.write("Sentiment Analysis:")
                st.session_state.sentiments = []
                
                for idx, comment in enumerate(comments_df['comment'], 1):
                    with st.container():
                        cleaned_text = self.analyzer.clean_text(comment)
                        sentiment = self.analyzer.analyze_sentiment(cleaned_text)
                        
                        # Store sentiment result
                        st.session_state.sentiments.append({
                            'comment': comment,
                            'sentiment': sentiment
                        })
                        
                        # Display result
                        st.write(f"Comment {idx}:")
                        st.write(comment)
                        st.write(sentiment)
                        st.divider()

                # Display summary
                try:
                    sentiments = []
                    for s in st.session_state.sentiments:
                        if 'Positive' in s['sentiment']:
                            sentiments.append('Positive')
                        elif 'Negative' in s['sentiment']:
                            sentiments.append('Negative')
                        elif 'Neutral' in s['sentiment']:
                            sentiments.append('Neutral')
                        else:
                            sentiments.append('Neutral')  # default case
                    
                    positive = sentiments.count('Positive')
                    negative = sentiments.count('Negative')
                    neutral = sentiments.count('Neutral')
                    
                    st.write("### Summary")
                    st.write(f"- Positive comments: {positive}")
                    st.write(f"- Negative comments: {negative}")
                    st.write(f"- Neutral comments: {neutral}")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
    # Load config and API key
with open("config.json") as f:
    config = json.load(f)
YOUTUBE_API_KEY = config["GOOGLE_YT_API_KEY"]

def check_video_existence(video_id):
    """Check if a YouTube video exists using the YouTube Data API"""
    url = f"https://www.googleapis.com/youtube/v3/videos"
    params = {
        'id': video_id,
        'key': YOUTUBE_API_KEY,
        'part': 'id'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return len(data.get('items', [])) > 0
    return False

def get_random_video_id():
    """Get a random YouTube video ID from predefined list"""
    # Popular video IDs that are likely to exist long-term
    video_ids = [
        "dQw4w9WgXcQ",  # Never Gonna Give You Up
        "jNQXAC9IVRw",  # Me at the zoo (First YouTube video)
        "9bZkp7q19f0",  # Gangnam Style
        "kJQP7kiw5Fk",  # Despacito
        "JGwWNGJdvx8",  # Shape of You
        "OPf0YbXqDm0",  # Uptown Funk
        "dzsuE5ugxf4",  # Hello (Adele)
        "pRpeEdMmmQ0",  # Shake It Off
        "60ItHLz5WEA",  # Faded
        "RgKAFK5djSk"   # See You Again
    ]
    return random.choice(video_ids)

# Sidebar alignment
st.markdown("""
        <style>
        .video-code {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-family: monospace;
            font-size: 1.2rem;
        }
        .url-display {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            word-break: break-all;
        }
        </style>
        """, unsafe_allow_html=True)
    
if st.sidebar.button('Get Random Video ID'):
    video_id = get_random_video_id()
    
    # Store in session state
    if 'current_video' not in st.session_state:
        st.session_state.current_video = video_id
    else:
        st.session_state.current_video = video_id

    # st.markdown(f'<div class="video-code">{video_id}</div>', unsafe_allow_html=True)
    # Display results
    st.sidebar.markdown(f"""
    #### Video URL:
    ```
    https://youtube.com/watch?v={video_id}
    ```
    """)