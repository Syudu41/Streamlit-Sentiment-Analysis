# Sentiment Scholar

A comprehensive sentiment analysis platform featuring multiple analysis modules powered by modern language models and web scraping capabilities.

## 🌟 Features

### Three Powerful Modules

1. **Direct Analysis Module**
   - Compares sentiment analysis results across multiple models:
   - BERT NLP (pre-trained) - from Hugging Face `nlptown`
   - ChatGPT
   - Google Gemini
   - Accepts CSV files as input for batch processing

2. **Chatbot Module**
   - Interactive sentiment analysis through a ChatGPT-powered conversational interface
   - Real-time analysis and feedback

3. **Web Scraping Module**
   - Extract and analyze sentiments from:
   - YouTube comments
   - Website content
   - Automated data collection and processing

## ⚙️ Prerequisites

Before running the application, ensure you have:

- All required Python packages installed
- Valid API keys for:
  - OpenAI (ChatGPT)
  - Google Gemini
- All project files downloaded to your local system

## 🚀 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/streamlit-sentiment-analysis.git
cd streamlit-sentiment-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
   - Open `config.json`
   - Add your API keys:
   ```json
   {
     "openai_key": "your-chatgpt-api-key",
     "gemini_key": "your-gemini-api-key"
   }
   ```

## 🎯 Usage

1. Launch the Streamlit app:
```bash
streamlit run Home.py
```

2. Navigate through the different modules using the sidebar menu

## ⚠️ Known Issues & Troubleshooting

- If you encounter a JSON error when using ChatGPT and Gemini simultaneously, try refreshing the page
- Ensure all API keys are correctly configured in `config.json` before running any analysis
- Verify all required files are present in your local system before launching the application

## 📝 License

MIT License
