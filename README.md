# Weather & Document Chatbot

Two agents + an LLM router, built with Streamlit.

- **Weather agent**: OpenWeatherMap -> tables, line / bar / pie charts.
- **Document agent**: upload PDF/TXT, answers come only from retrieved passages (TF-IDF RAG).
- **Router**: a local Ollama LLM picks the tool (`weather`, `document`, or `web` fallback with sources).

## Setup
1. Install Python 3.10+ and [Ollama](https://ollama.com), then: `ollama pull llama3.2`
2. `python -m venv venv` and activate it
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your OpenWeatherMap key
5. `streamlit run app.py`

## Sample queries
- What is the weather in Chennai today?
- What was the weather in Bengaluru yesterday?
- What will the weather be like in London tomorrow?
- List the last 10 days of weather in Chennai
- What are the skills in my resume? (after uploading)
- Who is the CEO of Google? (web fallback)

## Note
Historical weather needs the OpenWeatherMap One Call 3.0 plan. Without it, the app shows a clear limitation message and never invents data.
