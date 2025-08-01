# 🤖 AI Powered Prompt-based Data Analyzer

**Data Analyzer** is an interactive Streamlit-based tool that uses **Google Gemini (Gemini 1.5)** to analyze your data using natural language — just upload a CSV, ask questions, and get instant visualizations or answers.

---

## 💡 Why Use This?

This project is designed to **bridge the gap between technical and non-technical users**:

✅ No coding required — users can interact with data using simple questions.  
✅ AI-generated visualizations and filters — powered by Gemini and LangChain.  
✅ Supports multiple chart libraries — `matplotlib`, `seaborn`, and `plotly`.  
✅ Auto encoding detection — even messy CSV files are handled gracefully.  
✅ PDF export — download a full summary of your analysis.

Whether you're a **data analyst, student, educator, or business user**, this app makes data interaction more human and accessible.

---

## 🛠️ Setup Instructions

1. **Replace Gemini API Key**  
   Create a `.env` file in the project folder and add your Gemini API key like this:  
   `GOOGLE_API_KEY=your_gemini_api_key_here`

2. **Open Project Folder in Command Prompt**  
   Navigate to the project directory using `cd`.

3. **Create & Activate Virtual Environment**  
   Run the following commands:  
   `python -m venv venv`  
   `venv\Scripts\activate` (for Windows)  
   or  
   `source venv/bin/activate` (for macOS/Linux)

4. **Install Required Packages**  
   `pip install -r requirements.txt`

5. **Run the App**  
   `streamlit run app.py`

---
