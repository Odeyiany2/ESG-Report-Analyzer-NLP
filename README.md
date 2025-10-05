# **🌍ESG Report Analyzer & Predictor**
## **📌 Overview**

This project combines AI + Accounting to help companies assess and improve their Environmental, Social, and Governance (ESG) disclosures.

This MVP has tow main layers:

 - Chatbot ESG Report Analyzer (RAG) → Companies upload their sustainability reports, and a chatbot analyzes coverage against standards (GRI, SASB, IFRS) to detect vagueness, compliance gaps, and missing sections.

 - ESG Score Predictor & Forecasting → Machine learning models predict ESG category scores (Environmental, Social, Governance) and forecast future ESG trends based on company features.

## **🚀 Features**
 - Layer 1: ESG Chatbot Analyzer

    📂 Upload annual or sustainability reports (PDF).
    
    🤖 Chatbot Q&A with the report → "Does this report cover board independence?"
    
    ✅ Compliance breakdown with ESG standards (GRI, SASB, IFRS).
    
    📊 Coverage summary (percentage compliance, vague vs. clear disclosures).

- Layer 2: Predictive Models

    🔮 Regression Model → Predict ESG category scores (Environmental, Social, Governance).
    
    📈 Time Series Forecasting → Predict future ESG trends for a company.
    
    💬 Interactive assistant asks for company features and outputs scores + forecasts.
    
    📊 Visualization of ESG trends and predicted future compliance.
  ![](https://github.com/Odeyiany2/ESG-Report-Analyzer-NLP/blob/main/architecture/ESG_workflow.png)

## **🛠️ Tech Stack**

 - Backend: FastAPI (API for chatbot + models)

 - Frontend: Streamlit (Dashboard + Chat UI)

 - NLP / RAG: LangChain, HuggingFace Transformers, ChromaDB

 - ML Models:

     - Regression (scikit-learn)
    
     - Time Series Forecast (Prophet / statsmodels)
    
     - ESG Classification (TF-IDF + Logistic Regression, Transformers like FinBERT)

 - Data:

     - SEC Filings (10-K ESG sections)
    
     - Nigerian Stock Exchange Filings
    
     - UN Global Compact Reports
    
     - Kaggle ESG datasets (structured labels)
    
     - Storage: ChromaDB (vector store)

## **📅 Roadmap**

- [x] Setup repo + data collection

- [ ] Build ESG chatbot (RAG) MVP

- [ ] Add ESG text classification model

- [ ] Build regression ESG scoring model

- [ ] Add time-series forecasting for ESG trends

- [ ] Integrate all in Streamlit dashboard

- [ ] Deploy full solution (Streamlit Cloud / Render)
