# Financial Fraud & Risk Intelligence

Educational capstone combining supervised machine learning, policy retrieval (RAG), Gemini, and LangGraph in a Streamlit app.

## Components
- Colab notebook: exploratory data analysis, training experiments, RAG and agentic workflow exploration
- Random Forest fraud-risk scoring with a saved model and a 0.90 investigation threshold
- Fraud investigation policy retrieval using Sentence Transformers and ChromaDB
- Gemini-generated investigation reports, validation, and human-review flagging
- Streamlit interface and a policy Q&A tab

## Run the app
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run src/app/app.py
```
Run commands from this directory. The application loads the included Random Forest model. On first launch, it builds a small local ChromaDB policy index from `data/knowledge_base/fraud_investigation_policy.txt`; this requires downloading the `all-MiniLM-L6-v2` embedding model.

## Dataset and notebooks
The original ~470 MB PaySim CSV is intentionally excluded. To rerun training, obtain the dataset separately and place it at `data/raw/PS_20174392719_1491204439457_log.csv`. The notebook is a record of Colab experiments; some cells use Google Drive paths and may need adapting outside Colab. Notebook execution outputs were cleared for privacy and repository size.

## Important notes
- This repository includes a saved Random Forest model and neural-network weights, but the Streamlit application currently uses **Random Forest** for scoring.
- Generated ChromaDB files, credentials and the raw training data are excluded.
- The ML model is an educational prototype. Scores are decision-support signals, not confirmed fraud; consequential decisions require authorized human review.
- A GitHub upload is source-code publication, not live hosting.


## Streamlit Community Cloud deployment

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Select **Create app** / **Deploy a public app from GitHub**.
3. Repository: `SankethHegde712/LogicMojo-AI-ML-March26-Sanketh`
4. Branch: `main`
5. Main file path: `financial-fraud-risk-intelligence/src/app/app.py`
6. Choose Python **3.12** in advanced settings if available.
7. Add this secret in Streamlit's advanced settings or app Secrets page, using your own private key:
   ```toml
   GEMINI_API_KEY = "your_private_gemini_api_key"
   ```
8. Deploy and test a synthetic high-risk transfer and a low-risk payment.

The deployment-specific dependencies are in `src/app/requirements.txt`. The app creates its small policy vector index from the tracked synthetic policy text on first startup. The PaySim CSV is not required for predictions because the trained Random Forest model is included.

**Submission caveat:** This is a demonstration app, not production banking software. The human-review node flags cases for review; an auditable reviewer decision workflow and the complete capstone automated-test suite are not implemented or verified.
