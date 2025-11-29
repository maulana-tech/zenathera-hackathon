# Deployment Guide for Streamlit Community Cloud

Follow these steps to deploy your CoreTax Sentiment Dashboard to the web.

## 1. Prepare Your Repository
Ensure your project is pushed to a GitHub repository.
- The repository should contain:
    - `streamlit_app/` folder
    - `requirements.txt`
    - `models/` folder (Make sure your CSV data is here)
    - `outputs/` folder (For static images)

## 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository (`scarping-hackathon` or whatever you named it).
4. **Main file path**: Enter `streamlit_app/main.py`.
5. Click **"Deploy!"**.

## 3. Troubleshooting
- **ModuleNotFoundError**: If you see this, check `requirements.txt`. We just added `streamlit` and `plotly`, so it should be good.
- **File Not Found**: Ensure your `models/BERTopic-CoreTax-data.csv` is committed to GitHub. If it's ignored by `.gitignore`, you must remove it from `.gitignore` or use Git LFS if it's huge (though <100MB is fine for standard git).
- **Memory Issues**: BERTopic and Transformers can be heavy. If the app crashes on load, we might need to enable "Lite" mode or cache the models more aggressively.

## 4. Post-Deployment
Once deployed, you will get a URL (e.g., `https://coretax-sentiment.streamlit.app`). You can share this link with anyone!
