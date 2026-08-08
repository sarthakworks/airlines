# Airline AI — Support Bot + Flight Intelligence

Two apps built on real Indian aviation data:

- **`flight-intel/`** — **India Flight Intelligence** (Streamlit): predicts fares, ranks
  airlines (price × on-time), best time/booking window, route/sector issues, and searches
  real flight schedules — with a natural-language chat. *This is the deployed app.*
- **`project/`** — the Mini-Project 4 **support bot** (FastAPI + Streamlit): classifier →
  SQL/RAG → guardrails. Runs locally with Docker Postgres + Chroma. See `project/README.md`.

## Deploy the dashboard (free, Streamlit Community Cloud)
1. Push this repo to GitHub (done).
2. Go to https://share.streamlit.io → sign in with GitHub → **Create app → from GitHub**.
   - **Repository:** this repo · **Branch:** `main`
   - **Main file path:** `flight-intel/dashboard.py`
   - **Advanced → Python:** `3.11`
3. **Advanced → Secrets** (TOML):
   ```toml
   GROQ_API_KEY = "your_groq_key"
   ```
   (Only the chat tab needs it; the other tabs work without.)
4. Deploy → you get a public `*.streamlit.app` URL.

Dependencies for the deploy are in the **root `requirements.txt`**.

## Run the dashboard locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
streamlit run flight-intel/dashboard.py
```

## Notes
- This repo is **private** — it includes course PDFs (© TalentSprint); remove them before
  making it public.
- Secrets (`.env`) are gitignored and never committed; keys live only in deploy secrets.
- Data: DGCA on-time/complaints/traffic, a 300k-row fare dataset (2022), and ~340k
  historical schedule records (2018–2024). Fares/times are indicative, not live.
