# Deploying to Streamlit Community Cloud

This app is deploy-ready as-is: it's a standard Streamlit app, all game
data ships as local CSVs (no scraping or network calls at runtime), no
secrets or API keys are needed, and the full test suite (235 tests) is
green.

## 1. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account
   (first time: it'll ask to authorize Streamlit to access your repos)
2. Click "New app"
3. Pick this repo (`mam084/three-houses-optimizer`), branch `main`, and
   set the main file path to `app.py`
4. Click "Deploy" — the first build takes a couple of minutes while it
   installs `requirements.txt` (streamlit, plotly, pandas, numpy, Pillow,
   requests, beautifulsoup4, lxml)
5. You'll get a URL like `https://<something>.streamlit.app` — that's the
   link to share with friends. It isn't listed anywhere or indexed by
   search engines; only people you send the link to will find it.

## Updating it later

Any further changes are just a normal push to `main`:
```bash
git add -A
git commit -m "..."
git push
```
Streamlit Cloud watches the repo and redeploys automatically on every
push — no redeploy step needed on your end.

## If something goes wrong on first deploy

- **Build fails installing a package**: Streamlit Cloud's default Python
  version occasionally drifts from what a package needs. In the app's
  "Manage app" → "Settings" → "General" you can pin a specific Python
  version (3.10–3.12 all pass this project's CI, see
  `.github/workflows/tests.yml`).
- **App loads but looks broken / errors on a specific tab**: check the
  "Manage app" logs panel in the bottom-right of the deployed app — it
  shows the real Python traceback, not just a generic error page.
- **Portraits don't show**: expected for most characters — this repo
  doesn't bundle official Three Houses art (see
  `assets/portraits/README.md`); the color-coded house tiles are the
  intended fallback, not a bug.

## Free-tier limits worth knowing

Streamlit Community Cloud's free tier: 1 GB RAM per app, and the app
"sleeps" after a period with no visitors (it wakes back up in ~30s–1min
the next time someone opens the link — no action needed from you). Both
are fine for casual friend-testing traffic; if it ever needs more, the
straightforward next step is Render/Railway with a `Dockerfile`.
