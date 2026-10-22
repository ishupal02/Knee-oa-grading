# Put the app online (Streamlit Community Cloud, free)

You get a public link like https://your-name-knee-oa.streamlit.app to put on your CV.

## 1. Prepare the folder

Run these once, locally, from the folder that has `dataset/` and `knee_oa_model_best.pt`
(venv active):

    python evaluate.py        # writes metrics.json  (Model performance tab)
    python make_samples.py    # writes samples/      (sample buttons)
    streamlit run app.py      # check everything works locally first

The repo should contain only these (the `.gitignore` already skips venv and dataset):

    app.py
    requirements.txt
    knee_oa_model_best.pt     (about 45 MB, fine for GitHub)
    metrics.json
    samples/
    .streamlit/config.toml
    .gitignore

## 2. Push to GitHub

GitHub's website only accepts files up to 25 MB, so the model has to go through git.

1. Make a free account at github.com and create a new **public** repository,
   for example `knee-oa-grading` (do not tick "Add a README").
2. Install Git for Windows if `git --version` does not work in the terminal.
3. In the app folder, run (replace YOUR_USERNAME):

       git init
       git add .
       git commit -m "Knee OA grading app"
       git branch -M main
       git remote add origin https://github.com/YOUR_USERNAME/knee-oa-grading.git
       git push -u origin main

   Git may ask you to sign in to GitHub in a browser window the first time.

## 3. Deploy

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** (top right), choose your repository, branch `main`,
   and main file path `app.py`.
3. Click **Deploy**. The first build takes a few minutes because PyTorch is large.
   The CPU-only PyTorch line in `requirements.txt` keeps it small enough for the free tier.
4. Copy the `.streamlit.app` link into your CV and GitHub README.

## If something fails

- Open **Manage app** on the deployed page to see the build log, and send me a screenshot.
- Keep the dataset credit line in the app footer (the data is CC BY 4.0).
