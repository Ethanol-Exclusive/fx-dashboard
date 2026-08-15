# Exclusive FX — Daily Dashboard (GitHub Pages)

This turns the dashboard into a real website with a permanent link. A
scheduled GitHub Action re-runs the strategy every morning and publishes
the result automatically — you just open the link on your phone, no
downloading, emailing, or AirDropping files.

## One-time setup (10–15 minutes)

### 1. Create a free GitHub account
Go to github.com → sign up, if you don't already have one.

### 2. Create a new repository
- Click the **+** in the top right → **New repository**
- Name it something like `fx-dashboard`
- Set it to **Public** (required for free GitHub Pages)
- Don't add a README/gitignore (we already have the files)
- Click **Create repository**

### 3. Upload these files to the repository

**Easiest method:** you were given a `fx-dashboard-repo.zip` — unzip it on your
PC first. You'll end up with a folder containing `index.html`,
`strategy_v2.py`, `daily_dashboard.py`, `README.md`, and a `.github` folder.

On the new repo's page, click **"uploading an existing file"**, then drag
in `index.html`, `strategy_v2.py`, `daily_dashboard.py`, and `README.md`
together — commit those first.

**Then the workflow file separately** (GitHub's drag-and-drop web upload
often flattens nested folders, so do this one on its own):
- Click **Add file → Create new file**
- In the filename box, type the full path: `.github/workflows/update-dashboard.yml`
  (typing the slashes creates the folders automatically)
- Paste in the contents of that file from the zip
- Commit

If you'd rather avoid any of this fiddliness, using `git` from a terminal is more reliable:
```bash
cd fx-dashboard-repo
git init
git add .
git commit -m "Initial dashboard setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fx-dashboard.git
git push -u origin main
```

### 4. Enable GitHub Pages
- In your repo, go to **Settings** → **Pages** (left sidebar)
- Under "Build and deployment" → Source: select **Deploy from a branch**
- Branch: select **main**, folder: **/ (root)** → Save
- GitHub will give you a URL like:
  `https://yourusername.github.io/fx-dashboard/`
- This can take a minute or two to go live the first time

### 5. Enable and test the Action
- Go to the **Actions** tab in your repo
- You should see "Update Trading Dashboard" listed
- Click it → **Run workflow** → **Run workflow** (this triggers it manually so you don't have to wait for the schedule)
- Wait ~1 minute, refresh — it should show a green checkmark
- Visit your Pages URL — it should now show live data instead of the sample

### 6. Bookmark the link on your phone
Open `https://yourusername.github.io/fx-dashboard/` in your phone's browser,
then **Add to Home Screen** (Safari: Share → Add to Home Screen. Chrome: ⋮ menu → Add to Home Screen).
Now it behaves like an app icon that opens straight to your dashboard.

## What happens automatically after this
Every day at 11:00 UTC (~7 AM New York), GitHub runs the script fresh,
regenerates `index.html` with current market data, and pushes it live.
No PC needs to be on — GitHub's own servers run it.

## Changing the schedule
Open `.github/workflows/update-dashboard.yml`, edit this line:
```yaml
- cron: "0 11 * * *"
```
Cron format is `minute hour day month weekday`, always in **UTC**. Examples:
- `"0 12 * * *"` → 12:00 UTC daily
- `"30 11 * * 1-5"` → 11:30 UTC, weekdays only

You can also just click **Run workflow** manually anytime from the Actions tab if you want a fresh pull right before you check it.

## Troubleshooting
- **Page shows old/sample data:** check the Actions tab — if the run failed, click into it to see the error (most likely a package install issue, rare).
- **Workflow doesn't appear in Actions tab:** double check `.github/workflows/update-dashboard.yml` uploaded to the exact right path — GitHub only detects workflows in that specific folder.
- **"Nothing to commit" every run:** that's fine — it just means the data didn't change since last run (e.g. market closed).
