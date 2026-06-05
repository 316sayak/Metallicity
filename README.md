# Metallicity

This repository contains a static HTML viewer (index.html) with per-panel comment functionality saved to browser `localStorage`.

Hosting on GitHub Pages

Option A — using GitHub CLI (recommended):

1. Install and authenticate `gh` (https://cli.github.com/).
2. From the project root run:

```bash
# create a public repo, push current branch, and open in browser
gh repo create --public --source=. --push --remote=origin --add-readme
```

3. Enable GitHub Pages from the repository settings (choose `main` branch / root). The Pages site URL will be shown in the repo settings.

Option B — manual Git + website:

```bash
cd /home/sayak/Metallicity
git init
git add .
git commit -m "Add Metallicity static site with comments"
# create a repo on GitHub via the website, then add remote and push:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Then go to GitHub → Settings → Pages and set source to `main` branch, folder `/ (root)`. Wait a minute for the site to publish.

Notes

- Comments are stored in each user's browser `localStorage` (not shared). If you want centralized comments, I can add a simple backend and wire the UI to it.
- To clear local comments, open DevTools → Application → Local Storage and remove keys with `::comments`.
