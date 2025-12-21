# Git Hotfix Workflow

Quick reference for fixing bugs in `main` while working on `dev` branch.

## Create Hotfix Branch from Main

```bash
# Switch to main and update it
git checkout main
git pull origin main

# Create new branch for the bug fix
git checkout -b hotfix/bug-description

# Make your changes, then commit
git add .
git commit -m "Fix: description of the bug fix"
git push origin hotfix/bug-description
```

## Merge to Main (Command Line)

```bash
# Switch to main
git checkout main
git pull origin main

# Merge the hotfix
git merge hotfix/bug-description

# Push to main
git push origin main
```

## Bring Fix into Dev

```bash
# Switch to dev
git checkout dev

# Merge main (which now has the fix)
git merge main

# Push updated dev
git push origin dev
```

## Cleanup (Optional)

```bash
# Delete hotfix branch locally
git branch -d hotfix/bug-description

# Delete hotfix branch remotely
git push origin --delete hotfix/bug-description
```

## Alternative: Use Pull Request

If your repo requires PRs or code review:
- Push the hotfix branch
- Go to GitHub and create PR from `hotfix/bug-description` → `main`
- Merge via GitHub interface
- Then bring fix into dev as shown above

## Key Points

- Hotfix branch is created from `main`, not `dev`
- This keeps the bug fix separate from dev work
- After merging to main, merge main into dev to get the fix
- Each branch maintains clean, focused history
