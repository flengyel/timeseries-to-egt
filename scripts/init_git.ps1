param(
    [string]$RepoName = "timeseries-to-egt",
    [string]$GitUser  = "flengyel",    # optional override
    [string]$GitEmail = "florian.lengyel@gmail.com",    # optional override
    [string]$Visibility = "public"  # or "private"
)

# Fail on errors
$ErrorActionPreference = "Stop"

# Ensure we are in the project root (contains pyproject.toml and src/)
if (!(Test-Path -Path ".\pyproject.toml")) {
    Write-Error "Run this from the project root (pyproject.toml not found)."
}

# Basic Git config (optional)
if ($GitUser -ne "")  { git config user.name  "$GitUser" }
if ($GitEmail -ne "") { git config user.email "$GitEmail" }

# Windows line ending normalization
git config core.autocrlf true

# Init, add, commit
git init
git add .
git commit -m "Initial commit: ts2eg package, tests, demo, notebooks, math background"

# Create remote + push
# Preferred: GitHub CLI if present
if (Get-Command gh -ErrorAction SilentlyContinue) {
    # If repo already exists on GitHub, this will fail; fallback provided below
    try {
        gh repo create $RepoName --source . --$Visibility --remote origin --push
        Write-Host "Pushed to GitHub via gh CLI."
        exit 0
    } catch {
        Write-Warning "gh repo create failed (maybe repo exists). Falling back to manual remote add."
    }
}

Write-Host "Manual step:" -ForegroundColor Yellow
Write-Host "1) Create an empty repo on GitHub named $RepoName (Public/Private)." -ForegroundColor Yellow
Write-Host "2) Then run the following commands (replace YOURUSER if needed):" -ForegroundColor Yellow
Write-Host ""
Write-Host ("    git remote add origin https://github.com/YOURUSER/{0}.git" -f $RepoName)
Write-Host "    git branch -M main"
Write-Host "    git push -u origin main"
