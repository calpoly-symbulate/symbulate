# Getting Started: Software Development Tools and Workflow
### Symbulate Project — Summer 2026 — Cal Poly Statistics Frost SURP

This guide covers everything you need to set up your development environment and learn the tools we will use all summer. Work through each section in order — each step builds on the previous one.

At the end of each major section there is a **✅ Checkpoint** — confirm it before moving on. Catching a problem early is much easier than troubleshooting it later.

> **A note on experience level:** These instructions assume you have used Python before but are not a trained software developer. Every step is explained in enough detail that you should be able to follow it without prior experience. If something does not work as described, bring the exact error message to the team and we will sort it out together.

---

## Table of Contents

1. [Accounts and Access](#1-accounts-and-access)
2. [Getting Started with the Terminal](#2-getting-started-with-the-terminal)
3. [Python Environment: Anaconda Setup](#3-python-environment-anaconda-setup)
4. [Install and Configure VS Code](#4-install-and-configure-vs-code)
5. [Test Conda in VS Code — Choose Your Workflow](#5-test-conda-in-vs-code--choose-your-workflow)
6. [Git and GitHub Setup](#6-git-and-github-setup)
7. [Set Up Claude and Claude Code](#7-set-up-claude-and-claude-code)
8. [Learn the Tools: Recommended Tutorials](#8-learn-the-tools-recommended-tutorials)

---

## 1. Accounts and Access

### 1.1 Create a GitHub Account

If you do not already have one, create a free account at [github.com](https://github.com). Use a professional username — this account will be visible publicly and can be part of your portfolio.

### 1.2 Enable Two-Factor Authentication (2FA) on GitHub

GitHub requires all users to enable 2FA. **You must do this before July 27, 2026** or GitHub will restrict your account access — which would block you from contributing during the project.

1. Sign into GitHub → click your profile photo → **Settings**
2. Left sidebar → **Password and authentication**
3. Scroll to **Two-factor authentication** → click **Enable**
4. Choose **Authenticator app** (more reliable than SMS)
   - If you do not have one, download **Microsoft Authenticator** or **Google Authenticator** on your phone first
5. Follow GitHub's setup steps
6. **Save your recovery codes** somewhere safe — if you lose your phone, these are the only way back into your account

### 1.3 Share Your GitHub Username

Send your GitHub username to the supervisor so you can be added to the project repository. You will receive an email invitation to join.

### 1.4 Accept the Claude Team Invitation

You will receive an email invitation to join the team's Claude subscription. Accept it and sign in at [claude.ai](https://claude.ai) to confirm your account is active. You should see the team's shared Symbulate Project in the left sidebar.

✅ **Checkpoint 1:** GitHub account created with 2FA enabled, invitations accepted, Claude Team access confirmed.

---

## 2. Getting Started with the Terminal

Most setup steps in this guide involve typing commands into a terminal. This section gives you just enough to get through everything else.

### 2.1 What Is a Terminal?

A terminal is a text-based way to interact with your computer. Instead of clicking icons, you type commands and press Enter. Many developer tools — including Git, conda, and Claude Code — are primarily used through the terminal.

Every command in this guide that appears in a gray code box is meant to be typed into a terminal and run by pressing Enter.

### 2.2 How to Open a Terminal

**Windows:** Use **Git Bash**, installed automatically with Git (Section 6.1). Open it from the Windows Start menu — search for "Git Bash" and click it. A black window opens with a prompt like:
```
yourname@COMPUTER MINGW64 ~
$
```
The `MINGW64` in the prompt confirms you are in the correct mode. If you see `MSYS` instead, close it and reopen.

> Windows also has PowerShell and Command Prompt — do not use these for this project. They use different syntax and conda does not work correctly in them.

**Mac:** Use **Terminal**, built into macOS. Open it from Applications → Utilities → Terminal, or press Cmd+Space, type "Terminal," and press Enter. A window opens with a prompt like:
```
yourname@computer ~ %
```

### 2.3 The Two Terminals You Will Use (Windows only)

Mac users have one terminal that works for everything. Windows users will use two:

**Git Bash (standalone)** — opened from the Start menu. Used during initial Anaconda setup before VS Code is fully configured.

**VS Code integrated terminal** — opened inside VS Code with Ctrl+\`. Used for all day-to-day work once setup is complete.

Section 5 walks you through determining which workflow applies to your Windows machine.

### 2.4 Essential Commands

These commands work in both Git Bash (Windows) and Terminal (Mac):

| Command | What it does | Example |
|---|---|---|
| `pwd` | Show where you are | `pwd` |
| `ls` | List files in current folder | `ls` |
| `cd foldername` | Move into a folder | `cd Documents` |
| `cd ..` | Move up one folder | `cd ..` |
| `mkdir foldername` | Create a new folder | `mkdir projects` |

**Useful habits:**
- **Tab completion:** Start typing a name and press Tab — the terminal autocompletes it
- **Up arrow:** Cycles through previously typed commands
- **Copy/paste:**
  - Windows standalone Git Bash: **Shift+Insert** to paste
  - Windows VS Code terminal: **Ctrl+V**
  - Mac Terminal: **Cmd+V** everywhere

### 2.5 Practice: A Two-Minute Exercise

```bash
pwd                   # see where you are
ls                    # see what is in this folder
mkdir terminal-test   # create a new folder
cd terminal-test      # move into it
pwd                   # confirm you moved
cd ..                 # go back up
ls                    # confirm terminal-test is listed
```

✅ **Checkpoint 2:** All commands ran without errors and `pwd` showed a file path.

---

## 3. Python Environment: Anaconda Setup

Everyone on the team must use the same Python and package versions. This prevents the "works on my machine" problem.

> **Expected versions for this project:**
> - conda: 25.11.1
> - Python: 3.13.9
>
> Share your version numbers with each other to confirm everyone matches.

### 3.1 Check What You Already Have

**Windows:** Open Git Bash (Start menu → search "Git Bash")
**Mac:** Open Terminal

Run:
```bash
conda --version
python --version
```

- **If you see `conda 25.11.1` and `Python 3.13.9`:** Skip to Section 3.4.
- **If you see different versions or "command not found":** Continue to Section 3.2.

### 3.2 Uninstall Existing Anaconda (if installed)

> Skip this section if Anaconda was never installed on your machine.

#### Windows

**Step 1: Uninstall via Windows Settings**
1. Start menu → **Settings → Apps**
2. Search for **Anaconda** — uninstall all entries
3. When shown "Advanced Uninstallation Options," check all three boxes:
   - ✅ Remove user configuration files
   - ✅ Remove user data
   - ✅ Remove caches
4. If error dialogs appear ("Failed to run pre_uninstall" or "Failed to remove files") — click **Ignore** and continue. These are harmless.

**Step 2: Delete leftover folders manually**

Open File Explorer and delete these if they exist (replace `yourusername` with your Windows username):
```
C:\Users\yourusername\AppData\Local\anaconda3
C:\Users\yourusername\AppData\Local\Continuum
C:\Users\yourusername\AppData\Local\conda
C:\Users\yourusername\AppData\Local\conda-anaconda-tos
C:\Users\yourusername\AppData\Roaming\conda
C:\Users\yourusername\AppData\Roaming\.anaconda
C:\Users\yourusername\.conda
C:\Users\yourusername\.condarc
C:\Users\yourusername\.anaconda_backup
C:\anaconda3
C:\Anaconda3
```
> To navigate to hidden AppData folders: type the path directly into the File Explorer address bar and press Enter.
> Do NOT delete anything inside `AppData\Local\Programs\` — those belong to other installed applications.

**Step 3: Clean up Start menu shortcuts**

Delete this folder if it exists:
```
C:\Users\yourusername\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda (anaconda3)
```

**Step 4: Clean up PATH and environment variables**
1. Search "Environment Variables" in the Start menu → "Edit the system environment variables" → "Environment Variables"
2. Under User variables, find **Path** → click **Edit**
3. Delete any entries containing `anaconda`, `conda`, or `pyenv`
4. Click OK
5. Also delete these user variables entirely if they exist: `PYENV`, `PYENV_HOME`, `PYENV_ROOT`

**Step 5: Restart your computer** — do not skip this step.

**Step 6: Verify Anaconda is gone**
```bash
conda --version
python --version
```
Both should say "command not found."

#### Mac

**Step 1: Run the Anaconda cleanup tool**

Open Terminal and run:
```bash
conda install anaconda-clean
anaconda-clean --yes
```

**Step 2: Delete the Anaconda folder**

Depending on where Anaconda was installed, delete one of these:
```bash
rm -rf ~/anaconda3
rm -rf ~/opt/anaconda3
```

**Step 3: Remove conda from your shell config**

Open your shell config file. On newer Macs (zsh):
```bash
open ~/.zshrc
```
On older Macs (bash):
```bash
open ~/.bash_profile
```
Find and delete the block that looks like this:
```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
...
# <<< conda initialize <<<
```
Save the file, then close and reopen Terminal.

**Step 4: Verify Anaconda is gone**
```bash
conda --version
python --version
```
Both should say "command not found."

### 3.3 Install Anaconda Fresh

Download the latest **Anaconda Distribution** from [anaconda.com/download](https://www.anaconda.com/download). Choose the installer for your operating system.

#### Windows installation

Run the installer and make these specific choices:
- Install for **"Just Me"** (not "All Users")
- Accept the default install location
- On the Advanced Options screen:
  - ✅ **Add Anaconda3 to my PATH environment variable** — the installer warns "not recommended" — **check it anyway**. This is what makes conda work in Git Bash.
  - ✅ **Register Anaconda3 as default Python 3.13** — check this so VS Code finds it automatically
  - Leave "Clear the package cache" unchecked

> If the installer shows "Directory is not empty" — the old installation was not fully removed. Click OK, complete Section 3.2, then run the installer again.

#### Mac installation

Run the `.pkg` installer and accept all defaults. The Mac installer handles PATH setup automatically — no special choices needed.

> You will see **"A newer version of conda exists"** warning during or after installation — **ignore it** and proceed. We use the version bundled with the installer so everyone stays synchronized.

### 3.4 Connect Conda to Your Shell

#### Windows

Open **Git Bash** (Start menu → search "Git Bash") and run:
```bash
conda init bash
```
Then **close Git Bash completely and reopen it**. The init does not take effect until you open a fresh window.

#### Mac

Open **Terminal** and check which shell you are using:
```bash
echo $SHELL
```

If it shows `/bin/zsh` (most Macs):
```bash
conda init zsh
```

If it shows `/bin/bash`:
```bash
conda init bash
```

Then **close Terminal completely and reopen it**.

### 3.5 Verify the Installation

In a fresh terminal window:
```bash
conda --version
python --version
```

Expected output:
```
conda 25.11.1
Python 3.13.9
```

✅ **Checkpoint 3:** Both commands return the expected version numbers. Share them in the team chat.

---

## 4. Install and Configure VS Code

Visual Studio Code is our shared code editor. Everyone uses the same editor with the same settings.

> **If you already have VS Code installed:** No reinstall needed. Go to **Help → Check for Updates**, then continue from Section 4.2.

### 4.1 Download and Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com). Choose the installer for your operating system and install with default options.

### 4.2 Set the Default Terminal

#### Windows

VS Code defaults to PowerShell — change this immediately:

1. Open VS Code
2. Open Command Palette: **Ctrl+Shift+P**
3. Type **"Terminal: Select Default Profile"** and press Enter
4. Select **Git Bash**

Then verify the Git Bash configuration is correct:
1. Command Palette → **"Preferences: Open User Settings (JSON)"**
2. Make sure these lines exist (add them if missing):
```json
"terminal.integrated.profiles.windows": {
    "Git Bash": {
        "path": "C:\\Program Files\\Git\\bin\\bash.exe",
        "args": []
    }
},
"terminal.integrated.defaultProfile.windows": "Git Bash"
```
> The path must be `bin\bash.exe` (not `usr\bin\bash.exe`) and `args` must be empty `[]`.

#### Mac

VS Code uses zsh by default on Mac, which is correct — no changes needed.

### 4.3 Install Required Extensions

Open the Extensions panel (Ctrl+Shift+X on Windows, Cmd+Shift+X on Mac). Search for and install each:

| Extension | Publisher | Purpose |
|---|---|---|
| Python | Microsoft | Core Python support |
| Pylance | Microsoft | Type checking and autocomplete |
| Black Formatter | Microsoft | Auto-formats code on save |
| Claude Code | Anthropic | Claude AI inside VS Code |
| GitLens | GitKraken | Visual Git history and blame |
| GitHub Pull Requests | GitHub | Create and review PRs inside VS Code |
| autoDocstring | Nils Werner | Generates docstring templates |
| Error Lens | Alexander | Shows errors inline |

### 4.4 Uninstall the Python Environments Extension (Windows only)

> Mac users can skip this step.

On Windows, the Python Environments extension installs itself automatically alongside the Python extension and causes conda activation conflicts on some machines. Uninstall it:

1. Extensions panel → search **"Python Environments"**
2. Click on it (published by Microsoft)
3. Click **Uninstall**
4. Restart VS Code when prompted

If it does not appear under Installed, move on.

### 4.5 Sign Into GitHub

1. Click the **Accounts** icon at the bottom of the left sidebar (person silhouette)
2. Click **"Sign in with GitHub"** and follow the browser authentication
3. Also sign into GitLens when prompted

> **Ignore any GitHub Copilot sign-in prompts** — we use Claude, not Copilot.

### 4.6 How to Open VS Code for This Project

Always open VS Code first, then open the project folder:

1. Open VS Code from the Start menu (Windows) or Applications folder / Dock (Mac)
2. **File → Open Folder** → navigate to your `calpoly-symbulate` folder → **Select Folder**

Or click the folder name under **Recent** on the Welcome screen.

> Do not double-click Python files to launch VS Code — the shared project settings will not load correctly.

### 4.7 Opening the Terminal in VS Code

| | Windows | Mac |
|---|---|---|
| Open terminal | Ctrl+\` | Cmd+\` |
| New terminal | Ctrl+Shift+\` | Cmd+Shift+\` |

After opening the terminal, always activate the conda environment:
```bash
conda activate symbulate
```

✅ **Checkpoint 4:** VS Code open, correct terminal default set, all required extensions installed, Python Environments extension uninstalled (Windows), signed into GitHub.

---

## 5. Test Conda in VS Code — Choose Your Workflow

This step determines your day-to-day workflow. Conda activation works differently on different machines, especially on Windows.

> Complete Section 2 of the Symbulate project guide first (clone the repo and create the conda environment), then come back here.

### 5.1 Test Conda in the VS Code Terminal

Open VS Code with the project folder, open the terminal (Ctrl+\` or Cmd+\`), and run:
```bash
conda activate symbulate
```

---

### ✅ If it works — Workflow A (Single Terminal)

`(symbulate)` appears with no errors. **All work happens in the VS Code terminal.** Run `conda activate symbulate` at the start of every session. You do not need a separate terminal window.

Set the Python interpreter in your user settings:
1. Command Palette → **"Preferences: Open User Settings (JSON)"**
2. Add this line:

**Windows** (replace `yourusername`):
```json
"python.defaultInterpreterPath": "C:\\Users\\yourusername\\AppData\\Local\\anaconda3\\envs\\symbulate\\python.exe"
```

**Mac** (replace `yourusername`):
```json
"python.defaultInterpreterPath": "/Users/yourusername/anaconda3/envs/symbulate/bin/python"
```

Confirm the bottom status bar shows `symbulate (3.13.x)`.

---

### ❌ If it fails — Workflow B (Two Terminals, Windows only)

If you see an error like `bash: /cygdrive/c/...conda.exe: No such file or directory`, use this reliable fallback:

**Use two terminals side by side:**

| Terminal | Use for |
|---|---|
| **Standalone Git Bash** | `conda activate symbulate` only |
| **VS Code terminal** | Everything else: git, pytest, claude, python |

**Session start with Workflow B:**
1. Open **Git Bash** from the Start menu → run `conda activate symbulate` → leave open
2. Open **VS Code** → open project folder → open VS Code terminal → run `conda activate symbulate`

**Troubleshooting steps to try before using Workflow B:**
1. Verify Git Bash path in user settings is `C:\\Program Files\\Git\\bin\\bash.exe` with `"args": []`
2. Uninstall the Python Environments extension (Section 4.4) if not done already
3. Run `conda init bash` in standalone Git Bash, then close and reopen VS Code completely

If none of these work, use Workflow B and move on. It is reliable and you will not be disadvantaged.

> Mac users should not encounter this issue. If conda fails in the Mac VS Code terminal, contact the supervisor.

---

> The garbled text at VS Code terminal startup on Windows (`←]633;P;IsWindows=True...`) is harmless — ignore it.

✅ **Checkpoint 5:** You know which workflow applies (A or B) and conda activation works consistently.

---

## 6. Git and GitHub Setup

> **One-time setup.** Everything in this section is done once. Your daily Git workflow is in the Quick Reference sheet.

### 6.1 Install Git

**Windows:** Download from [git-scm.com](https://git-scm.com). During installation, choose "Git from the command line and also from 3rd-party software" when asked about PATH. Accept all other defaults.

**Mac:** Git may already be installed. Check with:
```bash
git --version
```
If not installed, macOS will prompt you to install Xcode Command Line Tools — click Install and wait for it to complete. Alternatively run:
```bash
xcode-select --install
```

### 6.2 Configure Git with Your Identity

Run in your terminal (Git Bash on Windows, Terminal on Mac):
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```
Use the same email as your GitHub account. Confirm it saved:
```bash
git config --global user.name
git config --global user.email
```

### 6.3 Set Up SSH Authentication

SSH lets you push code to GitHub without typing your password every time.

> **Windows:** Run these in Git Bash (Start menu), not PowerShell.
> **Mac:** Run these in Terminal.

**Generate a key:**
```bash
ssh-keygen -t ed25519 -C "your@email.com"
```
Press Enter three times to accept all defaults.

**Copy your public key:**

Windows:
```bash
cat ~/.ssh/id_ed25519.pub
```

Mac:
```bash
cat ~/.ssh/id_ed25519.pub
```
Or use the clipboard directly:
```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

Select and copy the entire output (starts with `ssh-ed25519`, ends with your email).

**Add to GitHub:**
GitHub → profile photo → Settings → SSH and GPG keys → **New SSH key** → paste → **Add SSH key**

**Test:**
```bash
ssh -T git@github.com
```
Expected: `Hi username! You've successfully authenticated...`

✅ **Checkpoint 6:** SSH test shows your GitHub username.

### 6.4 Complete GitHub Tutorials (Required before first team meeting)

**Introduction to GitHub:** [github.com/skills/introduction-to-github](https://github.com/skills/introduction-to-github) — ~1 hour

**Review Pull Requests:** [github.com/skills/review-pull-requests](https://github.com/skills/review-pull-requests) — ~30 minutes

---

## 7. Set Up Claude and Claude Code

### 7.1 Confirm Claude Web Access

Sign in at [claude.ai](https://claude.ai). Confirm the team's shared Symbulate Project is visible in the left sidebar. If not, contact the supervisor.

### 7.2 Install Node.js

Claude Code requires Node.js version 18 or later. Check in your terminal:
```bash
node --version
```

If not installed or older than v18, download from [nodejs.org](https://nodejs.org) — choose **LTS**. Install with default options.

After installing:
- **Windows:** Close and reopen VS Code completely before Node.js is recognized
- **Mac:** Close and reopen Terminal, then open a new VS Code terminal

### 7.3 Install Claude Code

In the VS Code terminal:
```bash
npm install -g @anthropic-ai/claude-code
```

### 7.4 Authenticate

```bash
claude
```

On first launch:
- Choose your preferred theme
- Select **"Yes, use recommended settings"** for terminal setup
- Select **"Claude account with subscription"** and log in with your Claude Team account

When authenticated correctly you will see: **Cal Poly Statistics Frost SURP**

### 7.5 Verify

```bash
claude --version
```

✅ **Checkpoint 7:** `claude --version` returns a version number.

---

## 8. Recommended Tutorials

- **Introduction to GitHub:** [github.com/skills/introduction-to-github](https://github.com/skills/introduction-to-github)
- **Review Pull Requests:** [github.com/skills/review-pull-requests](https://github.com/skills/review-pull-requests)
- **pytest:** [docs.pytest.org/en/stable/getting-started.html](https://docs.pytest.org/en/stable/getting-started.html) — ~45 minutes
- **NumPy Docstrings:** [numpydoc.readthedocs.io/en/latest/format.html](https://numpydoc.readthedocs.io/en/latest/format.html) — ~20 minutes
- **Claude Code:** [docs.anthropic.com/en/docs/claude-code](https://docs.anthropic.com/en/docs/claude-code) — ~30 minutes
- **Prompt Engineering:** [docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — ~30 minutes
- **Python Packaging:** [packaging.python.org/en/latest/tutorials/packaging-projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) — ~30 minutes

---

✅ **Final Checklist — ready to start the Symbulate project guide when:**
- [ ] GitHub account created with 2FA enabled
- [ ] Claude Team invitation accepted, shared Project visible at claude.ai
- [ ] `conda --version` = 25.11.1 and `python --version` = 3.13.9 — shared in team chat
- [ ] VS Code installed with all required extensions
- [ ] Python Environments extension uninstalled (Windows)
- [ ] Correct terminal default configured for your OS
- [ ] Workflow A or B identified and tested (Windows) — conda activation works
- [ ] `ssh -T git@github.com` shows your username
- [ ] `claude --version` returns a version number
- [ ] GitHub tutorials completed

---

*Last updated: June 2026 — Cal Poly Statistics Frost SURP. Direct questions to the project supervisor.*
