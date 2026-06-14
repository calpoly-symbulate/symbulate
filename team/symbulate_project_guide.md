# Getting Started: The Symbulate Project
### Summer 2026 — Cal Poly Statistics Frost SURP

This guide covers everything specific to the Symbulate package and how your team will work on it. Complete `symbulate_dev_setup.md` before starting here.

> **Windows vs. Mac:** Where instructions differ by operating system, sections are clearly labeled **Windows** and **Mac**. Where no label appears, the instructions are the same for both.

---

## Table of Contents

1. [Understanding the Codebase](#1-understanding-the-codebase)
2. [Clone and Install Symbulate](#2-clone-and-install-symbulate)
3. [About the Broken Main Branch](#3-about-the-broken-main-branch)
4. [Project Goals and Priorities](#4-project-goals-and-priorities)
5. [Git Workflow for This Project](#5-git-workflow-for-this-project)
6. [Write the CLAUDE.md File Together](#6-write-the-claudemd-file-together)
7. [Unit Testing: Start Now, Not Later](#7-unit-testing-start-now-not-later)
8. [Using Claude Effectively as a Team](#8-using-claude-effectively-as-a-team)
9. [Your First Contribution: Practice PR](#9-your-first-contribution-practice-pr)
10. [Daily Workflow Reference](#10-daily-workflow-reference)
11. [Team Conventions Cheat Sheet](#11-team-conventions-cheat-sheet)

---

## 1. Understanding the Codebase

Before writing a single line of code, spend time understanding what already exists. Students who skip this step spend the rest of the summer making changes that break things they did not know were connected.

### 1.1 The Repositories

**Project repository — this is where you work:**
[github.com/calpoly-symbulate/symbulate](https://github.com/calpoly-symbulate/symbulate)
Clone this repo. All team work happens here.

**Original repository — read-only reference:**
[github.com/dlsun/symbulate](https://github.com/dlsun/symbulate)
Dennis Sun's original. Do not clone or work in it. Use as a reference only.

**Additional reference fork — read-only reference:**
[github.com/kevindavisross/symbulate](https://github.com/kevindavisross/symbulate)
Contains additional features. Also read-only reference.

### 1.2 Key Files to Read First

Assign each team member one file to become the temporary "expert" on:

| File | What it contains |
|---|---|
| `symbulate/probability_space.py` | `BoxModel`, `DeckOfCards`, core probability spaces |
| `symbulate/distributions.py` | All probability distributions |
| `symbulate/results.py` | Simulation result objects |
| `symbulate/plot.py` | All plotting and graphics |
| `symbulate/__init__.py` | The public API — what users import |

Use Claude to help understand each file:
> *"Explain the architecture of this file and how it connects to the rest of Symbulate. I am a statistics student, not a software developer."*

Each student gives a 5-minute summary to the group. After this everyone should have a mental map of the whole package.

### 1.3 Compare the Reference Repos with Claude

> *"Here is the same file from two forks of Symbulate. What changes were made in the second version, and why might they have been made?"*

Paste both versions and discuss as a team. This surfaces useful ideas without committing to anything.

---

## 2. Clone and Install Symbulate

> **Which terminal to use:**
> - **Mac and Windows Workflow A users:** Use the VS Code terminal for everything below
> - **Windows Workflow B users:** Use standalone Git Bash for conda commands, VS Code terminal for git and everything else

### 2.1 Clone the Project Repository

In your terminal:
```bash
cd ~/Documents
git clone git@github.com:calpoly-symbulate/symbulate.git calpoly-symbulate
cd calpoly-symbulate
```

We use the folder name `calpoly-symbulate` to distinguish it from any other symbulate folders on your machine.

### 2.2 Verify Both Branches Exist

```bash
git branch -a
```

You should see both `main` and `remotes/origin/dev` listed.

### 2.3 Create the Conda Environment

The repository contains an `environment.yml` file that specifies every package the project needs:

```bash
conda env create -f environment.yml
```

This may take several minutes. When it finishes:

```bash
conda activate symbulate
```

You should see `(symbulate)` at the start of your prompt.

> **Windows Workflow B users:** Run `conda activate symbulate` in standalone Git Bash. Then also run it in the VS Code terminal.

### 2.4 Verify the Environment

```bash
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
```

Share your output in the team chat to confirm everyone has matching versions. If any command fails, contact the supervisor before proceeding.

### 2.5 Install Symbulate in Development Mode

```bash
pip install -e .
```

> **Note the dot at the end** — it is required and easy to miss. `pip install -e` without the dot will fail. The dot means "install the package in the current directory."

The `-e` flag means "editable" — when you edit the source code, changes take effect immediately without reinstalling.

### 2.6 Test the Install

```bash
python -c "import symbulate; print('Symbulate imported successfully')"
```

You may see warnings — that is expected given the known bugs. What matters is whether the import works at all.

### 2.7 Complete the Workflow Test

Now that the `symbulate` environment exists, go back to Section 5 of `symbulate_dev_setup.md` and complete the conda workflow test. Then come back here.

### 2.8 Set the Python Interpreter in VS Code

Add the interpreter path to your VS Code user settings:

1. Command Palette → **"Preferences: Open User Settings (JSON)"**
   - Windows: Ctrl+Shift+P
   - Mac: Cmd+Shift+P
2. Add this line (replace `yourusername` with your actual username):

**Windows:**
```json
"python.defaultInterpreterPath": "C:\\Users\\yourusername\\AppData\\Local\\anaconda3\\envs\\symbulate\\python.exe"
```

**Mac:**
```json
"python.defaultInterpreterPath": "/Users/yourusername/anaconda3/envs/symbulate/bin/python"
```

Open any `.py` file and confirm the bottom status bar shows `symbulate (3.13.x)`.

> If VS Code shows "Invalid Python interpreter" after switching branches or pulling changes, click **Select Python Interpreter** and reselect `symbulate (3.13.x)`. This is normal.

### 2.9 Open the Project in VS Code

From the terminal:
```bash
code .
```

Or open VS Code → **File → Open Folder** → navigate to `calpoly-symbulate`.

✅ **Checkpoint 2:** `(symbulate)` active, all packages import, `Symbulate imported successfully` prints, status bar shows `symbulate (3.13.x)`, workflow confirmed.

---

## 3. About the Broken Main Branch

The current `main` branch of Symbulate has bugs that prevent it from running correctly. **This is expected and is one of the first things your team will fix.** It is not a problem with your setup.

### What "broken" means in practice

The package imports successfully but certain operations fail with errors. You will see these when you run the test suite.

### Why we use main anyway

`main` is the established, canonical version of Symbulate that users have installed. Fixing it — rather than replacing it with a fork — is the right approach. The reference forks contain many changes mixed together. Copying from them wholesale without understanding each change would make the codebase harder to maintain.

### How to handle this in the first week

**Step 1: Run the test suite and document what fails:**
```bash
pytest tests/ -v
```
Save this output — it is your baseline.

**Step 2: Browse the GitHub issues:**
[github.com/calpoly-symbulate/symbulate/issues](https://github.com/calpoly-symbulate/symbulate/issues)

**Step 3: Triage with Claude:**
> *"Group these issues by type, estimate difficulty, and suggest a priority order for a team of statistics students new to software development."*

**Step 4: Fix the first bug as a team** before splitting into individual tasks.

### Using the reference forks

When you look at a bug, check whether a reference fork has addressed it. If so, copy code only after the team understands what it does:
> *"Explain what this change does and why it fixes the bug. Are there any tradeoffs or risks?"*

---

## 4. Project Goals and Priorities

Five areas of work in rough priority order:

**Priority 1: Fix Bugs and Ensure Compatibility**
Fix known bugs and ensure the package works with current Python (3.13.x) and common packages. Everything else depends on a working package.

**Priority 2: Enhance Documentation**
Improve docstrings, error messages, tutorial notebooks, and vignettes. Run **in parallel with everything else** — every time a student touches a function, they improve its docstring and error messages in the same PR.

**Priority 3: Overhaul Graphics**
Improve default plot appearance, expand customization, maintain the all-purpose `.plot()` interface. Before writing any code, collect examples of current plots, identify specific problems, and decide what "better" looks like.

**Priority 4: Add Probability Models**
New distributions, hierarchical specifications, stochastic processes. Each new distribution is largely self-contained — well-suited to parallel work once bug fixes are stable.

**Priority 5: More Efficient Simulation Methods**
MCMC and other methods for slow or difficult-to-simulate models. Most technically demanding — save for the second half of the summer.

---

## 5. Git Workflow for This Project

### 5.1 Branch Structure

```
main          ← stable baseline; protected; never commit here directly
    ↑
dev           ← integration branch; all team work merges here first
    ↑
feature branches  ← one per task; all actual work happens here
```

### 5.2 Branch Naming

```
fix/issue-NUMBER-short-description      # bug fixes
docs/what-is-being-documented           # documentation
feature/what-is-being-added             # new functionality
test/what-is-being-tested               # adding tests
```

Examples:
```
fix/issue-93-boxmodel-typeerror
docs/normal-distribution-docstring
feature/beta-binomial-distribution
```

### 5.3 Starting a New Task

```bash
git checkout dev
git pull origin dev
git checkout -b fix/issue-93-boxmodel-typeerror
```

Or in VS Code: click the branch name in the bottom-left corner → "Create new branch from..." → select `dev` → type the name.

### 5.4 During a Work Session

Commit small and often. If you cannot summarize the change in one sentence, it should probably be multiple commits.

**Via terminal:**
```bash
git add -p                            # review changes before staging
git commit -m "fix: description"
git push origin your-branch-name
```

**Via VS Code Source Control panel:**
1. Click **+** next to files to stage them
2. Type commit message in the box at the top
3. Click the checkmark to commit
4. Click **Sync Changes** to push

### 5.5 Opening a Pull Request

> ⚠️ **CRITICAL: GitHub always defaults to the wrong repository.**
>
> When you open a pull request, GitHub defaults to merging into **`dlsun/symbulate`** (Dennis's original). This is wrong.
>
> **Before clicking "Create pull request" always verify:**
>
> | Field | Must say | Must NOT say |
> |---|---|---|
> | base repository | `calpoly-symbulate/symbulate` | `dlsun/symbulate` ❌ |
> | base branch | `dev` | `master` or `main` ❌ |
>
> Change these in the dropdowns every single time.

**Via VS Code GitHub Pull Requests panel:**
1. Click the GitHub Pull Requests icon in the left sidebar → **+**
2. Set base to `dev`
3. Write description: what changed, why, how tested, GitHub issue number

**Via GitHub website:**
After pushing, click the yellow banner → "Compare & pull request" → check the dropdowns → fill in description → "Create pull request"

### 5.6 Pull Request Checklist

- [ ] `pytest tests/` passes (or pre-existing failures are documented)
- [ ] Changed functions have updated NumPy-style docstrings
- [ ] New functionality has at least one test
- [ ] Error messages are student-friendly
- [ ] Branch is up to date: `git merge dev` before opening PR
- [ ] PR description references the GitHub issue number
- [ ] base repository = `calpoly-symbulate/symbulate` ✅
- [ ] base branch = `dev` ✅

### 5.7 After Your PR is Approved and Merged

```bash
git checkout dev
git pull origin dev
git branch -d your-branch-name
```

Also delete the branch on GitHub (click "Delete branch" after merging).

### 5.8 Commit Message Format

```
type: short description (under 72 characters)
```

| Type | Use for |
|---|---|
| `fix` | bug fixes |
| `docs` | documentation only |
| `feature` | new functionality |
| `test` | adding or fixing tests |
| `refactor` | restructuring without behavior change |
| `style` | formatting only |

---

## 6. Write the CLAUDE.md File Together

Every time anyone opens a Claude Code session in the Symbulate repo, Claude automatically reads `CLAUDE.md` first. Think of it as a standing briefing: *"Here is what you need to know before touching this codebase."*

### When to write it

**First team meeting, before anyone writes any code.** Budget 90 minutes: 45 minutes exploring the codebase together, 45 minutes drafting. Commit it before the meeting ends.

### What to include

**Project overview** — what Symbulate is, who uses it (undergraduate statistics students, not programmers), what the team is doing this summer.

**User philosophy** — error messages must explain what went wrong and how to fix it. Default behavior should work without configuration. Clarity over cleverness.

**Codebase map** — which file contains what (copy from Section 1.2).

**Code conventions** — look at the existing code together and describe what you see. Do not invent conventions that conflict with existing patterns.

**Docstring standard** — paste one example of a well-written docstring. All docstrings follow NumPy style.

**Error message standard** — a before/after example: bare error vs. student-friendly one.

**Testing requirements** — every new function and every bug fix includes a pytest test. No exceptions.

**Graphics standards** — all plots via `.plot()`, defaults look good without arguments, matplotlib only.

**New distribution checklist** — every file that needs to be touched when adding a distribution.

**Git workflow** — branch naming, PR requirements, no direct pushes to `main` or `dev`.

**Do not section** — explicit prohibitions: no direct pushes, no API changes without discussion, no new dependencies without agreement.

### Keep it alive

Treat CLAUDE.md as a living document. When the team makes a new decision, add it the same day. Assign one person to own it and prompt for updates at weekly meetings.

---

## 7. Unit Testing: Start Now, Not Later

The temptation is to defer testing until "after we get the code working." Resist this.

### Test file structure

```
symbulate/
    probability_space.py
    distributions.py
    plot.py
tests/
    test_probability_space.py
    test_distributions.py
    test_plot.py
```

### Four types of tests

**Deterministic** — exact values that should never change:
```python
def test_normal_mean():
    assert Normal(0, 1).mean() == 0
```

**Simulation** — approximate values with fixed random seed:
```python
def test_normal_simulation_mean():
    import numpy as np
    np.random.seed(42)
    result = RV(Normal(0, 1)).sim(10000).mean()
    assert abs(result) < 0.05
```

**Regression** — confirm existing behavior has not changed. Run documented examples now, save outputs, write tests that verify the same outputs later.

**Error** — confirm bad inputs raise the right exception:
```python
def test_boxmodel_helpful_error():
    import pytest
    with pytest.raises(ValueError, match="draw"):
        BoxModel([1, 2, 3], size=10, replace=False)
```

### Running tests

```bash
pytest tests/            # run all tests
pytest tests/ -v         # verbose output
pytest tests/ -k "normal"  # tests matching "normal"
pytest tests/ --tb=short   # shorter error output
```

In VS Code: click the **Testing** icon (flask) in the left sidebar.

---

## 8. Using Claude Effectively as a Team

### Start every session from the shared Project

Go to [claude.ai](https://claude.ai) and open the team's shared Symbulate Project. Starting from the Project means Claude already knows the codebase.

### Translate statistics into code

> *"Implement a Beta-Binomial distribution in Symbulate. Parameters: n, alpha, beta. Mean is n·α/(α+β), variance is n·α·β·(α+β+n) / [(α+β)²·(α+β+1)]. Follow the patterns in distributions.py."*

### Use Claude for code review

When a student opens a PR, a different student pastes the diff into Claude:
> *"Review this change to a Python statistics package. Check for: correctness of mathematical formulas, edge cases, compatibility with the rest of the codebase, and whether error messages would be clear to a statistics student."*

### Generate test cases from math

> *"Given a Normal(μ, σ) distribution, list 10 mathematical properties that should always hold and that I could write pytest tests for."*

### Ask Claude to explain before editing

> *"What does this function do? What would break if I changed line 47? What edge cases does it handle?"*

### When Claude gets something wrong

Always run code Claude produces before committing it. Ask:
> *"What could go wrong with this implementation? What edge cases might it handle incorrectly?"*

---

## 9. Your First Contribution: Practice PR

Before working on any real task, everyone completes this practice exercise to verify the full workflow functions correctly.

### Step 1: Create a branch

```bash
git checkout dev
git pull origin dev
git checkout -b docs/add-my-name-to-contributors
```

### Step 2: Add your name to CONTRIBUTORS.md

Open `CONTRIBUTORS.md` in VS Code and add your name and role. Save the file.

### Step 3: Write a smoke test

Open or create `tests/test_smoke.py` and add:
```python
def test_symbulate_imports():
    """Confirm Symbulate can be imported and core objects are accessible."""
    import symbulate
    assert hasattr(symbulate, 'RV')
    assert hasattr(symbulate, 'BoxModel')
```

Run it:
```bash
pytest tests/test_smoke.py -v
```

### Step 4: Commit and push

```bash
git add CONTRIBUTORS.md tests/test_smoke.py
git commit -m "docs: add my name to contributors, add smoke test"
git push origin docs/add-my-name-to-contributors
```

### Step 5: Open a pull request

Go to GitHub. **Remember:** change base repository to `calpoly-symbulate/symbulate` and base branch to `dev`.

### Step 6: Review a teammate's PR

While waiting for your review, review one of your teammates' PRs in the GitHub Pull Requests panel. Leave at least one comment.

---

## 10. Daily Workflow Reference

### Starting a work session

**Mac and Windows Workflow A:**
```bash
# In VS Code terminal:
conda activate symbulate
git checkout dev
git pull origin dev
git checkout your-branch-name
git merge dev
```

**Windows Workflow B:**
```bash
# In standalone Git Bash:
conda activate symbulate        # leave this window open

# In VS Code terminal:
conda activate symbulate
git checkout dev
git pull origin dev
git checkout your-branch-name
git merge dev
```

✅ Prompt should show: `(symbulate)` + correct folder + branch name

### During a work session

- Run `pytest tests/` frequently
- Commit small, logical changes often
- If stuck: ask Claude → check GitHub issue → ask a teammate

### Ending a work session

```bash
pytest tests/
git add -p
git commit -m "type: description"
git push origin your-branch-name
```

### When your task is complete

1. Open PR to `dev` — verify base repository and base branch
2. Complete the PR checklist
3. Request a teammate as reviewer
4. After merge: `git checkout dev` → `git pull origin dev` → `git branch -d your-branch`

---

## 11. Team Conventions Cheat Sheet

### Quick Reference Commands

```bash
# Conda
conda activate symbulate            # every session, every terminal
conda deactivate                    # return to base
conda env update -f environment.yml # after environment file changes

# Git
git status                          # what has changed?
git log --oneline -10               # last 10 commits
git diff                            # see unstaged changes
git stash / git stash pop           # set changes aside / restore

# Tests
pytest tests/                       # run all
pytest tests/ -v                    # verbose
pytest tests/ -k "normal"           # matching "normal"
pytest tests/ --tb=short            # shorter errors

# Claude
claude                              # interactive session
claude "your question"              # quick question
```

### Rules That Never Change

- `conda activate symbulate` — every session, every terminal
- Never push directly to `main` or `dev`
- Every PR: verify base repository and base branch before creating
- Every function you touch gets an updated docstring
- Every bug fix and new feature includes a test
- Run `pytest tests/` before pushing
- No new dependencies without team agreement
- No public API changes without team discussion

---

*Last updated: June 2026 — Cal Poly Statistics Frost SURP. Direct questions to the project supervisor.*
