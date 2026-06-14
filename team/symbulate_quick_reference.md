# Symbulate Project — Quick Reference Sheet
### Cal Poly Statistics Frost SURP — Summer 2026

Keep this sheet open during every work session.

---

## Every Session: Start Here

### Mac and Windows Workflow A — VS Code terminal only
```bash
conda activate symbulate
git checkout dev
git pull origin dev
git checkout your-branch-name
git merge dev
```

### Windows Workflow B — two terminals
```bash
# In standalone Git Bash (Start menu):
conda activate symbulate             # leave this window open

# In VS Code terminal:
conda activate symbulate
git checkout dev
git pull origin dev
git checkout your-branch-name
git merge dev
```

✅ Prompt should show: `(symbulate)` + `calpoly-symbulate` folder + branch name

---

## Every Session: End Here

```bash
pytest tests/                        # confirm nothing broken
git add -p                           # review and stage changes
git commit -m "type: description"
git push origin your-branch-name
```

---

## Starting a New Task

```bash
git checkout dev
git pull origin dev
git checkout -b type/short-description
```

Branch naming:
```
fix/issue-93-boxmodel-typeerror
docs/normal-distribution-docstring
feature/beta-binomial-distribution
test/poisson-simulation-tests
```

---

## ⚠️ Pull Request Warning — Check Every Time

When you open a PR, GitHub **always defaults to the wrong repo.**

**Before clicking "Create pull request" verify:**

| Field | Must say | Must NOT say |
|---|---|---|
| base repository | `calpoly-symbulate/symbulate` | `dlsun/symbulate` ❌ |
| base branch | `dev` | `master` or `main` ❌ |

Change the dropdowns BEFORE creating. Every single time.

---

## Pull Request Checklist

- [ ] `pytest tests/` passes locally
- [ ] Changed functions have updated NumPy docstrings
- [ ] New functionality has at least one test
- [ ] Error messages are student-friendly
- [ ] Branch is up to date: `git merge dev` before opening
- [ ] PR description references the GitHub issue number
- [ ] base repository = `calpoly-symbulate/symbulate` ✅
- [ ] base branch = `dev` ✅

---

## After PR is Merged

```bash
git checkout dev
git pull origin dev
git branch -d your-branch-name
```

Also delete the branch on GitHub.

---

## Commit Message Format

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

## Running Tests

```bash
pytest tests/                 # run everything
pytest tests/ -v              # verbose
pytest tests/ -k "normal"     # tests matching "normal"
pytest tests/ --tb=short      # shorter error messages
```

In VS Code: click the **Testing** icon (flask) in the left sidebar.

---

## Git Essentials

```bash
git status                    # what has changed?
git log --oneline -10         # last 10 commits
git diff                      # see unstaged changes
git stash                     # set changes aside
git stash pop                 # restore stashed changes
git branch -a                 # list all branches
```

---

## Conda Essentials

```bash
conda activate symbulate          # run at start of EVERY session
conda deactivate                  # return to base
conda env update -f environment.yml  # after environment file changes
```

---

## Using Claude

**In the browser:** Always start from the team's shared Symbulate Project at [claude.ai](https://claude.ai)

**In the terminal:**
```bash
claude                          # interactive session
claude "your question here"     # quick one-off question
```

**Useful prompts:**
- *"Explain what this function does and what would break if I changed line X"*
- *"Review this diff for mathematical correctness, edge cases, and student-friendly error messages"*
- *"Implement [distribution] following distributions.py patterns. Parameters: [...]. Mean: [...]. Variance: [...]."*
- *"List 10 mathematical properties of [distribution] I could use as pytest tests"*
- *"Write a NumPy-style docstring for this function following this example: [paste example]"*

**Always:** Run code Claude produces before committing. Ask *"What could go wrong with this?"*

---

## VS Code Keyboard Shortcuts

| Action | Windows | Mac |
|---|---|---|
| Open/close terminal | Ctrl+\` | Cmd+\` |
| Command Palette | Ctrl+Shift+P | Cmd+Shift+P |
| Extensions panel | Ctrl+Shift+X | Cmd+Shift+X |
| Source Control | Ctrl+Shift+G | Cmd+Shift+G |
| Save file | Ctrl+S | Cmd+S |
| Format document | Shift+Alt+F | Shift+Option+F |
| Undo | Ctrl+Z | Cmd+Z |

---

## When Stuck

1. **Ask Claude** — paste the exact error and relevant code
2. **Check the GitHub issue** — context may already be there
3. **Ask a teammate**
4. **Bring to team meeting**

---

## Rules That Never Change

- `conda activate symbulate` — every session, every terminal
- Never push directly to `main` or `dev`
- Every PR: check base repository and base branch before creating
- Every function you touch gets an updated docstring
- Every bug fix or new feature includes a test
- Run `pytest tests/` before pushing
- No new dependencies without team agreement

---

*Cal Poly Statistics Frost SURP — Summer 2026*
