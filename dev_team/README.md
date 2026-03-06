# SpeakFlow Dev Team

Multi-agent development pipeline built with CrewAI. Given a module requirements spec, this team autonomously designs, implements, reviews, tests, and ships code via GitHub PRs.

## Agents

| Agent               | Role                                                | LLM           |
| ------------------- | --------------------------------------------------- | ------------- |
| `architect`         | Reads requirements → outputs class/method design    | Claude Sonnet |
| `backend_engineer`  | Implements Python module from design                | Claude Sonnet |
| `code_reviewer`     | Reviews implementation → APPROVE or REQUEST_CHANGES | Claude Sonnet |
| `test_engineer`     | Writes pytest suite with mocks                      | Claude Sonnet |
| `frontend_engineer` | Builds Gradio prototype UI                          | Claude Sonnet |

## Pipeline

```
requirements → architect → backend_engineer → code_reviewer → test_engineer → frontend_engineer
                ↓               ↓                  ↓               ↓               ↓
          design.md        module.py          review.md       test_module.py    app.py
                                                   ↓
                                         GitHub PR (auto-merge if APPROVE)
```

## Setup

```bash
pip install crewai PyGithub gitpython
export OPENAI_API_KEY=your_key
export GITHUB_TOKEN=your_token        # repo scope required
export GITHUB_REPO=username/speakflow
```

## Run

```bash
# Build Turn Analyzer module
python main.py

# Build a different module
python main.py --module coach_policy

# Run without GitHub (local only)
python main.py --skip-github
```

## Output Files

```
output/
├── turn_analyzer.py          # Generated implementation
├── turn_analyzer.py_design.md  # Architect spec
├── turn_analyzer.py_review.md  # Code review report
├── test_turn_analyzer.py     # pytest suite
└── app.py                    # Gradio prototype
```

## Adding New Modules

Edit `main.py` and add an entry to the `MODULES` dict:

```python
MODULES = {
    "turn_analyzer": { ... },
    "coach_policy": {
        "module_name": "coach_policy.py",
        "class_name": "CoachPolicyAgent",
        "requirements": "..."
    }
}
```

Then run: `python main.py --module coach_policy`
