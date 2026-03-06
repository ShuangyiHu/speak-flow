"""
github_integration.py

Handles automatic branch creation, file commits, PR opening,
code review posting, and conditional auto-merge for the SpeakFlow Dev Team.

Requirements:
    pip install PyGithub gitpython

Environment variables:
    GITHUB_TOKEN   — Personal access token with repo scope
    GITHUB_REPO    — Format: "username/repo-name"
"""

import os
import re
from pathlib import Path
from datetime import datetime

try:
    from github import Github, GithubException
    from git import Repo, InvalidGitRepositoryError
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class GitHubPRManager:
    """
    Manages the full PR lifecycle for a generated module:
    
    1. create_feature_branch()   — git checkout -b feature/<module>-<timestamp>
    2. commit_output_files()     — git add output/* && git commit
    3. push_branch()             — git push origin <branch>
    4. open_pull_request()       — GitHub API: create PR with review as description
    5. post_review_comment()     — GitHub API: post reviewer agent output as PR review
    6. auto_merge_if_approved()  — Merge if review decision is APPROVE
    """

    def __init__(self, token: str, repo: str):
        if not GITHUB_AVAILABLE:
            raise ImportError(
                "GitHub integration requires: pip install PyGithub gitpython"
            )
        self.gh = Github(token)
        self.gh_repo = self.gh.get_repo(repo)
        self.token = token
        self.repo_name = repo

    def create_pr_from_output(
        self,
        module_name: str,
        output_dir: Path,
        review_text: str,
    ):
        """
        Full pipeline: branch → commit → push → PR → review comment → maybe merge.
        """
        branch_name = self._branch_name(module_name)
        base_branch = self.gh_repo.default_branch

        print(f"\n[GitHub] Creating branch: {branch_name}")
        self._create_and_push_branch(branch_name, base_branch, output_dir, module_name)

        print(f"[GitHub] Opening Pull Request...")
        pr = self._open_pull_request(branch_name, base_branch, module_name, review_text)
        print(f"[GitHub] PR opened: {pr.html_url}")

        decision = self._extract_decision(review_text)
        print(f"[GitHub] Reviewer decision: {decision}")

        # Post the full review as a PR review comment
        self._post_pr_review(pr, review_text, decision)

        if decision == "APPROVE":
            print("[GitHub] Auto-merging PR (APPROVE)...")
            try:
                pr.merge(
                    commit_title=f"feat: add {module_name} [auto-merged by Dev Team]",
                    merge_method="squash",
                )
                print("[GitHub] Merged successfully.")
            except GithubException as e:
                print(f"[GitHub] Auto-merge failed (may need manual merge): {e}")
        else:
            print("[GitHub] PR left open — REQUEST_CHANGES requires human review.")

        return pr

    # ── Private helpers ───────────────────────────────────────────────────────

    def _branch_name(self, module_name: str) -> str:
        stem = module_name.replace(".py", "").replace("_", "-")
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        return f"feat/{stem}-{ts}"

    def _create_and_push_branch(
        self,
        branch_name: str,
        base_branch: str,
        output_dir: Path,
        module_name: str,
    ):
        """
        Uses local git repo (assumes script runs inside the repo).
        Creates a branch, moves generated files to src/, commits, and pushes.
        """
        try:
            local_repo = Repo(search_parent_directories=True)
        except InvalidGitRepositoryError:
            print("[GitHub] WARNING: Not inside a git repo. Skipping branch/commit step.")
            return

        # Create branch from base
        origin = local_repo.remote("origin")
        local_repo.git.fetch("origin")
        local_repo.git.checkout(base_branch)
        local_repo.git.checkout("-b", branch_name)

        # Copy output files to src/ directory
        src_dir = Path(local_repo.working_dir) / "src"
        tests_dir = Path(local_repo.working_dir) / "tests"
        app_dir = Path(local_repo.working_dir) / "app"

        src_dir.mkdir(exist_ok=True)
        tests_dir.mkdir(exist_ok=True)
        app_dir.mkdir(exist_ok=True)

        file_mappings = {
            output_dir / module_name: src_dir / module_name,
            output_dir / f"test_{module_name}": tests_dir / f"test_{module_name}",
            output_dir / "app.py": app_dir / "app.py",
            output_dir / f"{module_name}_design.md": Path(local_repo.working_dir) / "docs" / f"{module_name}_design.md",
            output_dir / f"{module_name}_review.md": Path(local_repo.working_dir) / "docs" / f"{module_name}_review.md",
        }

        committed_files = []
        for src_path, dest_path in file_mappings.items():
            if src_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(src_path.read_text())
                local_repo.index.add([str(dest_path.relative_to(local_repo.working_dir))])
                committed_files.append(dest_path.name)

        if committed_files:
            local_repo.index.commit(
                f"feat({module_name.replace('.py','')}): add generated module, tests, and design docs\n\n"
                f"Generated by SpeakFlow Dev Team (CrewAI)\n"
                f"Files: {', '.join(committed_files)}"
            )
            # Push with token auth
            push_url = f"https://{self.token}@github.com/{self.repo_name}.git"
            local_repo.git.push(push_url, branch_name)
            print(f"[GitHub] Pushed {len(committed_files)} files to {branch_name}")
        else:
            print("[GitHub] No output files found to commit.")

    def _open_pull_request(
        self,
        branch_name: str,
        base_branch: str,
        module_name: str,
        review_text: str,
    ):
        stem = module_name.replace(".py", "")
        decision = self._extract_decision(review_text)
        status_emoji = "✅" if decision == "APPROVE" else "🔄"

        body = f"""## {status_emoji} Dev Team PR: `{module_name}`

**Generated by:** SpeakFlow AI Dev Team (CrewAI multi-agent pipeline)  
**Reviewer decision:** `{decision}`

---

### Files Changed
| File | Description |
|------|-------------|
| `src/{module_name}` | Backend implementation |
| `tests/test_{module_name}` | pytest test suite |
| `app/app.py` | Gradio prototype UI |
| `docs/{stem}_design.md` | Architect design spec |
| `docs/{stem}_review.md` | Code reviewer report |

---

### Reviewer Agent Report

{review_text[:3000] if review_text else '_No review output found._'}

---
_This PR was automatically created by the SpeakFlow Dev Team agent pipeline._
"""
        return self.gh_repo.create_pull(
            title=f"feat: add {stem} module [Dev Team]",
            body=body,
            head=branch_name,
            base=base_branch,
        )

    def _post_pr_review(self, pr, review_text: str, decision: str):
        """Post the reviewer agent output as a formal GitHub PR review."""
        event = "APPROVE" if decision == "APPROVE" else "REQUEST_CHANGES"
        try:
            pr.create_review(
                body=f"**Reviewer Agent Decision: {decision}**\n\n{review_text}",
                event=event,
            )
            print(f"[GitHub] Posted PR review with event: {event}")
        except GithubException as e:
            print(f"[GitHub] Could not post formal review (posting as comment instead): {e}")
            pr.create_issue_comment(f"**Reviewer Agent Output:**\n\n{review_text}")

    def _extract_decision(self, review_text: str) -> str:
        """Parse APPROVE or REQUEST_CHANGES from reviewer output."""
        if not review_text:
            return "REQUEST_CHANGES"
        match = re.search(r"DECISION:\s*(APPROVE|REQUEST_CHANGES)", review_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback: look for the words anywhere
        if "APPROVE" in review_text.upper() and "REQUEST_CHANGES" not in review_text.upper():
            return "APPROVE"
        return "REQUEST_CHANGES"
