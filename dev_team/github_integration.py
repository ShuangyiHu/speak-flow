"""
github_integration.py

Creates branches, uploads files, and opens PRs entirely via the GitHub API.
Never touches local git — no checkout, no stash, no push. Safe to run at any time.

Requirements:
    pip install PyGithub

Environment variables:
    GITHUB_TOKEN   — Personal access token with 'repo' scope
    GITHUB_REPO    — Format: "username/repo-name"
"""

import os
import re
from datetime import datetime
from pathlib import Path

try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class GitHubPRManager:
    """
    Full PR lifecycle using GitHub API only.
    Never touches local git — no checkout, no stash, no push.

    1. Create feature branch from default branch (API)
    2. Upload generated files to the branch (API)
    3. Open a Pull Request with reviewer report as description (API)
    4. Post reviewer agent output as a formal PR review (API)
    5. Auto-merge if reviewer decision is APPROVE (API)
    """

    def __init__(self, token: str, repo: str):
        if not GITHUB_AVAILABLE:
            raise ImportError("GitHub integration requires: pip install PyGithub")
        self.gh = Github(token)
        self.gh_repo = self.gh.get_repo(repo)

    def create_pr_from_output(self, module_name: str, output_dir: Path, review_text: str):
        """Full pipeline: branch → upload files → PR → review → maybe merge."""
        stem = module_name.replace(".py", "")
        branch_name = self._branch_name(stem)
        base_branch = self.gh_repo.default_branch

        print(f"\n[GitHub] Creating branch: {branch_name}")
        self._create_branch(branch_name, base_branch)

        print(f"[GitHub] Uploading files...")
        uploaded = self._upload_files(branch_name, module_name, stem, output_dir)
        if not uploaded:
            print("[GitHub] No output files found — check dev_team/output/")
            return None

        print(f"[GitHub] Opening Pull Request...")
        pr = self._open_pull_request(branch_name, base_branch, module_name, stem, review_text, uploaded)
        print(f"[GitHub] PR opened: {pr.html_url}")

        decision = self._extract_decision(review_text)
        print(f"[GitHub] Reviewer decision: {decision}")
        self._post_pr_review(pr, review_text, decision)

        if decision == "APPROVE":
            print("[GitHub] Auto-merging PR (APPROVE)...")
            try:
                pr.merge(
                    commit_title=f"feat: add {stem} [auto-merged by Dev Team]",
                    merge_method="squash",
                )
                print("[GitHub] Merged successfully.")
            except GithubException as e:
                print(f"[GitHub] Auto-merge failed (merge manually on GitHub): {e}")
        else:
            print("[GitHub] PR left open — REQUEST_CHANGES requires human review.")

        return pr

    # ── Private helpers ───────────────────────────────────────────────────────

    def _branch_name(self, stem: str) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        return f"feat/{stem.replace('_', '-')}-{ts}"

    def _create_branch(self, branch_name: str, base_branch: str):
        base_sha = self.gh_repo.get_branch(base_branch).commit.sha
        self.gh_repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)

    def _upload_files(self, branch_name: str, module_name: str, stem: str, output_dir: Path) -> list:
        """Upload each generated file to its destination path in the repo via API."""
        uploaded = []
        mappings = {
            output_dir / module_name:                   f"src/{module_name}",
            output_dir / f"test_{module_name}":         f"tests/test_{module_name}",
            output_dir / "app.py":                      "app/app.py",
            output_dir / f"{module_name}_design.md":    f"docs/{stem}_design.md",
            output_dir / f"{module_name}_review.md":    f"docs/{stem}_review.md",
        }

        for local_path, repo_path in mappings.items():
            if not local_path.exists():
                print(f"  [GitHub] Skipping (not found): {local_path.name}")
                continue

            content = local_path.read_text(encoding="utf-8")
            commit_msg = f"feat({stem}): add {local_path.name} [Dev Team]"

            try:
                existing = self.gh_repo.get_contents(repo_path, ref=branch_name)
                self.gh_repo.update_file(repo_path, commit_msg, content, existing.sha, branch=branch_name)
                print(f"  [GitHub] Updated: {repo_path}")
            except GithubException:
                self.gh_repo.create_file(repo_path, commit_msg, content, branch=branch_name)
                print(f"  [GitHub] Created: {repo_path}")

            uploaded.append(repo_path)

        return uploaded

    def _open_pull_request(self, branch_name, base_branch, module_name, stem, review_text, uploaded):
        decision = self._extract_decision(review_text)
        status_emoji = "✅" if decision == "APPROVE" else "🔄"
        files_table = "\n".join(f"| `{p}` | generated by Dev Team |" for p in uploaded)

        body = f"""## {status_emoji} Dev Team PR: `{module_name}`

**Generated by:** SpeakFlow AI Dev Team (CrewAI multi-agent pipeline)
**Reviewer decision:** `{decision}`

---

### Files
| Path | Note |
|------|------|
{files_table}

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
        event = "APPROVE" if decision == "APPROVE" else "REQUEST_CHANGES"
        try:
            pr.create_review(
                body=f"**Reviewer Agent Decision: {decision}**\n\n{review_text}",
                event=event,
            )
            print(f"[GitHub] Posted PR review: {event}")
        except GithubException as e:
            print(f"[GitHub] Could not post formal review, posting as comment: {e}")
            pr.create_issue_comment(f"**Reviewer Agent Output:**\n\n{review_text}")

    def _extract_decision(self, review_text: str) -> str:
        if not review_text:
            return "REQUEST_CHANGES"
        match = re.search(r"DECISION:\s*(APPROVE|REQUEST_CHANGES)", review_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        if "APPROVE" in review_text.upper() and "REQUEST_CHANGES" not in review_text.upper():
            return "APPROVE"
        return "REQUEST_CHANGES"