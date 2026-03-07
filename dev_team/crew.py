from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class SpeakFlowDevTeam():
    """
    SpeakFlow AI — Multi-Agent Development Team

    Main pipeline (sequential):
        design_task → code_task → review_task → test_task → frontend_task

    Revision loop (called from main.py if review is REQUEST_CHANGES):
        revision_crew() → fix → re-review
        Repeats up to MAX_REVISIONS times until APPROVE.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # ── Agents ────────────────────────────────────────────────────────────────

    @agent
    def architect(self) -> Agent:
        return Agent(config=self.agents_config['architect'], verbose=True)

    @agent
    def backend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['backend_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3,
        )

    @agent
    def revision_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['revision_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3,
        )

    @agent
    def code_reviewer(self) -> Agent:
        return Agent(config=self.agents_config['code_reviewer'], verbose=True)

    @agent
    def test_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['test_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3,
        )

    @agent
    def frontend_engineer(self) -> Agent:
        return Agent(config=self.agents_config['frontend_engineer'], verbose=True)

    # ── Tasks (main pipeline only — no revision tasks here) ───────────────────

    @task
    def design_task(self) -> Task:
        return Task(config=self.tasks_config['design_task'])

    @task
    def code_task(self) -> Task:
        return Task(config=self.tasks_config['code_task'])

    @task
    def review_task(self) -> Task:
        return Task(config=self.tasks_config['review_task'])

    @task
    def test_task(self) -> Task:
        return Task(config=self.tasks_config['test_task'])

    @task
    def frontend_task(self) -> Task:
        return Task(config=self.tasks_config['frontend_task'])

    # ── Crews ─────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Main pipeline: design → code → review → test → frontend."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def revision_crew(self, module_name: str, review_feedback: str, current_code: str) -> Crew:
        """
        Revision loop: fix issues → re-review.
        Built with explicit Task objects (not @task) so CrewBase doesn't scan them.
        Call kickoff() directly on the returned Crew — no extra inputs needed.
        """
        revision_task = Task(
            description=(
                f"The code reviewer has requested changes to {module_name}.\n\n"
                f"REVIEWER FEEDBACK:\n{review_feedback}\n\n"
                f"CURRENT IMPLEMENTATION:\n{current_code}\n\n"
                "Fix ALL issues marked [CRITICAL] first, then [MINOR].\n"
                "Do not change any behavior that was already correct.\n"
                "Output ONLY raw Python code. No markdown. No backticks."
            ),
            expected_output=f"Corrected Python code for {module_name}. No markdown. Valid Python only.",
            agent=self.revision_engineer(),
            output_file=f"output/{module_name}",
        )

        re_review_task = Task(
            description=(
                f"Re-review the revised implementation of {module_name}.\n\n"
                f"PREVIOUS FEEDBACK:\n{review_feedback}\n\n"
                "Check that ALL previously raised issues have been fixed.\n"
                "Also check for any new issues introduced by the revision.\n\n"
                "Output format:\n"
                "DECISION: APPROVE or REQUEST_CHANGES\n\n"
                "ISSUES (if any):\n"
                "- [CRITICAL/MINOR] Description and suggested fix\n\n"
                "SUMMARY: One paragraph assessment."
            ),
            expected_output="Structured code review with APPROVE or REQUEST_CHANGES decision.",
            agent=self.code_reviewer(),
            context=[revision_task],
            output_file=f"output/{module_name}_review.md",
        )

        return Crew(
            agents=[self.revision_engineer(), self.code_reviewer()],
            tasks=[revision_task, re_review_task],
            process=Process.sequential,
            tracing=True,
            verbose=True,
        )