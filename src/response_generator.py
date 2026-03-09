import asyncio
import json
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from anthropic import AsyncAnthropic
import anthropic

from shared_types import (
    CoachingStrategy,
    CoachingAction,
    ToneMode,
    ResponseRequest,
    GeneratedResponse,
)


STRATEGY_TONE_MAPPING: Dict[CoachingStrategy, ToneMode] = {
    CoachingStrategy.PROBE:                 ToneMode.SOCRATIC,
    CoachingStrategy.CHALLENGE:             ToneMode.CHALLENGING,
    CoachingStrategy.REDIRECT:              ToneMode.SOCRATIC,
    CoachingStrategy.PRAISE:                ToneMode.AFFIRMING,
    CoachingStrategy.CORRECT_PRONUNCIATION: ToneMode.AFFIRMING,
}

FALLBACK_RESPONSES: Dict[CoachingStrategy, str] = {
    CoachingStrategy.PROBE:                 "Good point about the risks! Try expanding on this: you could mention how teenagers specifically are affected  -  for example, how false health advice spreads on TikTok.",
    CoachingStrategy.CHALLENGE:             "Strong argument! Take it further: try talking about a specific group of people most harmed  -  for example, elderly users who can't verify sources.",
    CoachingStrategy.REDIRECT:              "Good start! Focus your next point: pick ONE specific way social media causes harm and explain who it affects  -  for example, mental health in young people.",
    CoachingStrategy.PRAISE:                "Well done! Build on this: you could add a specific real-world example  -  for example, the spread of COVID misinformation in 2020.",
    CoachingStrategy.CORRECT_PRONUNCIATION: "Good effort  -  keep going with your argument.",
}

MAX_RESPONSE_WORDS = 50
TARGET_SENTENCES = 2
ESTIMATED_WORDS_PER_SECOND = 2.5

DEFAULT_MODEL = "claude-sonnet-4-5"
API_TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_TIME_SECONDS = 25.0


class ResponseGenerator:
    def __init__(self):
        """Initialize ResponseGenerator with Anthropic client.
        
        - Sets up AsyncAnthropic client with timeout from environment
        - Configures model from ANTHROPIC_MODEL env var or default
        """
        self._client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=API_TIMEOUT_SECONDS
        )
        self._model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    async def generate_response(self, request: ResponseRequest) -> GeneratedResponse:
        """Generate a natural debate partner response from coaching action.
        
        Args:
            request: ResponseRequest containing coaching action and context
            
        Returns:
            GeneratedResponse with natural language text, tone, and metadata
            
        Raises:
            ValueError: If request is invalid
            TimeoutError: If generation exceeds 2 seconds
        """
        if not request.coaching_action:
            raise ValueError("Request must include coaching_action")
        
        if not request.topic or not request.topic.strip():
            raise ValueError("Request must include non-empty topic")
        
        if not request.user_position or not request.user_position.strip():
            raise ValueError("Request must include non-empty user_position")
        
        strategy = request.coaching_action.strategy
        tone = STRATEGY_TONE_MAPPING.get(strategy, ToneMode.SOCRATIC)
        
        try:
            async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS):
                prompt = self._build_generation_prompt(request, tone)
                raw_response = await self._call_anthropic_api(prompt)
                parsed = self._parse_claude_response(raw_response)
                text = parsed.get("text", "")
                
                # If repetitive or empty, retry once with explicit variation instruction
                if not text or self._check_repetition(text, request.prior_responses):
                    varied_prompt = prompt + (
                        "\n\nIMPORTANT: Your previous response was too similar to a prior one. "
                        "Generate a DIFFERENT response that covers a different aspect."
                    )
                    raw_response = await self._call_anthropic_api(varied_prompt)
                    parsed = self._parse_claude_response(raw_response)
                    text = parsed.get("text", "")
                
                if not text:
                    return self._create_fallback_response(strategy, tone)

                follow_up = parsed.get("follow_up_prompt")
                now = datetime.now()
                return GeneratedResponse(
                    text=text,
                    strategy_used=strategy,
                    tone=tone,
                    follow_up_prompt=follow_up,
                    estimated_speaking_seconds=self._estimate_speaking_time(text),
                    timestamp=now,
                    latency_ms=0,
                )
        except asyncio.TimeoutError:
            return self._create_fallback_response(strategy, tone)
        except (anthropic.APIError, anthropic.APITimeoutError) as e:
            return self._create_fallback_response(strategy, tone)

    async def generate_improved_version(
        self,
        original_transcript: str,
        topic: str,
        user_position: str,
        vocabulary_flags: list,
    ) -> str:
        """
        Generate a lightly improved version of the student's own words.
        - Fixes grammar errors
        - Replaces informal/weak vocabulary with more academic alternatives
        - Adds a brief evidence phrase if missing
        - Keeps the same argument structure and core ideas (no full rewrite)
        - Written for Chinese L2 English learners

        Returns improved transcript as plain text, or fallback on error.
        """
        flags_str = ", ".join(vocabulary_flags) if vocabulary_flags else "none"

        prompt = f"""You are a language coach helping a Chinese student improve their English debate skills.

The student said:
\"\"\"{original_transcript}\"\"\"

Topic: {topic}
Their position: {user_position}
Vocabulary issues flagged: {flags_str}

Your task: Write a LIGHTLY IMPROVED VERSION of exactly what the student said.

STRICT RULES  -  you must follow all of these:
- Keep EVERY idea, point, and example the student mentioned  -  do not add any new ones
- If the student gave no example, do NOT add an example
- If the student gave no statistics, do NOT add statistics
- Fix grammar errors and unnatural phrasing only
- Replace informal or unclear vocabulary with more precise academic alternatives where flagged
- Improve sentence flow and connectives (e.g. "because", "therefore", "however")
- Keep the same length and structure as the original
- Do NOT add new arguments, new evidence, or new ideas of any kind
- Output ONLY the improved version text, no preamble or explanation"""

        try:
            async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS * 3):
                raw = await self._call_anthropic_api(prompt)
                return raw.strip()
        except Exception:
            return ""  # caller will hide the box if empty

    async def generate_language_tips(
        self,
        original_transcript: str,
        improved_transcript: str,
        vocabulary_flags: list,
        clarity_feedback: str,
        fluency_feedback: str,
    ) -> str:
        """One-sentence language tip on the single most important grammar/vocabulary issue."""
        flags_str = ", ".join(vocabulary_flags) if vocabulary_flags else "none"

        prompt = (
            f'You are an English language teacher giving brief feedback to a Chinese student.\n\n'
            f'The student said:\n"{original_transcript}"\n\n'
            f'Vocabulary issues flagged: {flags_str}\n\n'
            'Write ONE sentence highlighting the single most important language issue: grammar, word choice, or sentence structure.\n'
            'Do NOT comment on the argument or reasoning.\n\n'
            'Good examples:\n'
            '"Watch your tense: use past tense \"were panicking\" instead of \"are panicking\" to match the past context."\n'
            '"Replace informal \"very bad\" with a more precise phrase like \"severely damaging\" or \"deeply harmful\"."\n'
            '"Add a connector before your conclusion: \"Therefore, ...\" or \"This shows that ...\" makes your point flow better."\n\n'
            'Output ONE sentence only, no preamble.'
        )

        try:
            async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS * 3):
                raw = await self._call_anthropic_api(prompt)
                return raw.strip()
        except Exception:
            return ""

    async def generate_session_debate_summary(
        self,
        topic: str,
        user_position: str,
        transcripts: list,
        argument_scores: list,
    ) -> str:
        """
        End-of-session debate summary: overall argument quality, strengths, one improvement tip.
        Focused on debate content — not language.
        """
        turns_text = "\n".join(
            f"Turn {i+1} (score {argument_scores[i]:.2f}): {t}"
            for i, t in enumerate(transcripts)
        )
        avg = sum(argument_scores) / len(argument_scores) if argument_scores else 0

        prompt = (
            f'You are a debate coach giving end-of-session feedback to a Chinese English learner.\n\n'
            f'Topic: "{topic}" | Student position: {user_position}\n\n'
            f'What the student said across {len(transcripts)} turns:\n{turns_text}\n\n'
            f'Average argument score: {avg:.2f}/1.0\n\n'
            'Write a 3-4 sentence session summary:\n'
            '1. One specific thing they did well across the session (name a concrete moment)\n'
            '2. One key area to improve in their argument structure or reasoning\n'
            '3. An encouraging closing remark\n\n'
            'Be specific, warm, and direct. Do NOT comment on language or grammar.\n'
            'Output plain text only, no JSON, no bullets.'
        )

        try:
            async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS):
                message = await self._client.messages.create(
                    model=self._model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text.strip()
        except Exception:
            return "Well done on completing this session! Your arguments showed clear thinking and good effort."

    async def generate_session_language_summary(
        self,
        transcripts: list,
        per_turn_tips: list,
    ) -> str:
        """
        End-of-session language summary: patterns across all turns, top 2 language habits to fix.
        Focused purely on language — not debate content.
        """
        turns_text = "\n".join(
            f"Turn {i+1}: {t}" for i, t in enumerate(transcripts)
        )
        tips_text = "\n".join(
            f"Turn {i+1} tip: {t}" for i, t in enumerate(per_turn_tips) if t
        )

        prompt = (
            'You are an English language teacher summarising patterns across a student\'s speaking session.\n\n'
            f'What the student said:\n{turns_text}\n\n'
            f'Per-turn language notes:\n{tips_text if tips_text else "none recorded"}\n\n'
            'Write 2 sentences:\n'
            '1. Name ONE recurring language pattern or habit that appeared across multiple turns '
            '(grammar, vocabulary, connectors, sentence structure)\n'
            '2. Give ONE concrete, specific thing to practise before the next session '
            '(e.g. a phrase template, a grammar rule, a vocabulary range)\n\n'
            'Focus purely on HOW they spoke, not what they argued.\n'
            'Output plain text only, no JSON, no bullets.'
        )

        try:
            async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS):
                message = await self._client.messages.create(
                    model=self._model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text.strip()
        except Exception:
            return "Focus on using connectors like 'therefore' and 'however' to link your ideas more clearly."

    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API to generate response text.
        
        Args:
            prompt: Formatted prompt for Claude
            
        Returns:
            Raw response text from Claude
            
        Raises:
            Exception: Any API-related errors
        """
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _build_generation_prompt(self, request: ResponseRequest, tone: ToneMode) -> str:
        """Build prompt for Claude based on request and tone."""
        prior_text = " | ".join(request.prior_responses[-3:]) if request.prior_responses else "None"
        intent = request.coaching_action.intent or ""
        strategy_key = request.coaching_action.strategy.value
        turn = request.turn_number

        # After turn 3, do a wrap-up  -  don't keep probing
        if turn >= 3:
            return (
                f'You are an English debate coach giving a warm round wrap-up.\n\n'
                f'Context:\n{intent}\n\n'
                f'The student has completed {turn} turns. '
                f'Write 2 sentences:\n'
                f'1. Name ONE specific thing they did well (reference their actual words)\n'
                f'2. Give ONE language or argument tip for next time\n\n'
                f'Under 50 words. Warm tone. '
                f'Do NOT ask "would you like to continue" - that is handled by buttons.\n'
                f'Do NOT repeat: "{prior_text}"\n\n'
                f'Output JSON only: {{"text": "your response"}}'
            )

        strategy_instructions = {
            "PROBE": (
                "Your ONLY job: help the student know what to say NEXT.\n"
                "Step 1 (1 sentence): Briefly acknowledge what they said  -  name their specific point.\n"
                "Step 2 (1-2 sentences): Give them a CONCRETE NEXT TALKING POINT to develop, like:\n"
                "  'You could talk about [specific scenario or group]...'\n"
                "  'For example, think about what happens when [specific situation]...'\n"
                "  'A strong next point would be [specific direction]...'\n"
                "BANNED: Any version of 'can you tell me more', 'specifically', 'elaborate', 'expand on that'.\n"
                "The student must be able to START SPEAKING immediately after reading your response."
            ),
            "REDIRECT": (
                "The student's response lacked a clear argument. Be warm.\n"
                "Step 1: Acknowledge the one thing they did manage to say.\n"
                "Step 2: Give them ONE concrete talking point to try  -  name an exact direction or example."
            ),
            "PRAISE": (
                "Step 1: Name the SPECIFIC strength  -  quote or paraphrase their actual words.\n"
                "Step 2: Give them a concrete NEXT talking point to go one level deeper.\n"
                "Do NOT ask an open-ended question. Give them direction."
            ),
            "CHALLENGE": (
                "The student gave a strong argument. Give them a DEEPER angle on their OWN side.\n"
                "Suggest a specific scenario, group, or real-world example they could address next."
            ),
            "CORRECT_PRONUNCIATION": (
                "Naturally use the target word in a sentence. Keep it brief and conversational."
            ),
        }
        instruction = strategy_instructions.get(strategy_key, strategy_instructions["PROBE"])

        return f"""You are an English debate coach having a real conversation with a student.

Context:
{intent}

Your task ({strategy_key}): {instruction}

Rules:
- 2-3 sentences maximum, under 65 words total
- Be specific to THIS student's exact words  -  never be generic
- Always include a concrete talking point or example hint (except CORRECT_PRONUNCIATION)
- Do NOT repeat or rephrase: "{prior_text}"
- Sound like a real coach, warm and direct

Output JSON only: {{"text": "your response"}}"""

    def _parse_claude_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse Claude's JSON response, handling markdown fences.
        
        Args:
            raw_response: Raw text from Claude API
            
        Returns:
            Parsed JSON dict
            
        Raises:
            json.JSONDecodeError: If response isn't valid JSON
        """
        cleaned = raw_response.strip()
        
        # Remove outer markdown fences with improved pattern
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Handle nested backticks by finding the JSON boundaries
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            cleaned = cleaned[json_start:json_end + 1]
        
        return json.loads(cleaned.strip())

    def _create_fallback_response(self, strategy: CoachingStrategy, tone: ToneMode) -> GeneratedResponse:
        """Create fallback response when API fails.
        
        Args:
            strategy: Coaching strategy for fallback selection
            tone: Tone mode for the response
            
        Returns:
            GeneratedResponse using predefined fallback text
        """
        text = FALLBACK_RESPONSES.get(strategy, "What are your thoughts on that?")
        return GeneratedResponse(
            text=text,
            strategy_used=strategy,
            tone=tone,
            follow_up_prompt=None,
            estimated_speaking_seconds=self._estimate_speaking_time(text),
            timestamp=datetime.now(),
            latency_ms=0,
        )

    def _estimate_speaking_time(self, text: str) -> float:
        """Estimate speaking duration for response text.
        
        Args:
            text: Response text to analyze
            
        Returns:
            Estimated speaking time in seconds
        """
        word_count = len(text.split())
        return word_count / ESTIMATED_WORDS_PER_SECOND

    def _check_repetition(self, new_text: str, prior_responses: List[str]) -> bool:
        """Check if new response is too similar to recent responses.
        
        Args:
            new_text: Candidate response text
            prior_responses: Last 3 response texts
            
        Returns:
            True if response is repetitive, False otherwise
        """
        if not prior_responses:
            return False
            
        new_words = set(new_text.lower().split())
        if not new_words:
            return True
            
        for prior in prior_responses:
            prior_words = set(prior.lower().split())
            if not prior_words:
                continue
                
            overlap = len(new_words & prior_words)
            total = len(new_words | prior_words)
            if total > 0 and overlap / total > 0.7:
                return True
        return False