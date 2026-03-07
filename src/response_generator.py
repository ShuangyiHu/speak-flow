import asyncio
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from anthropic import AsyncAnthropic
import anthropic


class CoachingStrategy(Enum):
    CLARIFY_POSITION = "clarify_position"
    STRENGTHEN_ARGUMENT = "strengthen_argument"
    COUNTER_ARGUMENT = "counter_argument"
    EXPLORE_IMPLICATIONS = "explore_implications"
    FIND_EVIDENCE = "find_evidence"
    ACKNOWLEDGE_GOOD_POINT = "acknowledge_good_point"


class ToneMode(Enum):
    SOCRATIC = "socratic"
    CHALLENGING = "challenging"
    AFFIRMING = "affirming"


@dataclass
class CoachingAction:
    strategy: CoachingStrategy
    target_skill: str
    confidence_score: float


@dataclass
class ResponseRequest:
    coaching_action: CoachingAction
    topic: str
    user_position: str
    prior_responses: List[str]
    turn_number: int


@dataclass
class GeneratedResponse:
    text: str
    tone: ToneMode
    follow_up_prompt: Optional[str]
    estimated_speaking_seconds: float


STRATEGY_TONE_MAPPING: Dict[CoachingStrategy, ToneMode] = {
    CoachingStrategy.CLARIFY_POSITION: ToneMode.SOCRATIC,
    CoachingStrategy.STRENGTHEN_ARGUMENT: ToneMode.SOCRATIC,
    CoachingStrategy.COUNTER_ARGUMENT: ToneMode.CHALLENGING,
    CoachingStrategy.EXPLORE_IMPLICATIONS: ToneMode.SOCRATIC,
    CoachingStrategy.FIND_EVIDENCE: ToneMode.SOCRATIC,
    CoachingStrategy.ACKNOWLEDGE_GOOD_POINT: ToneMode.AFFIRMING,
}

FALLBACK_RESPONSES: Dict[CoachingStrategy, str] = {
    CoachingStrategy.CLARIFY_POSITION: "Can you clarify what you mean by that?",
    CoachingStrategy.STRENGTHEN_ARGUMENT: "What evidence supports your view?",
    CoachingStrategy.COUNTER_ARGUMENT: "I see it differently - what if we consider the opposing perspective?",
    CoachingStrategy.EXPLORE_IMPLICATIONS: "What would be the consequences of that approach?",
    CoachingStrategy.FIND_EVIDENCE: "What data or examples back up that claim?",
    CoachingStrategy.ACKNOWLEDGE_GOOD_POINT: "That's a compelling point you've made.",
}

MAX_RESPONSE_WORDS = 50
TARGET_SENTENCES = 2
ESTIMATED_WORDS_PER_SECOND = 2.5

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
API_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_TIME_SECONDS = 2.0


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
        tone = self._map_strategy_to_tone(strategy)
        
        try:
            async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS):
                prompt = self._build_generation_prompt(request, tone)
                raw_response = await self._call_anthropic_api(prompt)
                parsed = self._parse_claude_response(raw_response)
                
                text = parsed.get("text", "")
                follow_up = parsed.get("follow_up_prompt")
                
                if not text or self._check_repetition(text, request.prior_responses):
                    return self._create_fallback_response(strategy, tone)
                
                return GeneratedResponse(
                    text=text,
                    tone=tone,
                    follow_up_prompt=follow_up,
                    estimated_speaking_seconds=self._estimate_speaking_time(text)
                )
        except asyncio.TimeoutError:
            return self._create_fallback_response(strategy, tone)
        except (anthropic.APIError, anthropic.APITimeoutError) as e:
            return self._create_fallback_response(strategy, tone)

    def _map_strategy_to_tone(self, strategy: CoachingStrategy) -> ToneMode:
        """Map coaching strategy to appropriate tone mode.
        
        Args:
            strategy: The coaching strategy to map
            
        Returns:
            ToneMode corresponding to the strategy
        """
        return STRATEGY_TONE_MAPPING.get(strategy, ToneMode.SOCRATIC)

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
        """Build prompt for Claude based on request and tone.
        
        Args:
            request: ResponseRequest with context
            tone: Target tone mode
            
        Returns:
            Formatted prompt string for Claude
        """
        prior_text = ", ".join(request.prior_responses) if request.prior_responses else "None"
        
        return f"""You are a debate partner helping someone practice. 

Topic: {request.topic}
Their position: {request.user_position}
Strategy: {request.coaching_action.strategy.value}
Tone: {tone.value}
Turn: {request.turn_number}

Generate a {tone.value} response that's 1-2 sentences, under 50 words.
Avoid repeating these recent responses: {prior_text}

Return JSON: {{"text": "your response", "follow_up_prompt": "optional question"}}"""

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
            tone=tone,
            follow_up_prompt=None,
            estimated_speaking_seconds=self._estimate_speaking_time(text)
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