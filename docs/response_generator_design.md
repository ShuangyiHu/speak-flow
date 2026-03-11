# ResponseGenerator Module Technical Design

## Overview

The `response_generator.py` module decouples natural language generation from coaching policy decisions. It transforms structured coaching actions into natural, varied debate partner responses.

## External Dependencies

```python
import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from anthropic import AsyncAnthropic
```

## Data Models

### Enums

```python
class CoachingStrategy(Enum):
    CLARIFY_POSITION = "clarify_position"
    STRENGTHEN_ARGUMENT = "strengthen_argument"
    COUNTER_ARGUMENT = "counter_argument"
    EXPLORE_IMPLICATIONS = "explore_implications"
    FIND_EVIDENCE = "find_evidence"
    ACKNOWLEDGE_GOOD_POINT = "acknowledge_good_point"

class ToneMode(Enum):
    SOCRATIC = "socratic"       # Questioning approach
    CHALLENGING = "challenging" # Pushback approach
    AFFIRMING = "affirming"     # Praise approach
```

### Request/Response Models

```python
@dataclass
class CoachingAction:
    strategy: CoachingStrategy
    target_skill: str
    confidence_score: float  # 0.0 to 1.0

@dataclass
class ResponseRequest:
    coaching_action: CoachingAction
    topic: str
    user_position: str
    prior_responses: List[str]  # Last 3 response texts for repetition avoidance
    turn_number: int

@dataclass
class GeneratedResponse:
    text: str
    tone: ToneMode
    follow_up_prompt: Optional[str]
    estimated_speaking_seconds: float
```

## Configuration Constants

```python
# Strategy to tone mapping
STRATEGY_TONE_MAPPING: Dict[CoachingStrategy, ToneMode] = {
    CoachingStrategy.CLARIFY_POSITION: ToneMode.SOCRATIC,
    CoachingStrategy.STRENGTHEN_ARGUMENT: ToneMode.SOCRATIC,
    CoachingStrategy.COUNTER_ARGUMENT: ToneMode.CHALLENGING,
    CoachingStrategy.EXPLORE_IMPLICATIONS: ToneMode.SOCRATIC,
    CoachingStrategy.FIND_EVIDENCE: ToneMode.SOCRATIC,
    CoachingStrategy.ACKNOWLEDGE_GOOD_POINT: ToneMode.AFFIRMING,
}

# Fallback responses for each strategy (no API required)
FALLBACK_RESPONSES: Dict[CoachingStrategy, str] = {
    CoachingStrategy.CLARIFY_POSITION: "Can you clarify what you mean by that?",
    CoachingStrategy.STRENGTHEN_ARGUMENT: "What evidence supports your view?",
    CoachingStrategy.COUNTER_ARGUMENT: "I see it differently - what if we consider the opposing perspective?",
    CoachingStrategy.EXPLORE_IMPLICATIONS: "What would be the consequences of that approach?",
    CoachingStrategy.FIND_EVIDENCE: "What data or examples back up that claim?",
    CoachingStrategy.ACKNOWLEDGE_GOOD_POINT: "That's a compelling point you've made.",
}

# Response length constraints
MAX_RESPONSE_WORDS = 50
TARGET_SENTENCES = 2
ESTIMATED_WORDS_PER_SECOND = 2.5

# API configuration
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
API_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_TIME_SECONDS = 25.0
```

## Main Class

```python
class ResponseGenerator:
    def __init__(self):
        """Initialize ResponseGenerator with Anthropic client.

        - Sets up AsyncAnthropic client with timeout from environment
        - Configures model from ANTHROPIC_MODEL env var or default
        """

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

    def _map_strategy_to_tone(self, strategy: CoachingStrategy) -> ToneMode:
        """Map coaching strategy to appropriate tone mode.

        Args:
            strategy: The coaching strategy to map

        Returns:
            ToneMode corresponding to the strategy
        """

    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API to generate response text.

        Args:
            prompt: Formatted prompt for Claude

        Returns:
            Raw response text from Claude

        Raises:
            Exception: Any API-related errors
        """

    def _build_generation_prompt(self, request: ResponseRequest, tone: ToneMode) -> str:
        """Build prompt for Claude based on request and tone.

        Args:
            request: ResponseRequest with context
            tone: Target tone mode

        Returns:
            Formatted prompt string for Claude
        """

    def _parse_claude_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse Claude's JSON response, handling markdown fences.

        Args:
            raw_response: Raw text from Claude API

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If response isn't valid JSON
        """

    def _create_fallback_response(self, strategy: CoachingStrategy, tone: ToneMode) -> GeneratedResponse:
        """Create fallback response when API fails.

        Args:
            strategy: Coaching strategy for fallback selection
            tone: Tone mode for the response

        Returns:
            GeneratedResponse using predefined fallback text
        """

    def _estimate_speaking_time(self, text: str) -> float:
        """Estimate speaking duration for response text.

        Args:
            text: Response text to analyze

        Returns:
            Estimated speaking time in seconds
        """

    def _check_repetition(self, new_text: str, prior_responses: List[str]) -> bool:
        """Check if new response is too similar to recent responses.

        Args:
            new_text: Candidate response text
            prior_responses: Last 3 response texts

        Returns:
            True if response is repetitive, False otherwise
        """
```

## Implementation Details

### Anthropic API Integration

```python
# Client initialization in __init__
self._client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=API_TIMEOUT_SECONDS
)
self._model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

# API call pattern in _call_anthropic_api
message = await self._client.messages.create(
    model=self._model,
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}]
)
return message.content[0].text
```

### JSON Response Parsing

````python
# Strip markdown fences before parsing
def _parse_claude_response(self, raw_response: str) -> Dict[str, Any]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())
````

### Prompt Template Structure

```python
# Example prompt format in _build_generation_prompt
f"""You are a debate partner helping someone practice.

Topic: {request.topic}
Their position: {request.user_position}
Strategy: {request.coaching_action.strategy.value}
Tone: {tone.value}
Turn: {request.turn_number}

Generate a {tone.value} response that's 1-2 sentences, under 50 words.
Avoid repeating these recent responses: {prior_responses}

Return JSON: {{"text": "your response", "follow_up_prompt": "optional question"}}"""
```

### Error Handling Pattern

```python
# Timeout and fallback pattern in generate_response
try:
    async with asyncio.timeout(MAX_RESPONSE_TIME_SECONDS):
        # API generation attempt
        pass
except (asyncio.TimeoutError, Exception):
    # Use fallback response
    return self._create_fallback_response(strategy, tone)
```

### Repetition Detection

```python
# Simple similarity check in _check_repetition
def _check_repetition(self, new_text: str, prior_responses: List[str]) -> bool:
    new_words = set(new_text.lower().split())
    for prior in prior_responses:
        prior_words = set(prior.lower().split())
        # Check for high word overlap (>70%)
        if len(new_words & prior_words) / len(new_words | prior_words) > 0.7:
            return True
    return False
```

This design provides a complete, implementable specification for the ResponseGenerator module with all required functionality, error handling, and integration patterns clearly defined.
