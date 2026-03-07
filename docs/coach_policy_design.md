# Technical Design: coach_policy.py

## Overview

This module implements the CoachPolicyAgent for SpeakFlow AI's debate coaching platform. It receives TurnAnalysis objects and decides coaching actions using rule-based strategy selection and Claude AI for response generation.

## External Dependencies

```python
from anthropic import AsyncAnthropic
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import asyncio
import re
```

## Data Models

### CoachingStrategy Enum

```python
from enum import Enum

class CoachingStrategy(Enum):
    PROBE = "probe"
    CHALLENGE = "challenge"
    REDIRECT = "redirect"
    PRAISE_AND_PUSH = "praise_and_push"
    CORRECT_PRONUNCIATION = "correct_pronunciation"
```

### CoachingAction Dataclass

```python
@dataclass
class CoachingAction:
    strategy: CoachingStrategy
    response_text: str
    internal_reason: str
    target_word: str
    difficulty_delta: int
```

### SessionContext Dataclass

```python
@dataclass
class SessionContext:
    session_id: str
    topic: str
    user_position: str  # "for" or "against"
    turn_number: int
    coaching_history: List[CoachingStrategy]
    argument_scores: List[float]
```

## Constants

```python
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
API_TIMEOUT_SECONDS = 30
DECIDE_TIMEOUT_SECONDS = 10
LOW_ARGUMENT_THRESHOLD = 0.1
HIGH_ARGUMENT_THRESHOLD = 0.7
DIFFICULTY_THRESHOLD = 0.3
REPEAT_PREVENTION_LOOKBACK = 2
MAX_RESPONSE_LENGTH = 500
```

## Main Class

### CoachPolicyAgent

```python
class CoachPolicyAgent:
    def __init__(self, anthropic_api_key: str) -> None:
        """
        Initialize the coach policy agent with Anthropic client.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude access
        """
        
    async def decide(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingAction:
        """
        Main entry point to decide coaching action based on turn analysis.
        ASYNC METHOD - calls Claude API
        
        Args:
            analysis: TurnAnalysis object from TurnAnalyzer
            context: Current session context and history
            
        Returns:
            CoachingAction with strategy, response, and metadata
            
        Behavior:
            - Completes within 10 seconds total
            - Returns default action on any error (no exceptions raised)
            - Handles empty transcript by returning PROBE with opening question
        """
        
    def _select_strategy(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingStrategy:
        """
        Select coaching strategy using rule-based logic.
        SYNCHRONOUS METHOD - no LLM calls, pure deterministic logic
        
        Args:
            analysis: TurnAnalysis object
            context: SessionContext with history
            
        Returns:
            CoachingStrategy enum value
            
        Rules (in priority order):
            1. If mispronounced_words has severity==MAJOR: CORRECT_PRONUNCIATION
            2. If transcript empty or argument_score < 0.1: REDIRECT  
            3. If last 2 coaching_history entries same: force different strategy
            4. If argument_score >= 0.7 and turn_number > 2: CHALLENGE
            5. If argument_score >= 0.7: PRAISE_AND_PUSH
            6. If has_claim=True but has_reasoning=False: PROBE
            7. If has_reasoning=True but has_evidence=False: PROBE
            8. Default: PROBE
        """
        
    async def _generate_response(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str:
        """
        Generate response text using Claude API.
        ASYNC METHOD - makes HTTP call to Anthropic
        
        Args:
            analysis: TurnAnalysis with student's input
            context: Session context for personalization
            strategy: Selected coaching strategy
            
        Returns:
            Response text (1-3 sentences, markdown stripped)
            
        Behavior:
            - Calls AsyncAnthropic.messages.create()
            - Uses claude-3-5-sonnet model
            - 30 second timeout on API call
            - Strips markdown from response
            - Returns conversational, partner-style response
        """
        
    def _build_prompt(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str:
        """
        Build prompt string for Claude API call.
        SYNCHRONOUS METHOD - string construction only
        
        Args:
            analysis: TurnAnalysis object
            context: SessionContext for history
            strategy: Selected strategy for prompt customization
            
        Returns:
            Complete prompt string for Claude
            
        Content:
            - Student transcript
            - Strategy-specific instructions
            - Debate topic and position
            - Last 3 turns of coaching history
            - Response format requirements (1-3 sentences, no bullet points)
        """
        
    def _create_default_action(self, context: SessionContext) -> CoachingAction:
        """
        Create fallback CoachingAction for error cases.
        SYNCHRONOUS METHOD - no external calls
        
        Args:
            context: SessionContext for topic-relevant question
            
        Returns:
            CoachingAction with strategy=PROBE and generic opening question
        """
        
    def _calculate_difficulty_delta(self, context: SessionContext) -> int:
        """
        Calculate difficulty adjustment based on recent performance.
        SYNCHRONOUS METHOD - pure calculation
        
        Args:
            context: SessionContext with argument_scores history
            
        Returns:
            -1 (easier), 0 (same), or +1 (harder)
            
        Logic:
            +1 if last 2 argument_scores >= 0.7
            -1 if last 2 argument_scores < 0.3
             0 otherwise
        """
        
    def _get_target_word(self, analysis: TurnAnalysis) -> str:
        """
        Extract target word for pronunciation correction.
        SYNCHRONOUS METHOD - data extraction only
        
        Args:
            analysis: TurnAnalysis with pronunciation data
            
        Returns:
            First mispronounced word with MAJOR severity, or "" if none
        """
        
    def _strip_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from response text.
        SYNCHRONOUS METHOD - regex processing
        
        Args:
            text: Raw response from Claude
            
        Returns:
            Clean text with markdown removed
        """
        
    def _avoid_repetition(self, preferred_strategy: CoachingStrategy, context: SessionContext) -> CoachingStrategy:
        """
        Check if strategy was used in last 2 turns and return alternative.
        SYNCHRONOUS METHOD - list processing
        
        Args:
            preferred_strategy: Strategy selected by main rules
            context: SessionContext with coaching_history
            
        Returns:
            Original strategy or alternative if repetition detected
        """
```

## Anthropic SDK Integration Pattern

```python
# In __init__:
self._client = AsyncAnthropic(api_key=anthropic_api_key)

# In _generate_response:
response = await self._client.messages.create(
    model=CLAUDE_MODEL,
    max_tokens=MAX_RESPONSE_LENGTH,
    timeout=API_TIMEOUT_SECONDS,
    messages=[{
        "role": "user", 
        "content": prompt
    }]
)
return response.content[0].text
```

## Error Handling Strategy

- All exceptions in `decide()` caught and converted to default CoachingAction
- API timeouts handled with fallback responses
- Empty/invalid analysis objects handled gracefully
- No exceptions propagated to caller

## Key Implementation Details

### Strategy Selection Logic

```python
# Priority-ordered rule evaluation:
# 1. Check for major pronunciation issues
# 2. Check for off-topic or empty input  
# 3. Check for recent strategy repetition
# 4. Check argument quality and turn number
# 5. Check argument structure (claim/reasoning/evidence)
# 6. Default to PROBE
```

### Response Generation

- Model: `claude-3-5-sonnet-20241022`
- Max tokens: 500
- Timeout: 30 seconds  
- Output: 1-3 sentences, conversational tone
- No markdown, bullet points, or teacher-like praise

### Anti-Repetition Logic

- Track last 2 strategies in coaching_history
- If preferred strategy matches last 2, select next applicable strategy from priority list
- Ensures variety in coaching approach

This design provides a complete, implementable specification for coach_policy.py that meets all requirements and acceptance criteria.