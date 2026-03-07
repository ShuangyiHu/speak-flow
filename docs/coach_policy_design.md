# CoachPolicyAgent Technical Design

## Module Dependencies

```python
import asyncio
import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from anthropic import AsyncAnthropic
```

## Data Models

### CoachingStrategy Enum
```python
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
    target_word: str = ""
    difficulty_delta: int = 0
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

## Main Class

### CoachPolicyAgent Class
```python
class CoachPolicyAgent:
    """
    AI debate coaching policy agent that decides coaching strategies and generates responses.
    """
    
    # Class constants
    MODEL_NAME = "claude-3-5-sonnet-20241022"
    API_TIMEOUT = 30.0
    DECIDE_TIMEOUT = 10.0
    MAX_TOKENS = 150
    
    def __init__(self, anthropic_api_key: str) -> None:
        """
        Initialize the CoachPolicyAgent with Anthropic API client.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude service
        """
```

### Public Methods

#### Main Entry Point
```python
async def decide(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingAction:
    """
    Main entry point to decide coaching action based on turn analysis.
    
    Args:
        analysis: TurnAnalysis object containing student's turn data
        context: SessionContext with session state and history
        
    Returns:
        CoachingAction with strategy, response text, and metadata
        
    Notes:
        - Completes within 10 seconds total
        - Returns default action on any error (never raises)
        - Handles empty transcript with PROBE strategy
    """
```

### Private Methods

#### Strategy Selection
```python
def _select_strategy(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingStrategy:
    """
    SYNCHRONOUS method to select coaching strategy using deterministic rules.
    
    Args:
        analysis: TurnAnalysis object
        context: SessionContext object
        
    Returns:
        CoachingStrategy enum value
        
    Priority rules:
        1. CORRECT_PRONUNCIATION if major mispronunciation detected
        2. REDIRECT if empty/off-topic (argument_score < 0.1)
        3. Skip strategy if used in last 2 turns
        4. CHALLENGE if argument_score >= 0.7 and turn_number > 2
        5. PRAISE_AND_PUSH if argument_score >= 0.7
        6. PROBE if has_claim=True but has_reasoning=False
        7. PROBE if has_reasoning=True but has_evidence=False
        8. Default: PROBE
    """
```

#### Response Generation
```python
async def _generate_response(
    self, 
    analysis: TurnAnalysis, 
    context: SessionContext, 
    strategy: CoachingStrategy
) -> str:
    """
    Generate response text using Claude API.
    
    Args:
        analysis: TurnAnalysis object
        context: SessionContext object
        strategy: Selected CoachingStrategy
        
    Returns:
        Response text (1-3 sentences, markdown stripped)
        
    Notes:
        - Calls Anthropic API with 30s timeout
        - Includes last 3 turns of coaching history in prompt
        - Strategy-specific response generation
        - Strips markdown formatting from response
    """
```

#### Prompt Building
```python
def _build_prompt(
    self, 
    analysis: TurnAnalysis, 
    context: SessionContext, 
    strategy: CoachingStrategy
) -> str:
    """
    Build Claude API prompt string with strategy-specific instructions.
    
    Args:
        analysis: TurnAnalysis object
        context: SessionContext object
        strategy: Selected CoachingStrategy
        
    Returns:
        Complete prompt string for Claude API
        
    Includes:
        - Student transcript
        - Debate topic and position
        - Strategy-specific instructions
        - Response format requirements (1-3 sentences, no bullets)
        - Last 3 coaching history entries
    """
```

#### Difficulty Calculation
```python
def _calculate_difficulty_delta(self, context: SessionContext) -> int:
    """
    Calculate difficulty adjustment based on recent performance.
    
    Args:
        context: SessionContext with argument_scores history
        
    Returns:
        int: -1 (easier), 0 (same), +1 (harder)
        
    Logic:
        +1 if last 2 argument_scores >= 0.7
        -1 if last 2 argument_scores < 0.3
         0 otherwise
    """
```

#### Anti-Repetition Logic
```python
def _should_skip_strategy(self, strategy: CoachingStrategy, context: SessionContext) -> bool:
    """
    Check if strategy should be skipped due to recent usage.
    
    Args:
        strategy: CoachingStrategy to check
        context: SessionContext with coaching_history
        
    Returns:
        bool: True if strategy was used in last 2 turns
    """
```

#### Pronunciation Target Extraction
```python
def _get_pronunciation_target(self, analysis: TurnAnalysis) -> str:
    """
    Extract target word for pronunciation correction.
    
    Args:
        analysis: TurnAnalysis object
        
    Returns:
        str: First major mispronounced word, or "" if none
    """
```

#### Fallback Action
```python
def _create_default_action(self, context: SessionContext) -> CoachingAction:
    """
    Create safe fallback CoachingAction.
    
    Args:
        context: SessionContext for topic-relevant question
        
    Returns:
        CoachingAction with strategy=PROBE and generic opening question
    """
```

#### Utility Methods
```python
def _strip_markdown(self, text: str) -> str:
    """
    Remove markdown formatting from text.
    
    Args:
        text: Input text potentially containing markdown
        
    Returns:
        str: Text with markdown removed
    """

def _is_off_topic(self, analysis: TurnAnalysis) -> bool:
    """
    Determine if student response is off-topic.
    
    Args:
        analysis: TurnAnalysis object
        
    Returns:
        bool: True if argument_score < 0.1 or transcript is empty
    """

def _get_recent_strategies(self, context: SessionContext, count: int = 3) -> List[CoachingStrategy]:
    """
    Get most recent coaching strategies from history.
    
    Args:
        context: SessionContext with coaching_history
        count: Number of recent strategies to return
        
    Returns:
        List[CoachingStrategy]: Recent strategies (newest first)
    """
```

## External API Integration

### Anthropic SDK Usage
```python
# Client initialization in __init__
self._client = AsyncAnthropic(
    api_key=anthropic_api_key,
    timeout=self.API_TIMEOUT
)

# API call pattern in _generate_response
response = await self._client.messages.create(
    model=self.MODEL_NAME,
    max_tokens=self.MAX_TOKENS,
    temperature=0.7,
    messages=[{
        "role": "user",
        "content": prompt
    }]
)
```

## Constants and Configuration

```python
class CoachPolicyAgent:
    # Anthropic API configuration
    MODEL_NAME = "claude-3-5-sonnet-20241022"
    API_TIMEOUT = 30.0
    MAX_TOKENS = 150
    
    # Performance constraints
    DECIDE_TIMEOUT = 10.0
    
    # Strategy selection thresholds
    OFF_TOPIC_THRESHOLD = 0.1
    STRONG_ARGUMENT_THRESHOLD = 0.7
    WEAK_ARGUMENT_THRESHOLD = 0.3
    
    # Anti-repetition settings
    MAX_CONSECUTIVE_SAME_STRATEGY = 2
    
    # Default response templates
    DEFAULT_OPENING_QUESTIONS = {
        "for": "What's your strongest reason for supporting this position?",
        "against": "What's your main concern with this approach?",
        "neutral": "What's your initial take on this topic?"
    }
```

## Error Handling Patterns

```python
# In decide() method - never raise exceptions
try:
    # Strategy selection and response generation
    pass
except asyncio.TimeoutError:
    return self._create_default_action(context)
except Exception as e:
    # Log error internally but don't raise
    return self._create_default_action(context)

# Timeout wrapper for decide()
async def decide(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingAction:
    try:
        return await asyncio.wait_for(
            self._decide_internal(analysis, context),
            timeout=self.DECIDE_TIMEOUT
        )
    except asyncio.TimeoutError:
        return self._create_default_action(context)
```

## Import Requirements

The module requires these external dependencies:
- `anthropic` - AsyncAnthropic client for Claude API
- Standard library: `asyncio`, `re`, `enum`, `dataclasses`, `typing`

Expected import of external types (not defined in this module):
```python
from turn_analyzer import TurnAnalysis  # Contains transcript, argument analysis, pronunciation data
```