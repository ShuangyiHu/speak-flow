# Technical Design: pronunciation_coach.py

## Module Overview

The `pronunciation_coach.py` module generates actionable pronunciation correction feedback for Chinese L2 English learners based on `PronunciationResult` from turn analysis. It uses Claude AI to generate personalized corrections and practice sentences.

## Data Models

```python
from dataclasses import dataclass
from typing import List
from shared_types import ErrorSeverity

@dataclass
class WordCorrection:
    word: str                   # The mispronounced word
    error_description: str      # Human-readable description of the error
    correction_tip: str         # Specific tip for Chinese L2 speakers
    model_sentence: str         # A short example sentence using this word correctly
    severity: ErrorSeverity     # Passed through from WordError

@dataclass
class PronunciationFeedback:
    corrections: List[WordCorrection]   # One per WordError, max 3 (highest severity first)
    drill_sentence: str                 # A sentence containing all target phonemes for practice
    fluency_comment: str                # One sentence comment on fluency_score
    overall_message: str                # Encouraging 1-sentence summary (always positive framing)
    has_errors: bool                    # True if corrections is non-empty
    latency_ms: int                     # Time taken to generate feedback
```

## Constants

```python
CHINESE_L2_PRIORITY_PHONEMES = {
    "/θ/": "Chinese has no /θ/ sound. Tip: place tongue between teeth.",
    "/ð/": "The voiced /ð/ (as in 'the') has no equivalent in Mandarin.",
    "/v/": "Mandarin speakers often substitute /w/ for /v/.",
    "/r/": "English /r/ is retroflex; do not substitute Mandarin /r/ (closer to /ʒ/).",
    "/l/": "Final /l/ is often dropped by Mandarin speakers — keep the tongue up.",
    "/æ/": "The /æ/ vowel (as in 'cat') is often raised to /ɛ/ by Chinese speakers.",
    "/ŋ/": "Final /-ng/ is often denasalized — keep the back of tongue raised.",
}
```

## Main Class

```python
class PronunciationCoach:
    def __init__(self, anthropic_api_key: str) -> None:
        """
        Initialize the pronunciation coach with Anthropic API client.
        
        Args:
            anthropic_api_key: API key for Claude AI
        """
        
    async def generate_feedback(
        self,
        pronunciation_result: PronunciationResult,
        transcript: str,
        topic: str,
    ) -> PronunciationFeedback:
        """
        Generate pronunciation feedback based on analysis results.
        
        Args:
            pronunciation_result: Analysis result with mispronounced_words and fluency_score
            transcript: The original spoken text
            topic: Discussion topic for context in drill sentences
            
        Returns:
            PronunciationFeedback with corrections and practice materials
            
        Timeout: 2.8 seconds via asyncio.timeout
        """
        
    async def _generate_corrections(
        self,
        errors: List[WordError],
        transcript: str,
    ) -> List[WordCorrection]:
        """
        Generate WordCorrection objects for multiple errors concurrently.
        
        Args:
            errors: List of WordError objects (max 3)
            transcript: Original transcript for context
            
        Returns:
            List of WordCorrection objects, one per error
            
        Implementation: Uses asyncio.gather() for concurrent LLM calls
        """
        
    async def _generate_single_correction(
        self,
        error: WordError,
        transcript: str
    ) -> WordCorrection:
        """
        Generate a single WordCorrection using Claude AI.
        
        Args:
            error: WordError with word, expected_ipa, actual_ipa, severity
            transcript: Original transcript for context
            
        Returns:
            WordCorrection with personalized tips for Chinese L2 speakers
            
        LLM Call: Claude with JSON response parsing
        Fallback: Generic correction on any exception
        """
        
    async def _generate_drill_sentence(
        self,
        errors: List[WordError],
        topic: str
    ) -> str:
        """
        Generate a practice sentence containing target phonemes.
        
        Args:
            errors: List of WordError objects to extract phonemes from
            topic: Topic context for natural sentence generation
            
        Returns:
            10-15 word sentence naturally containing target sounds
            Empty string if errors is empty
            
        LLM Call: Claude with simple text response
        Fallback: Sentence using first error word on exception
        """
```

## Dependencies and External Calls

### Required Imports
```python
import os
import json
import time
import re
import asyncio
from typing import List, Optional
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from shared_types import WordError, PronunciationResult, ErrorSeverity
from dotenv import load_dotenv

# Critical file header (after imports)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
```

### Anthropic SDK Usage Pattern
```python
# Initialization
self._client = AsyncAnthropic(
    api_key=anthropic_api_key,
    timeout=30.0  # Set timeout at client level
)

# LLM Call Pattern
try:
    response = await self._client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
        # No timeout parameter here - causes TypeError
    )
    raw_text = response.content[0].text
    # Strip markdown fences before JSON parsing
    clean_text = re.sub(r"```[a-z]*\n?|```", "", raw_text).strip()
    parsed_data = json.loads(clean_text)
except Exception:
    # Return fallback data
    pass
```

## Implementation Flow

### generate_feedback() Logic
1. **Start timer** for latency tracking
2. **Check empty case**: If `pronunciation_result.mispronounced_words` is empty:
   - Return `PronunciationFeedback` with `corrections=[]`, `has_errors=False`
   - Generate fluency comment based on `fluency_score`
   - Set `overall_message="Your pronunciation was clear this turn. Keep it up!"`
   - No LLM calls needed
3. **Filter errors**: Take top 3 by severity (HIGH > MEDIUM > LOW)
4. **Generate corrections**: Call `_generate_corrections()` concurrently
5. **Generate drill sentence**: Call `_generate_drill_sentence()`
6. **Build fluency comment**: 
   - If `fluency_score >= 0.7`: Positive comment
   - Else: "Try to speak a bit more smoothly — aim for fewer pauses."
7. **Return feedback** with calculated latency

### Concurrent Processing
- `_generate_corrections()` uses `asyncio.gather()` to process multiple errors in parallel
- Each error gets its own `_generate_single_correction()` call
- Fallback handling per individual correction prevents total failure

### Error Handling Strategy
- All LLM calls wrapped in try/except
- Individual correction failures don't break the entire response
- Generic fallback corrections provided for any API failures
- No exceptions propagate from `generate_feedback()`

### Timeout Management
- Overall method timeout: `asyncio.timeout(2.8)` in `generate_feedback()`
- Anthropic client timeout: `30.0` seconds at initialization
- Individual LLM calls inherit client timeout, no additional timeout parameter

This design ensures robust, fast pronunciation feedback generation with proper fallback handling and concurrent processing for optimal performance.