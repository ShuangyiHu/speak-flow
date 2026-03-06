# TurnAnalyzer Technical Design

## Module: turn_analyzer.py

### Data Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

@dataclass
class TurnInput:
    transcript: str
    session_id: str
    turn_number: int
    topic: str
    user_position: str  # "pro" or "con"
    audio_path: Path
    prior_turns: List[Dict[str, Any]] = Field(default_factory=list)

@dataclass
class ArgumentResult:
    has_claim: bool
    has_reasoning: bool
    has_evidence: bool
    logical_gaps: List[str]
    vocabulary_flags: List[str]  # advanced/inappropriate vocabulary usage
    argument_score: float  # 0.0 to 1.0
    summary: str

@dataclass
class WordError:
    word: str
    expected_ipa: str
    actual_ipa: str
    severity: str  # "minor", "moderate", "severe"

@dataclass
class PronunciationResult:
    mispronounced_words: List[WordError]
    fluency_score: float  # 0.0 to 1.0
    target_phonemes: List[str]  # phonemes to focus on for improvement

@dataclass
class TurnAnalysis:
    turn_input: TurnInput
    argument: ArgumentResult
    pronunciation: PronunciationResult
    timestamp: datetime
    latency_ms: int
```

### Constants and Configuration

```python
# Environment variables
USE_STUB_MFA: bool = os.getenv("USE_STUB_MFA", "false").lower() == "true"
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
ANALYSIS_TIMEOUT_SECONDS: int = 3

# Scoring thresholds
MIN_CLAIM_CONFIDENCE: float = 0.6
HIGH_ARGUMENT_SCORE_THRESHOLD: float = 0.7
LOW_ARGUMENT_SCORE_THRESHOLD: float = 0.4

# MFA configuration
MFA_COMMAND_TEMPLATE: str = "mfa align {audio_path} {dict_path} {acoustic_model} {output_dir}"
MFA_DICT_PATH: Path = Path("/opt/mfa/dictionaries/english_us_arpa.dict")
MFA_ACOUSTIC_MODEL: str = "english_us_arpa"
```

### External Dependencies

```python
import asyncio
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json
import tempfile

from anthropic import AsyncAnthropic
import logging

logger = logging.getLogger(__name__)
```

### Main Class

```python
class TurnAnalyzer:
    def __init__(self):
        self.anthropic_client: AsyncAnthropic = AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY
        )
        
    async def analyze(self, turn_input: TurnInput) -> TurnAnalysis:
        """
        Main analysis method. Runs argument and pronunciation analysis concurrently.
        
        Args:
            turn_input: TurnInput containing transcript, audio, and context
            
        Returns:
            TurnAnalysis: Complete analysis results
            
        Raises:
            asyncio.TimeoutError: If analysis exceeds ANALYSIS_TIMEOUT_SECONDS
        """
        
    async def _analyze_argument(self, turn_input: TurnInput) -> ArgumentResult:
        """
        Analyze argument structure using Claude API.
        
        Args:
            turn_input: Input containing transcript and debate context
            
        Returns:
            ArgumentResult: Structured argument analysis
        """
        
    async def _analyze_pronunciation(self, turn_input: TurnInput) -> PronunciationResult:
        """
        Analyze pronunciation using MFA or stub implementation.
        
        Args:
            turn_input: Input containing audio path and transcript
            
        Returns:
            PronunciationResult: Pronunciation analysis with errors and scores
        """
        
    async def _call_mfa_real(self, audio_path: Path, transcript: str) -> PronunciationResult:
        """
        Call Montreal Forced Aligner subprocess using asyncio.to_thread.
        
        Args:
            audio_path: Path to audio file
            transcript: Expected transcript for alignment
            
        Returns:
            PronunciationResult: Parsed MFA output
        """
        
    def _call_mfa_stub(self, audio_path: Path, transcript: str) -> PronunciationResult:
        """
        Stub implementation of MFA for testing/development.
        
        Args:
            audio_path: Path to audio file (unused in stub)
            transcript: Expected transcript (unused in stub)
            
        Returns:
            PronunciationResult: Clean result with no errors
        """
        
    async def _build_claude_prompt(self, turn_input: TurnInput) -> str:
        """
        Build Claude API prompt for argument analysis.
        
        Args:
            turn_input: Input with transcript, topic, position, and context
            
        Returns:
            str: Formatted prompt for Claude
        """
        
    def _parse_claude_response(self, response_text: str) -> ArgumentResult:
        """
        Parse Claude API response into ArgumentResult structure.
        
        Args:
            response_text: Raw response from Claude API
            
        Returns:
            ArgumentResult: Parsed argument analysis
        """
        
    def _parse_mfa_output(self, mfa_output_dir: Path, transcript: str) -> PronunciationResult:
        """
        Parse MFA TextGrid output into PronunciationResult.
        
        Args:
            mfa_output_dir: Directory containing MFA TextGrid files
            transcript: Original transcript for word matching
            
        Returns:
            PronunciationResult: Parsed pronunciation analysis
        """
        
    def _handle_empty_transcript(self, turn_input: TurnInput) -> TurnAnalysis:
        """
        Handle empty transcript case gracefully.
        
        Args:
            turn_input: Input with empty transcript
            
        Returns:
            TurnAnalysis: Default analysis for empty input
        """
        
    def _calculate_latency(self, start_time: datetime) -> int:
        """
        Calculate processing latency in milliseconds.
        
        Args:
            start_time: Analysis start timestamp
            
        Returns:
            int: Latency in milliseconds
        """
```

### Implementation Details

#### Claude API Integration Pattern
```python
# In _analyze_argument method:
async with self.anthropic_client.messages.stream(
    model=CLAUDE_MODEL,
    max_tokens=1000,
    messages=[{
        "role": "user", 
        "content": await self._build_claude_prompt(turn_input)
    }]
) as stream:
    response_text = ""
    async for chunk in stream:
        if chunk.type == "content_block_delta":
            response_text += chunk.delta.text
```

#### MFA Subprocess Call Pattern
```python
# In _call_mfa_real method:
def _run_mfa_sync(audio_path: Path, output_dir: Path) -> subprocess.CompletedProcess:
    cmd = MFA_COMMAND_TEMPLATE.format(
        audio_path=audio_path,
        dict_path=MFA_DICT_PATH,
        acoustic_model=MFA_ACOUSTIC_MODEL,
        output_dir=output_dir
    )
    return subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)

# Called via asyncio.to_thread
result = await asyncio.to_thread(_run_mfa_sync, audio_path, temp_dir)
```

#### Concurrent Analysis Pattern
```python
# In analyze method:
start_time = datetime.utcnow()

if not turn_input.transcript.strip():
    return self._handle_empty_transcript(turn_input)

try:
    argument_task = self._analyze_argument(turn_input)
    pronunciation_task = self._analyze_pronunciation(turn_input)
    
    argument_result, pronunciation_result = await asyncio.wait_for(
        asyncio.gather(argument_task, pronunciation_task),
        timeout=ANALYSIS_TIMEOUT_SECONDS
    )
    
    return TurnAnalysis(
        turn_input=turn_input,
        argument=argument_result,
        pronunciation=pronunciation_result,
        timestamp=start_time,
        latency_ms=self._calculate_latency(start_time)
    )
except asyncio.TimeoutError:
    logger.error(f"Analysis timeout for session {turn_input.session_id}")
    raise
```

#### Stub MFA Toggle Pattern
```python
# In _analyze_pronunciation method:
if USE_STUB_MFA:
    return self._call_mfa_stub(turn_input.audio_path, turn_input.transcript)
else:
    return await self._call_mfa_real(turn_input.audio_path, turn_input.transcript)
```

### Error Handling Strategy

- Empty transcript: Return default TurnAnalysis with zero scores
- MFA subprocess failure: Return PronunciationResult with empty errors and low fluency score  
- Claude API failure: Return ArgumentResult with has_claim=False and low argument_score
- Timeout: Raise asyncio.TimeoutError to caller
- File I/O errors: Log and return degraded results rather than failing

### Performance Requirements

- Target latency: < 3000ms for 30-word transcript
- Claude API call: < 2000ms expected
- MFA processing: < 2500ms for 30-word audio
- Concurrent execution reduces total latency vs sequential