import asyncio
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json
import tempfile

from anthropic import AsyncAnthropic
import logging

logger = logging.getLogger(__name__)

# Environment variables
USE_STUB_MFA = os.getenv("USE_STUB_MFA", "false").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
ANALYSIS_TIMEOUT_SECONDS = 3

# Scoring thresholds
MIN_CLAIM_CONFIDENCE = 0.6
HIGH_ARGUMENT_SCORE_THRESHOLD = 0.7
LOW_ARGUMENT_SCORE_THRESHOLD = 0.4

# MFA configuration
MFA_COMMAND_TEMPLATE = "mfa align {audio_path} {dict_path} {acoustic_model} {output_dir}"
MFA_DICT_PATH = Path("/opt/mfa/dictionaries/english_us_arpa.dict")
MFA_ACOUSTIC_MODEL = "english_us_arpa"

@dataclass
class TurnInput:
    transcript: str
    session_id: str
    turn_number: int
    topic: str
    user_position: str  # "pro" or "con"
    audio_path: Path
    prior_turns: List[Dict[str, Any]] = field(default_factory=list)

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

class TurnAnalyzer:
    def __init__(self):
        self.anthropic_client = AsyncAnthropic(
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
        
    async def _analyze_argument(self, turn_input: TurnInput) -> ArgumentResult:
        """
        Analyze argument structure using Claude API.
        
        Args:
            turn_input: Input containing transcript and debate context
            
        Returns:
            ArgumentResult: Structured argument analysis
        """
        try:
            prompt = await self._build_claude_prompt(turn_input)
            
            async with self.anthropic_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            ) as stream:
                response_text = ""
                async for chunk in stream:
                    if hasattr(chunk, 'type') and chunk.type == "content_block_delta":
                        response_text += chunk.delta.text
            
            return self._parse_claude_response(response_text)
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return ArgumentResult(
                has_claim=False,
                has_reasoning=False,
                has_evidence=False,
                logical_gaps=["API error occurred"],
                vocabulary_flags=[],
                argument_score=0.0,
                summary="Analysis failed due to API error"
            )
        
    async def _analyze_pronunciation(self, turn_input: TurnInput) -> PronunciationResult:
        """
        Analyze pronunciation using MFA or stub implementation.
        
        Args:
            turn_input: Input containing audio path and transcript
            
        Returns:
            PronunciationResult: Pronunciation analysis with errors and scores
        """
        if USE_STUB_MFA:
            return self._call_mfa_stub(turn_input.audio_path, turn_input.transcript)
        else:
            return await self._call_mfa_real(turn_input.audio_path, turn_input.transcript)
        
    async def _call_mfa_real(self, audio_path: Path, transcript: str) -> PronunciationResult:
        """
        Call Montreal Forced Aligner subprocess using asyncio.to_thread.
        
        Args:
            audio_path: Path to audio file
            transcript: Expected transcript for alignment
            
        Returns:
            PronunciationResult: Parsed MFA output
        """
        def _run_mfa_sync(audio_path: Path, output_dir: Path) -> subprocess.CompletedProcess:
            cmd = MFA_COMMAND_TEMPLATE.format(
                audio_path=audio_path,
                dict_path=MFA_DICT_PATH,
                acoustic_model=MFA_ACOUSTIC_MODEL,
                output_dir=output_dir
            )
            return subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir)
                
                result = await asyncio.to_thread(_run_mfa_sync, audio_path, temp_output_dir)
                
                if result.returncode == 0:
                    return self._parse_mfa_output(temp_output_dir, transcript)
                else:
                    logger.error(f"MFA subprocess failed: {result.stderr}")
                    return PronunciationResult(
                        mispronounced_words=[],
                        fluency_score=0.5,
                        target_phonemes=[]
                    )
                    
        except Exception as e:
            logger.error(f"MFA processing error: {e}")
            return PronunciationResult(
                mispronounced_words=[],
                fluency_score=0.5,
                target_phonemes=[]
            )
        
    def _call_mfa_stub(self, audio_path: Path, transcript: str) -> PronunciationResult:
        """
        Stub implementation of MFA for testing/development.
        
        Args:
            audio_path: Path to audio file (unused in stub)
            transcript: Expected transcript (unused in stub)
            
        Returns:
            PronunciationResult: Clean result with no errors
        """
        return PronunciationResult(
            mispronounced_words=[],
            fluency_score=0.8,
            target_phonemes=[]
        )
        
    async def _build_claude_prompt(self, turn_input: TurnInput) -> str:
        """
        Build Claude API prompt for argument analysis.
        
        Args:
            turn_input: Input with transcript, topic, position, and context
            
        Returns:
            str: Formatted prompt for Claude
        """
        prior_context = ""
        if turn_input.prior_turns:
            prior_context = "\n\nPrior turns in debate:\n" + "\n".join([
                f"Turn {turn.get('turn_number', 'N/A')}: {turn.get('transcript', '')}"
                for turn in turn_input.prior_turns[-3:]  # Last 3 turns for context
            ])
        
        prompt = f"""
Analyze this debate turn using the Claim-Reason-Evidence (CRE) framework for English L2 learners.

Topic: {turn_input.topic}
Student Position: {turn_input.user_position}
Turn #{turn_input.turn_number}
Transcript: "{turn_input.transcript}"
{prior_context}

Evaluate:
1. Has clear CLAIM (position statement)? 
2. Has REASONING (logical support)?
3. Has EVIDENCE (facts, examples, data)?
4. Logical gaps or weaknesses?
5. Vocabulary appropriateness for L2 learner?
6. Overall argument strength (0.0-1.0)?

Return JSON format:
{{
    "has_claim": boolean,
    "has_reasoning": boolean, 
    "has_evidence": boolean,
    "logical_gaps": ["gap1", "gap2"],
    "vocabulary_flags": ["flag1", "flag2"],
    "argument_score": 0.0-1.0,
    "summary": "Brief analysis summary"
}}
"""
        return prompt
        
    def _parse_claude_response(self, response_text: str) -> ArgumentResult:
        """
        Parse Claude API response into ArgumentResult structure.
        
        Args:
            response_text: Raw response from Claude API
            
        Returns:
            ArgumentResult: Parsed argument analysis
        """
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return ArgumentResult(
                    has_claim=data.get('has_claim', False),
                    has_reasoning=data.get('has_reasoning', False),
                    has_evidence=data.get('has_evidence', False),
                    logical_gaps=data.get('logical_gaps', []),
                    vocabulary_flags=data.get('vocabulary_flags', []),
                    argument_score=max(0.0, min(1.0, data.get('argument_score', 0.0))),
                    summary=data.get('summary', 'Unable to parse response')
                )
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return ArgumentResult(
                has_claim=False,
                has_reasoning=False,
                has_evidence=False,
                logical_gaps=["Response parsing error"],
                vocabulary_flags=[],
                argument_score=0.0,
                summary="Failed to analyze argument structure"
            )
        
    def _parse_mfa_output(self, mfa_output_dir: Path, transcript: str) -> PronunciationResult:
        """
        Parse MFA TextGrid output into PronunciationResult.
        
        Args:
            mfa_output_dir: Directory containing MFA TextGrid files
            transcript: Original transcript for word matching
            
        Returns:
            PronunciationResult: Parsed pronunciation analysis
        """
        # TODO: Implement TextGrid parsing
        # This is a simplified implementation
        mispronounced_words = []
        fluency_score = 0.8
        target_phonemes = ["θ", "ð", "ɹ"]  # Common problem phonemes for Chinese L2 learners
        
        return PronunciationResult(
            mispronounced_words=mispronounced_words,
            fluency_score=fluency_score,
            target_phonemes=target_phonemes
        )
        
    def _handle_empty_transcript(self, turn_input: TurnInput) -> TurnAnalysis:
        """
        Handle empty transcript case gracefully.
        
        Args:
            turn_input: Input with empty transcript
            
        Returns:
            TurnAnalysis: Default analysis for empty input
        """
        start_time = datetime.utcnow()
        
        default_argument = ArgumentResult(
            has_claim=False,
            has_reasoning=False,
            has_evidence=False,
            logical_gaps=["Empty transcript"],
            vocabulary_flags=[],
            argument_score=0.0,
            summary="No content to analyze"
        )
        
        default_pronunciation = PronunciationResult(
            mispronounced_words=[],
            fluency_score=0.0,
            target_phonemes=[]
        )
        
        return TurnAnalysis(
            turn_input=turn_input,
            argument=default_argument,
            pronunciation=default_pronunciation,
            timestamp=start_time,
            latency_ms=0
        )
        
    def _calculate_latency(self, start_time: datetime) -> int:
        """
        Calculate processing latency in milliseconds.
        
        Args:
            start_time: Analysis start timestamp
            
        Returns:
            int: Latency in milliseconds
        """
        end_time = datetime.utcnow()
        latency = (end_time - start_time).total_seconds() * 1000
        return int(latency)