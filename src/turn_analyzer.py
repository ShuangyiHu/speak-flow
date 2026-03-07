import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from anthropic import AsyncAnthropic


@dataclass
class TurnInput:
    transcript: str
    session_id: str
    turn_number: int
    topic: str
    user_position: str  # "for" or "against"
    audio_path: str
    prior_turns: List[str]


@dataclass
class ArgumentResult:
    has_claim: bool
    has_reasoning: bool
    has_evidence: bool
    logical_gaps: List[str]
    vocabulary_flags: List[str]
    argument_score: float  # 0.0 to 1.0
    summary: str


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class WordError:
    word: str
    expected_ipa: str
    actual_ipa: str
    severity: ErrorSeverity


@dataclass
class PronunciationResult:
    mispronounced_words: List[WordError]
    fluency_score: float  # 0.0 to 1.0
    target_phonemes: List[str]  # Phonemes to focus on for improvement


@dataclass
class TurnAnalysis:
    turn_input: TurnInput
    argument: ArgumentResult
    pronunciation: PronunciationResult
    timestamp: datetime
    latency_ms: int


# Configuration constants
ANTHROPIC_MODEL = "claude-sonnet-4-5"
ANALYSIS_TIMEOUT_SECONDS = 30.0
DEFAULT_ARGUMENT_SCORE = 0.0
DEFAULT_FLUENCY_SCORE = 0.8
MFA_TIMEOUT_SECONDS = 2.5
ANTHROPIC_TIMEOUT_SECONDS = 30.0

# Environment variables
USE_STUB_MFA = os.getenv("USE_STUB_MFA", "False").lower() == "true"
MFA_LEXICON_PATH = os.getenv("MFA_LEXICON_PATH", "/path/to/lexicon.txt")
MFA_ACOUSTIC_MODEL_PATH = os.getenv("MFA_ACOUSTIC_MODEL_PATH", "/path/to/acoustic_model")


class TurnAnalyzer:
    def __init__(self, anthropic_api_key: str):
        """Initialize TurnAnalyzer with Anthropic client."""
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_api_key, timeout=ANTHROPIC_TIMEOUT_SECONDS)
        self.use_stub_mfa = USE_STUB_MFA

    async def analyze(self, turn_input: TurnInput) -> TurnAnalysis:
        """
        Main entry point: analyze a spoken turn for argument structure and pronunciation.
        
        Args:
            turn_input: TurnInput containing transcript, audio, and context
            
        Returns:
            TurnAnalysis: Complete analysis results within 3 seconds
        """
        start_time = time.time()
        
        # Handle empty transcript gracefully
        if not turn_input.transcript.strip():
            return self._create_empty_analysis(turn_input, start_time)
        
        try:
            # Run argument and pronunciation analysis concurrently with timeout
            argument_task = self._analyze_argument(turn_input)
            pronunciation_task = self._analyze_pronunciation(turn_input)
            
            argument_result, pronunciation_result = await asyncio.wait_for(
                asyncio.gather(
                    argument_task,
                    pronunciation_task,
                    return_exceptions=True
                ),
                timeout=ANALYSIS_TIMEOUT_SECONDS
            )
            
            # Handle exceptions from concurrent tasks
            if isinstance(argument_result, Exception):
                logging.error(f"Argument analysis failed: {argument_result}")
                argument_result = self._create_default_argument_result()
            if isinstance(pronunciation_result, Exception):
                logging.error(f"Pronunciation analysis failed: {pronunciation_result}")
                pronunciation_result = self._create_default_pronunciation_result()
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            return TurnAnalysis(
                turn_input=turn_input,
                argument=argument_result,
                pronunciation=pronunciation_result,
                timestamp=datetime.now(),
                latency_ms=latency_ms
            )
            
        except asyncio.TimeoutError:
            logging.error(f"Analysis timed out after {ANALYSIS_TIMEOUT_SECONDS} seconds")
            return self._create_empty_analysis(turn_input, start_time)
        except Exception as e:
            logging.error(f"Unexpected error during analysis: {e}")
            return self._create_empty_analysis(turn_input, start_time)

    async def _analyze_argument(self, turn_input: TurnInput) -> ArgumentResult:
        """
        Analyze argument structure using Claude API.
        
        Args:
            turn_input: Input data containing transcript and context
            
        Returns:
            ArgumentResult: Analysis of claim, reasoning, evidence, and overall score
        """
        prompt = self._build_argument_analysis_prompt(turn_input)
        
        try:
            response = await self.anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1000,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return self._parse_argument_response(response.content[0].text)
            
        except asyncio.TimeoutError:
            logging.error(f"Anthropic API call timed out after {ANTHROPIC_TIMEOUT_SECONDS} seconds")
            return self._create_default_argument_result()
        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            return self._create_default_argument_result()

    async def _analyze_pronunciation(self, turn_input: TurnInput) -> PronunciationResult:
        """
        Analyze pronunciation using Montreal Forced Aligner or stub implementation.
        
        Args:
            turn_input: Input data containing audio path and transcript
            
        Returns:
            PronunciationResult: Pronunciation errors, fluency score, target phonemes
        """
        if self.use_stub_mfa:
            return self._create_stub_pronunciation_result()
        
        # Run MFA in thread to avoid blocking event loop
        return await asyncio.to_thread(self._run_mfa_analysis, turn_input)

    def _run_mfa_analysis(self, turn_input: TurnInput) -> PronunciationResult:
        """
        Synchronous MFA analysis (runs in separate thread).
        
        Args:
            turn_input: Input containing audio file path and transcript
            
        Returns:
            PronunciationResult: MFA-based pronunciation analysis
        """
        try:
            # Construct MFA command
            cmd = [
                "mfa", "align",
                "--clean",
                "--include_original_text",
                turn_input.audio_path,
                MFA_LEXICON_PATH,
                MFA_ACOUSTIC_MODEL_PATH,
                "/tmp/mfa_output"
            ]
            
            # Run MFA subprocess with timeout
            result = subprocess.run(
                cmd,
                timeout=MFA_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse MFA output to extract pronunciation errors
            return self._parse_mfa_output(result.stdout, turn_input.transcript)
            
        except subprocess.TimeoutExpired:
            logging.error(f"MFA subprocess timed out after {MFA_TIMEOUT_SECONDS} seconds")
            return self._create_default_pronunciation_result()
        except subprocess.CalledProcessError as e:
            logging.error(f"MFA subprocess failed with return code {e.returncode}: {e.stderr}")
            return self._create_default_pronunciation_result()
        except FileNotFoundError:
            logging.error("MFA executable not found")
            return self._create_default_pronunciation_result()

    def _build_argument_analysis_prompt(self, turn_input: TurnInput) -> str:
        """
        Build Claude prompt for argument analysis.
        
        Args:
            turn_input: Input containing transcript and debate context
            
        Returns:
            str: Formatted prompt for Claude API
        """
        prior_context = "\n".join(turn_input.prior_turns[-3:]) if turn_input.prior_turns else "None"
        
        return f"""
Analyze this English debate turn using the Claim-Reason-Evidence framework.

CONTEXT:
Topic: {turn_input.topic}
User Position: {turn_input.user_position}
Turn Number: {turn_input.turn_number}
Previous Turns: {prior_context}

TRANSCRIPT TO ANALYZE:
{turn_input.transcript}

ANALYSIS REQUIRED:
1. Has Clear Claim: Does this contain a clear position statement? (not questions or off-topic remarks)
2. Has Reasoning: Does this provide logical reasoning to support the claim?
3. Has Evidence: Does this include specific examples, data, or expert citations?
4. Logical Gaps: What logical weaknesses or gaps exist?
5. Vocabulary Flags: Any inappropriate/informal language for academic debate?
6. Argument Score: Rate 0.0-1.0 based on CRE framework completeness
7. Summary: One sentence summary of the argument's strength

Respond in this exact JSON format:
{{
    "has_claim": true/false,
    "has_reasoning": true/false,
    "has_evidence": true/false,
    "logical_gaps": ["gap1", "gap2"],
    "vocabulary_flags": ["word1", "word2"],
    "argument_score": 0.75,
    "summary": "Brief summary of argument quality"
}}
"""

    def _parse_argument_response(self, response_text: str) -> ArgumentResult:
        """
        Parse Claude API response into ArgumentResult object.
        
        Args:
            response_text: Raw JSON response from Claude
            
        Returns:
            ArgumentResult: Parsed argument analysis
        """
        try:
            response_text = re.sub(r"```json|```", "", response_text).strip()
            data = json.loads(response_text)
            
            return ArgumentResult(
                has_claim=data.get("has_claim", False),
                has_reasoning=data.get("has_reasoning", False),
                has_evidence=data.get("has_evidence", False),
                logical_gaps=data.get("logical_gaps", []),
                vocabulary_flags=data.get("vocabulary_flags", []),
                argument_score=float(data.get("argument_score", DEFAULT_ARGUMENT_SCORE)),
                summary=data.get("summary", "No analysis available")
            )
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from Anthropic API: {e}")
            return self._create_default_argument_result()
        except (KeyError, ValueError) as e:
            logging.error(f"Invalid data format in Anthropic API response: {e}")
            return self._create_default_argument_result()

    def _parse_mfa_output(self, mfa_output: str, transcript: str) -> PronunciationResult:
        """
        Parse MFA alignment output to identify pronunciation errors.
        
        Args:
            mfa_output: Raw output from MFA alignment
            transcript: Original transcript for word matching
            
        Returns:
            PronunciationResult: Parsed pronunciation analysis
        """
        try:
            # Parse MFA TextGrid output format
            # This is a simplified parser - real implementation would use textgrid library
            mispronounced_words = []
            words = transcript.split()
            
            # Mock parsing logic - replace with actual TextGrid parsing
            for word in words:
                if self._is_mispronounced(word, mfa_output):
                    error = WordError(
                        word=word,
                        expected_ipa=self._get_expected_ipa(word),
                        actual_ipa=self._get_actual_ipa(word, mfa_output),
                        severity=self._calculate_severity(word)
                    )
                    mispronounced_words.append(error)
            
            fluency_score = self._calculate_fluency_score(mfa_output, len(words))
            target_phonemes = self._identify_target_phonemes(mispronounced_words)
            
            return PronunciationResult(
                mispronounced_words=mispronounced_words,
                fluency_score=fluency_score,
                target_phonemes=target_phonemes
            )
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON in MFA output: {e}")
            return self._create_default_pronunciation_result()
        except (ValueError, KeyError) as e:
            logging.error(f"Invalid format in MFA output: {e}")
            return self._create_default_pronunciation_result()

    def _is_mispronounced(self, word: str, mfa_output: str) -> bool:
        """Check if word is mispronounced based on MFA output."""
        # Stub implementation - replace with actual MFA output parsing
        return len(word) > 6  # Mock condition

    def _get_expected_ipa(self, word: str) -> str:
        """Get expected IPA pronunciation for word."""
        # Stub implementation - replace with dictionary lookup
        return f"/{word.lower()}/"

    def _get_actual_ipa(self, word: str, mfa_output: str) -> str:
        """Extract actual IPA from MFA output."""
        # Stub implementation - replace with MFA parsing
        return f"/{word.lower().replace('r', 'l')}/"

    def _calculate_severity(self, word: str) -> ErrorSeverity:
        """Calculate error severity based on phoneme differences."""
        # Stub implementation
        return ErrorSeverity.MEDIUM

    def _calculate_fluency_score(self, mfa_output: str, word_count: int) -> float:
        """Calculate fluency score from MFA timing data."""
        # Stub implementation - replace with actual timing analysis
        return max(0.0, 1.0 - (word_count * 0.02))

    def _identify_target_phonemes(self, errors: List[WordError]) -> List[str]:
        """Identify phonemes for targeted practice."""
        target_set = set()
        for error in errors:
            if 'r' in error.expected_ipa and 'l' in error.actual_ipa:
                target_set.add('/r/')
            if 'θ' in error.expected_ipa:
                target_set.add('/θ/')
            if 'ð' in error.expected_ipa:
                target_set.add('/ð/')
        return list(target_set)

    def _create_stub_pronunciation_result(self) -> PronunciationResult:
        """Create stub pronunciation result when USE_STUB_MFA=True."""
        return PronunciationResult(
            mispronounced_words=[],  # Always empty for stub
            fluency_score=DEFAULT_FLUENCY_SCORE,
            target_phonemes=[]
        )

    def _create_default_argument_result(self) -> ArgumentResult:
        """Create default ArgumentResult for error cases."""
        return ArgumentResult(
            has_claim=False,
            has_reasoning=False,
            has_evidence=False,
            logical_gaps=[],
            vocabulary_flags=[],
            argument_score=DEFAULT_ARGUMENT_SCORE,
            summary="Analysis unavailable"
        )

    def _create_default_pronunciation_result(self) -> PronunciationResult:
        """Create default PronunciationResult for error cases."""
        return PronunciationResult(
            mispronounced_words=[],
            fluency_score=DEFAULT_FLUENCY_SCORE,
            target_phonemes=[]
        )

    def _create_empty_analysis(self, turn_input: TurnInput, start_time: float) -> TurnAnalysis:
        """Create TurnAnalysis for empty transcript."""
        end_time = time.time