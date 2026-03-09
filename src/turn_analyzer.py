import asyncio
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from typing import List, Optional

from anthropic import AsyncAnthropic

from shared_types import (
    TurnInput,
    ArgumentResult,
    ErrorSeverity,
    WordError,
    PronunciationResult,
    TurnAnalysis,
    TurnIntent,
)

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
                max_tokens=2048,
                system=(
                    "You are a JSON-only API. You must respond with valid JSON and nothing else. "
                    "No markdown, no explanation, no code fences. Only a single JSON object."
                ),
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
        """Build prompt for 4-dimension argument analysis."""
        prior_context = ""
        if turn_input.prior_turns:
            last = turn_input.prior_turns[-1]
            prior_context = f"\nPrevious turn summary: {last.get('summary', '')}"

        return f"""Analyze this English debate statement from a Chinese L2 learner.

Topic: {turn_input.topic}
Position: {turn_input.user_position}
Turn: {turn_input.turn_number}{prior_context}

Student said: "{turn_input.transcript}"

Score on FOUR dimensions (each 0.0–1.0). Be GENEROUS — this is a language learner, not a competitive debater.

1. CLARITY (Has the student clearly stated their position and main point?)
   1.0 = Position crystal clear, sentences complete and well-structured
   0.7 = Position clear, minor grammatical issues
   0.4 = Position somewhat clear but hard to follow
   0.1 = Very unclear or off-topic

2. REASONING (Has the student given logical reasons to support their claim?)
   1.0 = Two or more well-connected reasons
   0.7 = One solid reason with explanation
   0.4 = Reason implied but not fully explained
   0.1 = No reasoning given

3. DEPTH (Has the student added any example, analogy, or real-world reference?)
   1.0 = Specific example or real-world reference given
   0.5 = Vague reference (e.g. "like many people say")
   0.0 = No example or elaboration at all
   NOTE: Personal experience and everyday examples count fully. Data not required.

4. FLUENCY (Is the English natural, grammatically acceptable, and easy to follow?)
   1.0 = Natural, mostly correct grammar, good connectives
   0.7 = Understandable with a few errors
   0.4 = Several errors that affect understanding
   0.1 = Very hard to understand

Weighted score = 0.3*clarity + 0.3*reasoning + 0.1*depth + 0.3*fluency

Also identify:
- has_claim: true if student stated a position
- has_reasoning: true if at least one reason given
- has_evidence: true if depth_score >= 0.4
- logical_gaps: up to 2 items, max 8 words each
- vocabulary_flags: informal or weak words used, max 3 items, max 5 words each
- summary: ONE encouraging sentence (max 20 words) naming the strongest aspect and the one thing to improve next

Output only this JSON, no extra text:
{{
    "clarity_score": 0.0,
    "reasoning_score": 0.0,
    "depth_score": 0.0,
    "fluency_score_arg": 0.0,
    "argument_score": 0.0,
    "has_claim": false,
    "has_reasoning": false,
    "has_evidence": false,
    "logical_gaps": [],
    "vocabulary_flags": [],
    "clarity_feedback": "one short phrase",
    "reasoning_feedback": "one short phrase",
    "depth_feedback": "one short phrase",
    "fluency_feedback": "one short phrase",
    "summary": "one encouraging sentence"
}}"""

    def _detect_intent(self, transcript: str) -> TurnIntent:
        """
        Classify the student's input before scoring.
        DEBATE_STATEMENT → normal scoring path
        META_QUESTION    → student is asking the coach something
        OFF_TOPIC        → too short or irrelevant
        """
        t = transcript.strip().lower()

        if len(t.split()) < 4:
            return TurnIntent.OFF_TOPIC

        meta_phrases = [
            # Asking for clarification
            "what do you mean", "what did you mean", "could you clarify",
            "can you explain", "what are you asking",
            "what question", "what do you want",
            # Pushing back — student says they already answered
            "i already said", "i already stated", "i already mentioned",
            "i just said", "i told you", "didn't i just", "didn't i already",
            "i just gave", "i just provided", "i just told",
            "i gave you", "i provided", "haven't i",
            "don't understand the question",
        ]
        if any(p in t for p in meta_phrases) or (t.endswith("?") and len(t.split()) < 15):
            return TurnIntent.META_QUESTION

        return TurnIntent.DEBATE_STATEMENT

    def _parse_argument_response(self, response_text: str) -> ArgumentResult:
        """Parse Claude API response into ArgumentResult. Uses regex fallback if JSON is malformed."""
        logging.debug(f"Raw argument response: {response_text}")

        # ── Attempt 1: clean and parse as JSON ───────────────────────────────
        cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
        cleaned = re.sub(r"```", "", cleaned).strip()
        json_start = cleaned.find('{')
        json_end   = cleaned.rfind('}')
        if json_start != -1 and json_end != -1:
            candidate = cleaned[json_start:json_end + 1]
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            try:
                data = json.loads(candidate)
                clarity   = float(data.get("clarity_score",    0.0))
                reasoning = float(data.get("reasoning_score",  0.0))
                depth     = float(data.get("depth_score",      0.0))
                fluency   = float(data.get("fluency_score_arg",0.0))
                score     = round(0.3*clarity + 0.3*reasoning + 0.1*depth + 0.3*fluency, 3)
                return ArgumentResult(
                    clarity_score=clarity,
                    reasoning_score=reasoning,
                    depth_score=depth,
                    fluency_score_arg=fluency,
                    argument_score=data.get("argument_score", score),
                    has_claim=bool(data.get("has_claim", False)),
                    has_reasoning=bool(data.get("has_reasoning", False)),
                    has_evidence=bool(data.get("has_evidence", depth >= 0.4)),
                    logical_gaps=data.get("logical_gaps", []),
                    vocabulary_flags=data.get("vocabulary_flags", []),
                    summary=data.get("summary", "No analysis available"),
                    clarity_feedback=data.get("clarity_feedback", ""),
                    reasoning_feedback=data.get("reasoning_feedback", ""),
                    depth_feedback=data.get("depth_feedback", ""),
                    fluency_feedback=data.get("fluency_feedback", ""),
                )
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parse failed ({e}), trying field-by-field extraction")

        # ── Attempt 2: regex field-by-field extraction ────────────────────────
        try:
            def extract_bool(field: str) -> bool:
                m = re.search(rf'"{field}"\s*:\s*(true|false)', response_text, re.IGNORECASE)
                return m.group(1).lower() == "true" if m else False

            def extract_float(field: str) -> float:
                m = re.search(rf'"{field}"\s*:\s*([0-9.]+)', response_text)
                return float(m.group(1)) if m else DEFAULT_ARGUMENT_SCORE

            def extract_str(field: str) -> str:
                m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', response_text)
                return m.group(1) if m else "No analysis available"

            def extract_list(field: str) -> list:
                m = re.search(rf'"{field}"\s*:\s*\[([^\]]*)\]', response_text, re.DOTALL)
                if not m:
                    return []
                return re.findall(r'"([^"]*)"', m.group(1))

            logging.info("Using field-by-field extraction fallback")
            clarity   = extract_float("clarity_score")
            reasoning = extract_float("reasoning_score")
            depth     = extract_float("depth_score")
            fluency   = extract_float("fluency_score_arg")
            return ArgumentResult(
                clarity_score=clarity,
                reasoning_score=reasoning,
                depth_score=depth,
                fluency_score_arg=fluency,
                argument_score=round(0.3*clarity + 0.3*reasoning + 0.1*depth + 0.3*fluency, 3),
                has_claim=extract_bool("has_claim"),
                has_reasoning=extract_bool("has_reasoning"),
                has_evidence=depth >= 0.4,
                logical_gaps=extract_list("logical_gaps"),
                vocabulary_flags=extract_list("vocabulary_flags"),
                summary=extract_str("summary"),
                clarity_feedback=extract_str("clarity_feedback"),
                reasoning_feedback=extract_str("reasoning_feedback"),
                depth_feedback=extract_str("depth_feedback"),
                fluency_feedback=extract_str("fluency_feedback"),
            )
        except Exception as e:
            logging.error(f"Field-by-field extraction also failed: {e}")
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
            clarity_score=0.0,
            reasoning_score=0.0,
            depth_score=0.0,
            fluency_score_arg=0.0,
            argument_score=0.0,
            has_claim=False,
            has_reasoning=False,
            has_evidence=False,
            logical_gaps=[],
            vocabulary_flags=[],
            summary="Analysis unavailable",
            clarity_feedback="",
            reasoning_feedback="",
            depth_feedback="",
            fluency_feedback="",
        )

    def _create_default_pronunciation_result(self) -> PronunciationResult:
        """Create default PronunciationResult for error cases."""
        return PronunciationResult(
            mispronounced_words=[],
            fluency_score=DEFAULT_FLUENCY_SCORE,
            target_phonemes=[]
        )

    def _create_empty_analysis(self, turn_input: TurnInput, start_time: float) -> TurnAnalysis:
        """Create TurnAnalysis for empty transcript or error fallback."""
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        return TurnAnalysis(
            turn_input=turn_input,
            argument=self._create_default_argument_result(),
            pronunciation=self._create_default_pronunciation_result(),
            timestamp=datetime.now(),
            latency_ms=latency_ms,
        )