import os
import json
import re
import asyncio
import sys
from typing import List, Optional
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from shared_types import WordError, PronunciationResult, ErrorSeverity
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

CHINESE_L2_PRIORITY_PHONEMES = {
    "/θ/": "Chinese has no /θ/ sound. Tip: place tongue between teeth.",
    "/ð/": "The voiced /ð/ (as in 'the') has no equivalent in Mandarin.",
    "/v/": "Mandarin speakers often substitute /w/ for /v/.",
    "/r/": "English /r/ is retroflex; do not substitute Mandarin /r/ (closer to /ʒ/).",
    "/l/": "Final /l/ is often dropped by Mandarin speakers — keep the tongue up.",
    "/æ/": "The /æ/ vowel (as in 'cat') is often raised to /ɛ/ by Chinese speakers.",
    "/ŋ/": "Final /-ng/ is often denasalized — keep the back of tongue raised.",
}

@dataclass
class WordCorrection:
    word: str
    error_description: str
    correction_tip: str
    model_sentence: str
    severity: ErrorSeverity

@dataclass
class PronunciationFeedback:
    corrections: List[WordCorrection]
    drill_sentence: str
    fluency_comment: str
    overall_message: str
    has_errors: bool
    latency_ms: int

class PronunciationCoach:
    def __init__(self, anthropic_api_key: str) -> None:
        """
        Initialize the pronunciation coach with Anthropic API client.
        
        Args:
            anthropic_api_key: API key for Claude AI
        """
        self._client = AsyncAnthropic(
            api_key=anthropic_api_key,
            timeout=30.0
        )
        self._mfa_enabled = False  # MFA stub implementation
        
    def toggle_mfa(self, enabled: bool) -> None:
        """
        Toggle Multi-Factor Authentication (MFA) - stub implementation.
        
        Args:
            enabled: Whether to enable or disable MFA
        """
        self._mfa_enabled = enabled
        
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
            
        Timeout: 2.8 seconds via asyncio.wait_for
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            timeout_task = self._generate_feedback_impl(
                pronunciation_result, transcript, topic, start_time
            )
            
            if sys.version_info >= (3, 11):
                async with asyncio.timeout(2.8):
                    return await timeout_task
            else:
                return await asyncio.wait_for(timeout_task, timeout=2.8)
                
        except asyncio.TimeoutError:
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return PronunciationFeedback(
                corrections=[],
                drill_sentence="",
                fluency_comment="Your pronunciation needs attention.",
                overall_message="Keep practicing — pronunciation improves with time!",
                has_errors=False,
                latency_ms=latency_ms
            )
        except (OSError, ConnectionError) as e:
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return PronunciationFeedback(
                corrections=[],
                drill_sentence="",
                fluency_comment="Your pronunciation needs attention.",
                overall_message="Network error occurred. Please try again.",
                has_errors=False,
                latency_ms=latency_ms
            )
        except (json.JSONDecodeError, ValueError) as e:
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return PronunciationFeedback(
                corrections=[],
                drill_sentence="",
                fluency_comment="Your pronunciation needs attention.",
                overall_message="Response parsing error. Please try again.",
                has_errors=False,
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return PronunciationFeedback(
                corrections=[],
                drill_sentence="",
                fluency_comment="Your pronunciation needs attention.",
                overall_message="Keep practicing — pronunciation improves with time!",
                has_errors=False,
                latency_ms=latency_ms
            )
    
    async def _generate_feedback_impl(
        self,
        pronunciation_result: PronunciationResult,
        transcript: str,
        topic: str,
        start_time: float
    ) -> PronunciationFeedback:
        """
        Internal implementation for generate_feedback.
        """
        # Check empty case
        if not pronunciation_result.mispronounced_words:
            fluency_comment = (
                "Your fluency sounds good — keep up the natural rhythm!" 
                if pronunciation_result.fluency_score >= 0.7 
                else "Try to speak a bit more smoothly — aim for fewer pauses."
            )
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return PronunciationFeedback(
                corrections=[],
                drill_sentence="",
                fluency_comment=fluency_comment,
                overall_message="Your pronunciation was clear this turn. Keep it up!",
                has_errors=False,
                latency_ms=latency_ms
            )
        
        # Filter to top 3 errors by severity
        errors = sorted(pronunciation_result.mispronounced_words, 
                      key=lambda x: (x.severity.value if hasattr(x.severity, 'value') else 0, x.word))[:3]
        
        # Generate corrections and drill sentence concurrently
        corrections_task = self._generate_corrections(errors, transcript)
        drill_task = self._generate_drill_sentence(errors, topic)
        
        corrections, drill_sentence = await asyncio.gather(
            corrections_task, drill_task
        )
        
        # Build fluency comment
        fluency_comment = (
            "Your fluency sounds good — keep up the natural rhythm!" 
            if pronunciation_result.fluency_score >= 0.7 
            else "Try to speak a bit more smoothly — aim for fewer pauses."
        )
        
        # Build overall message
        overall_message = "Great effort! Let's work on a few pronunciation details to make your speech even clearer."
        
        latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return PronunciationFeedback(
            corrections=corrections,
            drill_sentence=drill_sentence,
            fluency_comment=fluency_comment,
            overall_message=overall_message,
            has_errors=len(corrections) > 0,
            latency_ms=latency_ms
        )
    
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
        if not errors:
            return []
        
        try:
            correction_tasks = [
                self._generate_single_correction(error, transcript)
                for error in errors
            ]
            corrections = await asyncio.gather(*correction_tasks)
            return corrections
        except (OSError, ConnectionError):
            # Return fallback corrections for network errors
            return [
                WordCorrection(
                    word=error.word,
                    error_description="Network error during analysis",
                    correction_tip="Practice this word slowly and listen to native speaker recordings.",
                    model_sentence=f"Please practice saying '{error.word}' carefully.",
                    severity=error.severity
                )
                for error in errors
            ]
        except (json.JSONDecodeError, ValueError):
            # Return fallback corrections for parsing errors
            return [
                WordCorrection(
                    word=error.word,
                    error_description="Response parsing error occurred",
                    correction_tip="Practice this word slowly and listen to native speaker recordings.",
                    model_sentence=f"Please practice saying '{error.word}' carefully.",
                    severity=error.severity
                )
                for error in errors
            ]
        except Exception:
            # Return fallback corrections for all other errors
            return [
                WordCorrection(
                    word=error.word,
                    error_description="Pronunciation needs attention",
                    correction_tip="Practice this word slowly and listen to native speaker recordings.",
                    model_sentence=f"Please practice saying '{error.word}' carefully.",
                    severity=error.severity
                )
                for error in errors
            ]
    
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
        try:
            prompt = f"""A Chinese learner of English mispronounced the word '{error.word}'.
Expected IPA: {error.expected_ipa}. Detected IPA: {error.actual_ipa}.
The word appeared in this sentence: '{transcript}'.
Respond ONLY with valid JSON (no markdown fences) with keys:
error_description (1 sentence explaining what went wrong),
correction_tip (1-2 sentences specific to Chinese speakers, e.g. mouth position),
model_sentence (short sentence 6-10 words using this word correctly)"""

            response = await self._client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_text = response.content[0].text
            clean_text = re.sub(r"```[a-z]*\n?|```", "", raw_text).strip()
            parsed_data = json.loads(clean_text)
            
            return WordCorrection(
                word=error.word,
                error_description=parsed_data.get("error_description", "Pronunciation needs attention"),
                correction_tip=parsed_data.get("correction_tip", "Practice this word slowly and listen to native speaker recordings."),
                model_sentence=parsed_data.get("model_sentence", f"Please practice saying '{error.word}' carefully."),
                severity=error.severity
            )
            
        except (OSError, ConnectionError):
            return WordCorrection(
                word=error.word,
                error_description="Network error during analysis",
                correction_tip="Practice this word slowly and listen to native speaker recordings.",
                model_sentence=f"Please practice saying '{error.word}' carefully.",
                severity=error.severity
            )
        except (json.JSONDecodeError, ValueError):
            return WordCorrection(
                word=error.word,
                error_description="Response parsing error occurred",
                correction_tip="Practice this word slowly and listen to native speaker recordings.",
                model_sentence=f"Please practice saying '{error.word}' carefully.",
                severity=error.severity
            )
        except Exception:
            return WordCorrection(
                word=error.word,
                error_description="Pronunciation needs attention",
                correction_tip="Practice this word slowly and listen to native speaker recordings.",
                model_sentence=f"Please practice saying '{error.word}' carefully.",
                severity=error.severity
            )
    
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
        if not errors:
            return ""
        
        try:
            phoneme_list = [error.expected_ipa for error in errors]
            phonemes_str = ", ".join(phoneme_list)
            
            prompt = f"""Create a single English sentence (10-15 words) about the topic '{topic}' that naturally contains words with these sounds: {phonemes_str}.
Return ONLY the sentence, no explanation."""

            response = await self._client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            sentence = response.content[0].text.strip()
            return sentence
            
        except (OSError, ConnectionError):
            first_word = errors[0].word
            return f"Network error occurred. Please practice saying '{first_word}' for better pronunciation."
        except (json.JSONDecodeError, ValueError):
            first_word = errors[0].word
            return f"Response error occurred. Please practice saying '{first_word}' for better pronunciation."
        except Exception:
            # Fallback sentence using first error word
            first_word = errors[0].word
            return f"Please practice saying '{first_word}' in this sentence for better pronunciation."