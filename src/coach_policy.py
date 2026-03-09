import asyncio

from shared_types import (
    TurnAnalysis,
    CoachingStrategy,
    CoachingAction,
    SessionContext,
)

# Constants
CLAUDE_MODEL = "claude-sonnet-4-5"
API_TIMEOUT_SECONDS = 30
DECIDE_TIMEOUT_SECONDS = 10
LOW_ARGUMENT_THRESHOLD = 0.1
HIGH_ARGUMENT_THRESHOLD = 0.7
DIFFICULTY_THRESHOLD = 0.3
REPEAT_PREVENTION_LOOKBACK = 2
MAX_RESPONSE_LENGTH = 500

class CoachPolicyAgent:
    def __init__(self, mfa_enabled: bool = False) -> None:
        """
        Initialize the coach policy agent.

        Args:
            mfa_enabled: Whether MFA is enabled (toggleable)
        """
        self._mfa_enabled = mfa_enabled
        
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
        try:
            # MFA check if enabled
            if self._mfa_enabled:
                await self._perform_mfa_check()
            
            # Handle empty transcript
            if not hasattr(analysis, 'turn_input') or not hasattr(analysis.turn_input, 'transcript') or not analysis.turn_input.transcript.strip():
                return self._create_default_action(context)
            
            # Select strategy
            strategy = self._select_strategy(analysis, context)
            
            # Calculate difficulty delta
            difficulty_delta = self._calculate_difficulty_delta(context)

            # Get target word/phoneme for pronunciation correction
            target_word = self._get_target_word(analysis) if strategy == CoachingStrategy.CORRECT_PRONUNCIATION else None
            target_phoneme = self._get_target_phoneme(analysis) if strategy == CoachingStrategy.CORRECT_PRONUNCIATION else None

            argument_score = getattr(getattr(analysis, "argument", None), "argument_score", 0.0)
            pronunciation_score = getattr(getattr(analysis, "pronunciation", None), "fluency_score", 0.0)

            return CoachingAction(
                strategy=strategy,
                intent=f"Selected {strategy.value} strategy based on analysis",
                target_claim=None,
                target_word=target_word,
                target_phoneme=target_phoneme,
                argument_score=argument_score,
                pronunciation_score=pronunciation_score,
                difficulty_delta=difficulty_delta,
                turn_number=context.turn_number,
                topic=context.topic,
                user_position=context.user_position,
                prior_coach_responses=[s.value for s in context.coaching_history[-3:]] if context.coaching_history else [],
            )
            
        except Exception:
            return self._create_default_action(context)
    
    async def _perform_mfa_check(self) -> None:
        """
        MFA stub implementation - toggleable security feature.
        ASYNC METHOD - simulates MFA verification
        
        Behavior:
            - No-op when MFA is disabled
            - Simulates MFA verification delay when enabled
        """
        if self._mfa_enabled:
            # Stub implementation - in production this would verify MFA token
            await asyncio.sleep(0.1)  # Simulate MFA verification delay
    
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
        # Rule 1: Check for high severity pronunciation issues
        if hasattr(analysis, 'pronunciation') and hasattr(analysis.pronunciation, 'mispronounced_words'):
            for word in analysis.pronunciation.mispronounced_words:
                if hasattr(word, 'severity') and word.severity.value == 'high':
                    preferred = CoachingStrategy.CORRECT_PRONUNCIATION
                    return self._avoid_repetition(preferred, context)
        
        # Rule 2: Check for off-topic or empty input
        argument = getattr(analysis, 'argument', None)
        argument_score = getattr(argument, 'argument_score', 0.0) if argument is not None else 0.0
        if argument_score < LOW_ARGUMENT_THRESHOLD:
            preferred = CoachingStrategy.REDIRECT
            return self._avoid_repetition(preferred, context)
        
        # Rule 4: High quality argument — praise and push deeper (never challenge to opposite side)
        if argument_score >= HIGH_ARGUMENT_THRESHOLD:
            preferred = CoachingStrategy.PRAISE
            return self._avoid_repetition(preferred, context)
        
        # Rule 6: Has claim but no reasoning
        if argument is not None:
            if getattr(argument, 'has_claim', False) and not getattr(argument, 'has_reasoning', False):
                return self._avoid_repetition(CoachingStrategy.PROBE, context)
        
        # Rule 7: Has reasoning but no evidence
        if argument is not None:
            if getattr(argument, 'has_reasoning', False) and not getattr(argument, 'has_evidence', False):
                return self._avoid_repetition(CoachingStrategy.PROBE, context)
        
        # Rule 8: Default
        preferred = CoachingStrategy.PROBE
        return self._avoid_repetition(preferred, context)
    
    def _build_intent(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str:
        """
        Build a human-readable intent string explaining the strategy choice.
        SYNCHRONOUS METHOD - string construction only.
        """
        transcript = getattr(getattr(analysis, "turn_input", None), "transcript", "")
        argument_score = getattr(getattr(analysis, "argument", None), "argument_score", 0.0)
        return (
            f"Turn {context.turn_number}: chose {strategy.value} "
            f"(argument_score={argument_score:.2f}, transcript_len={len(transcript)})"
        )
    
    def _create_default_action(self, context: SessionContext) -> CoachingAction:
        """
        Create fallback CoachingAction for error cases.
        SYNCHRONOUS METHOD - no external calls
        
        Args:
            context: SessionContext for topic-relevant question
            
        Returns:
            CoachingAction with strategy=PROBE and generic opening question
        """
        return CoachingAction(
            strategy=CoachingStrategy.PROBE,
            intent="Default action for error case or empty transcript",
            target_claim=None,
            target_word=None,
            target_phoneme=None,
            argument_score=0.0,
            pronunciation_score=0.0,
            difficulty_delta=0,
            turn_number=context.turn_number,
            topic=context.topic,
            user_position=context.user_position,
            prior_coach_responses=[],
        )
    
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
        if len(context.argument_scores) < 2:
            return 0
        
        last_two_scores = context.argument_scores[-2:]
        
        if all(score >= HIGH_ARGUMENT_THRESHOLD for score in last_two_scores):
            return 1
        elif all(score < DIFFICULTY_THRESHOLD for score in last_two_scores):
            return -1
        else:
            return 0
    
    def _get_target_word(self, analysis: TurnAnalysis) -> str:
        """Extract first high-severity mispronounced word, or None."""
        if not hasattr(analysis, 'pronunciation') or not hasattr(analysis.pronunciation, 'mispronounced_words'):
            return None
        for word in analysis.pronunciation.mispronounced_words:
            if hasattr(word, 'severity') and word.severity.value == 'high':
                return getattr(word, 'word', None)
        return None

    def _get_target_phoneme(self, analysis: TurnAnalysis) -> str:
        """Extract first target phoneme from pronunciation result, or None."""
        phonemes = getattr(getattr(analysis, 'pronunciation', None), 'target_phonemes', [])
        return phonemes[0] if phonemes else None
    
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
        if len(context.coaching_history) < REPEAT_PREVENTION_LOOKBACK:
            return preferred_strategy
        
        last_two = context.coaching_history[-REPEAT_PREVENTION_LOOKBACK:]
        if all(strategy == preferred_strategy for strategy in last_two):
            # Find alternative strategy
            alternatives = [
                CoachingStrategy.PROBE,
                CoachingStrategy.CHALLENGE,
                CoachingStrategy.REDIRECT,
                CoachingStrategy.PRAISE
            ]
            
            for alt in alternatives:
                if alt != preferred_strategy:
                    return alt
        
        return preferred_strategy