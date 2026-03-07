from anthropic import AsyncAnthropic
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import asyncio
import re

# Constants
CLAUDE_MODEL = "claude-sonnet-4-5"
API_TIMEOUT_SECONDS = 30
DECIDE_TIMEOUT_SECONDS = 10
LOW_ARGUMENT_THRESHOLD = 0.1
HIGH_ARGUMENT_THRESHOLD = 0.7
DIFFICULTY_THRESHOLD = 0.3
REPEAT_PREVENTION_LOOKBACK = 2
MAX_RESPONSE_LENGTH = 500

# Import TurnAnalysis type from design spec
from typing import Any
TurnAnalysis = Any  # Placeholder until actual TurnAnalysis is available

class CoachingStrategy(Enum):
    PROBE = "probe"
    CHALLENGE = "challenge"
    REDIRECT = "redirect"
    PRAISE_AND_PUSH = "praise_and_push"
    CORRECT_PRONUNCIATION = "correct_pronunciation"

@dataclass
class CoachingAction:
    strategy: CoachingStrategy
    response_text: str
    internal_reason: str
    target_word: str
    difficulty_delta: int

@dataclass
class SessionContext:
    session_id: str
    topic: str
    user_position: str
    turn_number: int
    coaching_history: List[CoachingStrategy]
    argument_scores: List[float]

class CoachPolicyAgent:
    def __init__(self, anthropic_api_key: str, mfa_enabled: bool = False) -> None:
        """
        Initialize the coach policy agent with Anthropic client.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude access
            mfa_enabled: Whether MFA is enabled (toggleable)
        """
        self._client = AsyncAnthropic(api_key=anthropic_api_key)
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
            
            # Generate response with timeout
            response_task = self._generate_response(analysis, context, strategy)
            response_text = await asyncio.wait_for(response_task, timeout=DECIDE_TIMEOUT_SECONDS)
            
            # Calculate difficulty delta
            difficulty_delta = self._calculate_difficulty_delta(context)
            
            # Get target word for pronunciation correction
            target_word = self._get_target_word(analysis) if strategy == CoachingStrategy.CORRECT_PRONUNCIATION else ""
            
            # Build internal reason
            internal_reason = f"Selected {strategy.value} strategy based on analysis"
            
            return CoachingAction(
                strategy=strategy,
                response_text=response_text,
                internal_reason=internal_reason,
                target_word=target_word,
                difficulty_delta=difficulty_delta
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
        # Rule 1: Check for major pronunciation issues
        if hasattr(analysis, 'pronunciation') and hasattr(analysis.pronunciation, 'mispronounced_words'):
            for word in analysis.pronunciation.mispronounced_words:
                if hasattr(word, 'severity') and word.severity == 'MAJOR':
                    preferred = CoachingStrategy.CORRECT_PRONUNCIATION
                    return self._avoid_repetition(preferred, context)
        
        # Rule 2: Check for off-topic or empty input
        argument = getattr(analysis, 'argument', None)
        argument_score = getattr(argument, 'argument_score', 0.0) if argument is not None else 0.0
        if argument_score < LOW_ARGUMENT_THRESHOLD:
            preferred = CoachingStrategy.REDIRECT
            return self._avoid_repetition(preferred, context)
        
        # Rule 4: High quality argument with experience
        if argument_score >= HIGH_ARGUMENT_THRESHOLD and context.turn_number > 2:
            preferred = CoachingStrategy.CHALLENGE
            result = self._avoid_repetition(preferred, context)
            return result
        
        # Rule 5: High quality argument
        if argument_score >= HIGH_ARGUMENT_THRESHOLD:
            preferred = CoachingStrategy.PRAISE_AND_PUSH
            result = self._avoid_repetition(preferred, context)
            return result
        
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
        try:
            prompt = self._build_prompt(analysis, context, strategy)
            
            response = await self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_RESPONSE_LENGTH,
                timeout=API_TIMEOUT_SECONDS,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text
            response_text = re.sub(r"```[a-z]*\n?|```", "", response_text).strip()
            return response_text
            
        except Exception:
            return self._get_fallback_response(strategy, context, analysis)
    
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
        transcript = getattr(analysis.turn_input, 'transcript', '') if hasattr(analysis, 'turn_input') else ''
        
        base_prompt = f"""You are debating with a Chinese student learning English. The topic is: "{context.topic}"
They are arguing {context.user_position}. Their last statement: "{transcript}"

Your role is to be their debate PARTNER, not teacher. Never say "Good job!" or "You need to improve."
Engage with their argument naturally."""
        
        strategy_instructions = {
            CoachingStrategy.PROBE: "Ask a follow-up question to draw out more reasoning from their argument.",
            CoachingStrategy.CHALLENGE: "Push back on their claim with a reasonable counterargument from the opposing side.",
            CoachingStrategy.REDIRECT: "Steer them back on topic with a relevant question about the main debate issue.",
            CoachingStrategy.PRAISE_AND_PUSH: "Acknowledge their strong point briefly, then raise the bar with a tougher question.",
            CoachingStrategy.CORRECT_PRONUNCIATION: "Model correct pronunciation by naturally using the mispronounced word in your response."
        }
        
        strategy_instruction = strategy_instructions.get(strategy, "Ask a thoughtful follow-up question.")
        
        recent_history = ""
        if len(context.coaching_history) > 0:
            recent_strategies = context.coaching_history[-3:]
            recent_history = f"\nRecent coaching approaches used: {[s.value for s in recent_strategies]}. Vary your approach."
        
        return f"""{base_prompt}

Strategy: {strategy_instruction}{recent_history}

Respond in 1-3 sentences. Be conversational and natural. No bullet points or markdown formatting."""
    
    def _create_default_action(self, context: SessionContext) -> CoachingAction:
        """
        Create fallback CoachingAction for error cases.
        SYNCHRONOUS METHOD - no external calls
        
        Args:
            context: SessionContext for topic-relevant question
            
        Returns:
            CoachingAction with strategy=PROBE and generic opening question
        """
        response_text = f"What's your main argument about {context.topic}?"
        
        return CoachingAction(
            strategy=CoachingStrategy.PROBE,
            response_text=response_text,
            internal_reason="Default action for error case or empty transcript",
            target_word="",
            difficulty_delta=0
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
        """
        Extract target word for pronunciation correction.
        SYNCHRONOUS METHOD - data extraction only
        
        Args:
            analysis: TurnAnalysis with pronunciation data
            
        Returns:
            First mispronounced word with MAJOR severity, or "" if none
        """
        if not hasattr(analysis, 'pronunciation') or not hasattr(analysis.pronunciation, 'mispronounced_words'):
            return ""
        
        for word in analysis.pronunciation.mispronounced_words:
            if hasattr(word, 'severity') and word.severity == 'MAJOR':
                return getattr(word, 'word', "")
        
        return ""
    
    def _strip_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from response text.
        SYNCHRONOUS METHOD - regex processing
        
        Args:
            text: Raw response from Claude
            
        Returns:
            Clean text with markdown removed
        """
        # Remove bold and italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove bullet points
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Remove numbered lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
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
                CoachingStrategy.PRAISE_AND_PUSH
            ]
            
            for alt in alternatives:
                if alt != preferred_strategy:
                    return alt
        
        return preferred_strategy
    
    def _get_fallback_response(self, strategy: CoachingStrategy, context: SessionContext, analysis: TurnAnalysis = None) -> str:
        """
        Get fallback response when Claude API fails.
        SYNCHRONOUS METHOD - string generation only
        
        Args:
            strategy: Selected coaching strategy
            context: Session context for topic reference
            analysis: TurnAnalysis object for additional context
            
        Returns:
            Appropriate fallback response text
        """
        fallback_responses = {
            CoachingStrategy.PROBE: f"What's your main argument about {context.topic}?",
            CoachingStrategy.CHALLENGE: f"That's an interesting point — but what would someone who disagrees say?",
            CoachingStrategy.REDIRECT: f"Let's refocus — what's your position on {context.topic}?",
            CoachingStrategy.PRAISE_AND_PUSH: f"You're making progress. Can you back that up with a specific example?",
            CoachingStrategy.CORRECT_PRONUNCIATION: f"Could you say that again? I want to make sure I understood you.",
        }
        return fallback_responses.get(strategy, f"What do you think about {context.topic}?")