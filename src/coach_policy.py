import asyncio
import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from anthropic import AsyncAnthropic

# Import external types
try:
    from turn_analyzer import TurnAnalysis
except ImportError:
    # TODO: Add proper TurnAnalysis import when available
    class TurnAnalysis:
        pass


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
    target_word: str = ""
    difficulty_delta: int = 0


@dataclass
class SessionContext:
    session_id: str
    topic: str
    user_position: str  # "for" or "against"
    turn_number: int
    coaching_history: List[CoachingStrategy]
    argument_scores: List[float]


class CoachPolicyAgent:
    """
    AI debate coaching policy agent that decides coaching strategies and generates responses.
    """
    
    # Class constants
    MODEL_NAME = "claude-3-5-sonnet-20241022"
    API_TIMEOUT = 30.0
    DECIDE_TIMEOUT = 10.0
    MAX_TOKENS = 150
    
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

    def __init__(self, anthropic_api_key: str) -> None:
        """
        Initialize the CoachPolicyAgent with Anthropic API client.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude service
        """
        self._client = AsyncAnthropic(
            api_key=anthropic_api_key,
            timeout=self.API_TIMEOUT
        )

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
        try:
            return await asyncio.wait_for(
                self._decide_internal(analysis, context),
                timeout=self.DECIDE_TIMEOUT
            )
        except asyncio.TimeoutError:
            return self._create_default_action(context)
        except Exception:
            return self._create_default_action(context)

    async def _decide_internal(self, analysis: TurnAnalysis, context: SessionContext) -> CoachingAction:
        """Internal implementation of decide logic."""
        # Handle empty transcript
        if not hasattr(analysis, 'turn_input') or not hasattr(analysis.turn_input, 'transcript') or not analysis.turn_input.transcript.strip():
            return CoachingAction(
                strategy=CoachingStrategy.PROBE,
                response_text=self._get_opening_question(context),
                internal_reason="Empty transcript - opening question",
                difficulty_delta=0
            )
        
        # Select strategy
        strategy = self._select_strategy(analysis, context)
        
        # Generate response
        response_text = await self._generate_response(analysis, context, strategy)
        
        # Calculate difficulty delta
        difficulty_delta = self._calculate_difficulty_delta(context)
        
        # Get target word for pronunciation correction
        target_word = self._get_pronunciation_target(analysis) if strategy == CoachingStrategy.CORRECT_PRONUNCIATION else ""
        
        return CoachingAction(
            strategy=strategy,
            response_text=response_text,
            internal_reason=f"Selected {strategy.value} based on analysis",
            target_word=target_word,
            difficulty_delta=difficulty_delta
        )

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
        # Rule 1: Major mispronunciation
        if (hasattr(analysis, 'pronunciation') and 
            hasattr(analysis.pronunciation, 'mispronounced_words') and
            any(getattr(word, 'severity', None) == 'MAJOR' for word in analysis.pronunciation.mispronounced_words)):
            return CoachingStrategy.CORRECT_PRONUNCIATION
        
        # Rule 2: Off-topic or empty
        if self._is_off_topic(analysis):
            return CoachingStrategy.REDIRECT
        
        # Get argument score
        argument_score = getattr(analysis, 'argument_score', 0.0) if hasattr(analysis, 'argument_score') else 0.0
        
        # Get argument structure flags
        has_claim = getattr(analysis, 'has_claim', False) if hasattr(analysis, 'has_claim') else False
        has_reasoning = getattr(analysis, 'has_reasoning', False) if hasattr(analysis, 'has_reasoning') else False
        has_evidence = getattr(analysis, 'has_evidence', False) if hasattr(analysis, 'has_evidence') else False
        
        strategies_to_try = []
        
        # Rules 4-8: Build priority list
        if argument_score >= self.STRONG_ARGUMENT_THRESHOLD and context.turn_number > 2:
            strategies_to_try.append(CoachingStrategy.CHALLENGE)
        
        if argument_score >= self.STRONG_ARGUMENT_THRESHOLD:
            strategies_to_try.append(CoachingStrategy.PRAISE_AND_PUSH)
        
        if has_claim and not has_reasoning:
            strategies_to_try.append(CoachingStrategy.PROBE)
        
        if has_reasoning and not has_evidence:
            strategies_to_try.append(CoachingStrategy.PROBE)
        
        # Default
        strategies_to_try.append(CoachingStrategy.PROBE)
        
        # Rule 3: Skip if used in last 2 turns
        for strategy in strategies_to_try:
            if not self._should_skip_strategy(strategy, context):
                return strategy
        
        # Fallback to first strategy if all are skipped
        return strategies_to_try[0] if strategies_to_try else CoachingStrategy.PROBE

    async def _generate_response(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str:
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
        try:
            prompt = self._build_prompt(analysis, context, strategy)
            
            response = await self._client.messages.create(
                model=self.MODEL_NAME,
                max_tokens=self.MAX_TOKENS,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text if response.content else ""
            return self._strip_markdown(response_text)
            
        except Exception:
            return self._get_fallback_response(strategy, context)

    def _build_prompt(self, analysis: TurnAnalysis, context: SessionContext, strategy: CoachingStrategy) -> str:
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
        transcript = getattr(analysis.turn_input, 'transcript', '') if hasattr(analysis, 'turn_input') else ''
        
        base_prompt = f"""You are an AI debate partner coaching a Chinese L2 English learner. 

Debate topic: {context.topic}
Student's position: {context.user_position}
Student said: "{transcript}"

Recent coaching history: {self._get_recent_strategies(context, 3)}

Strategy: {strategy.value}

"""
        
        strategy_instructions = {
            CoachingStrategy.PROBE: "Ask a follow-up question to draw out more reasoning. Be curious about their thinking.",
            CoachingStrategy.CHALLENGE: "Push back on their claim with a counterargument. Take the opposing side clearly but not aggressively.",
            CoachingStrategy.REDIRECT: "Steer them back on topic. Ask a question that connects to the main debate.",
            CoachingStrategy.PRAISE_AND_PUSH: "Acknowledge their strong point then raise the bar with a deeper question.",
            CoachingStrategy.CORRECT_PRONUNCIATION: f"Model correct pronunciation of '{self._get_pronunciation_target(analysis)}' naturally in your response."
        }
        
        instruction = strategy_instructions.get(strategy, "Ask a follow-up question.")
        
        return base_prompt + f"""{instruction}

Respond in 1-3 sentences. Be conversational like a debate partner, not a teacher. No bullet points. No "Good job!" or "You need to improve." Engage with the argument itself."""

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
        if len(context.argument_scores) < 2:
            return 0
        
        last_two = context.argument_scores[-2:]
        
        if all(score >= self.STRONG_ARGUMENT_THRESHOLD for score in last_two):
            return 1
        elif all(score < self.WEAK_ARGUMENT_THRESHOLD for score in last_two):
            return -1
        else:
            return 0

    def _should_skip_strategy(self, strategy: CoachingStrategy, context: SessionContext) -> bool:
        """
        Check if strategy should be skipped due to recent usage.
        
        Args:
            strategy: CoachingStrategy to check
            context: SessionContext with coaching_history
            
        Returns:
            bool: True if strategy was used in last 2 turns
        """
        if len(context.coaching_history) < 2:
            return False
        
        return context.coaching_history[-2:] == [strategy, strategy]

    def _get_pronunciation_target(self, analysis: TurnAnalysis) -> str:
        """
        Extract target word for pronunciation correction.
        
        Args:
            analysis: TurnAnalysis object
            
        Returns:
            str: First major mispronounced word, or "" if none
        """
        if (hasattr(analysis, 'pronunciation') and 
            hasattr(analysis.pronunciation, 'mispronounced_words')):
            for word in analysis.pronunciation.mispronounced_words:
                if getattr(word, 'severity', None) == 'MAJOR':
                    return getattr(word, 'word', '')
        return ""

    def _create_default_action(self, context: SessionContext) -> CoachingAction:
        """
        Create safe fallback CoachingAction.
        
        Args:
            context: SessionContext for topic-relevant question
            
        Returns:
            CoachingAction with strategy=PROBE and generic opening question
        """
        return CoachingAction(
            strategy=CoachingStrategy.PROBE,
            response_text=self._get_opening_question(context),
            internal_reason="Default fallback action",
            difficulty_delta=0
        )

    def _strip_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from text.
        
        Args:
            text: Input text potentially containing markdown
            
        Returns:
            str: Text with markdown removed
        """
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'#{1,6}\s+', '', text)         # Headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        return text.strip()

    def _is_off_topic(self, analysis: TurnAnalysis) -> bool:
        """
        Determine if student response is off-topic.
        
        Args:
            analysis: TurnAnalysis object
            
        Returns:
            bool: True if argument_score < 0.1 or transcript is empty
        """
        if not hasattr(analysis, 'turn_input') or not hasattr(analysis.turn_input, 'transcript'):
            return True
        
        if not analysis.turn_input.transcript.strip():
            return True
        
        argument_score = getattr(analysis, 'argument_score', 0.0) if hasattr(analysis, 'argument_score') else 0.0
        return argument_score < self.OFF_TOPIC_THRESHOLD

    def _get_recent_strategies(self, context: SessionContext, count: int = 3) -> List[CoachingStrategy]:
        """
        Get most recent coaching strategies from history.
        
        Args:
            context: SessionContext with coaching_history
            count: Number of recent strategies to return
            
        Returns:
            List[CoachingStrategy]: Recent strategies (newest first)
        """
        return context.coaching_history[-count:] if len(context.coaching_history) >= count else context.coaching_history

    def _get_opening_question(self, context: SessionContext) -> str:
        """Get appropriate opening question based on user position."""
        return self.DEFAULT_OPENING_QUESTIONS.get(
            context.user_position,
            self.DEFAULT_OPENING_QUESTIONS["neutral"]
        )

    def _get_fallback_response(self, strategy: CoachingStrategy, context: SessionContext) -> str:
        """Get fallback response when API call fails."""
        fallbacks = {
            CoachingStrategy.