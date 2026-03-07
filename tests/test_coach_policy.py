import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List

# Import the classes we're testing (assuming they exist in coach_policy module)
from coach_policy import CoachPolicyAgent, CoachingStrategy, CoachingAction, SessionContext

# Mock data models that would come from other modules
@dataclass
class TurnInput:
    transcript: str
    timestamp: datetime
    audio_duration: float

@dataclass
class PronunciationIssue:
    word: str
    severity: str
    phoneme: str

@dataclass
class PronunciationAnalysis:
    mispronounced_words: List[PronunciationIssue]
    overall_score: float

@dataclass
class ArgumentAnalysis:
    has_claim: bool
    has_reasoning: bool
    has_evidence: bool
    argument_score: float
    is_on_topic: bool

@dataclass
class TurnAnalysis:
    turn_input: TurnInput
    pronunciation: PronunciationAnalysis
    argument: ArgumentAnalysis

class TestCoachPolicyAgent:
    
    @pytest.fixture
    def mock_anthropic_client(self):
        with patch('coach_policy.AsyncAnthropic') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock the messages.create response
            mock_response = AsyncMock()
            mock_response.content = [Mock(text="This is a test response from Claude.")]
            mock_client.messages.create.return_value = mock_response
            
            yield mock_client

    @pytest.fixture
    def agent(self, mock_anthropic_client):
        return CoachPolicyAgent("test-api-key")

    @pytest.fixture
    def sample_context(self):
        return SessionContext(
            session_id="test-session-123",
            topic="Should social media be banned for teenagers?",
            user_position="for",
            turn_number=3,
            coaching_history=[CoachingStrategy.PROBE, CoachingStrategy.CHALLENGE],
            argument_scores=[0.5, 0.8]
        )

    @pytest.fixture
    def sample_turn_analysis(self):
        return TurnAnalysis(
            turn_input=TurnInput(
                transcript="I think social media is bad for teenagers because it causes addiction.",
                timestamp=datetime.now(),
                audio_duration=5.2
            ),
            pronunciation=PronunciationAnalysis(
                mispronounced_words=[],
                overall_score=0.85
            ),
            argument=ArgumentAnalysis(
                has_claim=True,
                has_reasoning=True,
                has_evidence=False,
                argument_score=0.6,
                is_on_topic=True
            )
        )

    def test_init(self, mock_anthropic_client):
        agent = CoachPolicyAgent("test-api-key")
        assert agent is not None
        # Verify AsyncAnthropic was called with correct parameters
        mock_anthropic_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_decide_happy_path(self, agent, sample_turn_analysis, sample_context, mock_anthropic_client):
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.content = [Mock(text="What evidence supports your claim about addiction?")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = await agent.decide(sample_turn_analysis, sample_context)

        assert isinstance(result, CoachingAction)
        assert result.strategy == CoachingStrategy.PROBE  # has_reasoning=True, has_evidence=False
        assert len(result.response_text) > 0
        assert result.internal_reason != ""
        assert result.target_word == ""  # Not a pronunciation correction
        assert result.difficulty_delta == 0  # Not meeting criteria for +1 or -1

    @pytest.mark.asyncio
    async def test_decide_empty_transcript_returns_probe(self, agent, sample_context, mock_anthropic_client):
        empty_analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="", timestamp=datetime.now(), audio_duration=0.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=False, has_reasoning=False, has_evidence=False, argument_score=0.0, is_on_topic=False)
        )

        mock_response = AsyncMock()
        mock_response.content = [Mock(text="What's your main argument about social media for teenagers?")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = await agent.decide(empty_analysis, sample_context)

        assert result.strategy == CoachingStrategy.PROBE
        assert len(result.response_text) > 0

    @pytest.mark.asyncio
    async def test_decide_timeout_returns_default(self, agent, sample_turn_analysis, sample_context, mock_anthropic_client):
        # Make the API call timeout
        mock_anthropic_client.messages.create.side_effect = asyncio.TimeoutError()

        result = await agent.decide(sample_turn_analysis, sample_context)

        assert isinstance(result, CoachingAction)
        assert result.strategy == CoachingStrategy.PROBE
        assert len(result.response_text) > 0

    @pytest.mark.asyncio
    async def test_decide_api_error_returns_default(self, agent, sample_turn_analysis, sample_context, mock_anthropic_client):
        # Make the API call raise an exception
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        result = await agent.decide(sample_turn_analysis, sample_context)

        assert isinstance(result, CoachingAction)
        assert result.strategy == CoachingStrategy.PROBE
        assert len(result.response_text) > 0

    @pytest.mark.asyncio
    async def test_decide_completes_within_timeout(self, agent, sample_turn_analysis, sample_context, mock_anthropic_client):
        start_time = asyncio.get_event_loop().time()
        
        mock_response = AsyncMock()
        mock_response.content = [Mock(text="Test response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = await agent.decide(sample_turn_analysis, sample_context)
        
        end_time = asyncio.get_event_loop().time()
        assert (end_time - start_time) < 10.0  # Must complete within 10 seconds

    def test_select_strategy_pronunciation_major_severity(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="I think social media is bad", timestamp=datetime.now(), audio_duration=3.0),
            pronunciation=PronunciationAnalysis(
                mispronounced_words=[PronunciationIssue("social", "MAJOR", "s")],
                overall_score=0.5
            ),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=True, has_evidence=True, argument_score=0.8, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.CORRECT_PRONUNCIATION

    def test_select_strategy_redirect_low_argument_score(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="I like pizza", timestamp=datetime.now(), audio_duration=2.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=False, has_reasoning=False, has_evidence=False, argument_score=0.05, is_on_topic=False)
        )

        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.REDIRECT

    def test_select_strategy_redirect_empty_transcript(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="", timestamp=datetime.now(), audio_duration=0.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=False, has_reasoning=False, has_evidence=False, argument_score=0.0, is_on_topic=False)
        )

        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.REDIRECT

    def test_select_strategy_avoids_repeating_same_strategy(self, agent):
        context = SessionContext(
            session_id="test",
            topic="Test topic",
            user_position="for",
            turn_number=4,
            coaching_history=[CoachingStrategy.PROBE, CoachingStrategy.PROBE],  # Last 2 are same
            argument_scores=[0.5, 0.6]
        )
        
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="I think this because reasons", timestamp=datetime.now(), audio_duration=3.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=True, has_evidence=False, argument_score=0.6, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, context)
        # Should not be PROBE since last 2 were PROBE, and conditions don't match other high-priority strategies
        assert strategy != CoachingStrategy.PROBE

    def test_select_strategy_challenge_high_score_after_turn_2(self, agent):
        context = SessionContext(
            session_id="test",
            topic="Test topic",
            user_position="for",
            turn_number=5,
            coaching_history=[CoachingStrategy.PROBE, CoachingStrategy.PRAISE_AND_PUSH],
            argument_scores=[0.6, 0.8]
        )
        
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="Strong argument with evidence", timestamp=datetime.now(), audio_duration=4.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=True, has_evidence=True, argument_score=0.8, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, context)
        assert strategy == CoachingStrategy.CHALLENGE

    def test_select_strategy_praise_and_push_high_score_early_turns(self, agent):
        context = SessionContext(
            session_id="test",
            topic="Test topic",
            user_position="for",
            turn_number=2,
            coaching_history=[CoachingStrategy.PROBE],
            argument_scores=[0.5]
        )
        
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="Good argument", timestamp=datetime.now(), audio_duration=3.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=True, has_evidence=True, argument_score=0.8, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, context)
        assert strategy == CoachingStrategy.PRAISE_AND_PUSH

    def test_select_strategy_probe_has_claim_no_reasoning(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="Social media is bad", timestamp=datetime.now(), audio_duration=2.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=False, has_evidence=False, argument_score=0.4, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PROBE

    def test_select_strategy_probe_has_reasoning_no_evidence(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="Social media is bad because it's addictive", timestamp=datetime.now(), audio_duration=3.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=True, has_evidence=False, argument_score=0.6, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PROBE

    def test_select_strategy_default_probe(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="Some random statement", timestamp=datetime.now(), audio_duration=2.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=False, has_reasoning=False, has_evidence=False, argument_score=0.3, is_on_topic=True)
        )

        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PROBE

    def test_select_strategy_is_deterministic(self, agent, sample_context):
        analysis = TurnAnalysis(
            turn_input=TurnInput(transcript="Consistent input", timestamp=datetime.now(), audio_duration=3.0),
            pronunciation=PronunciationAnalysis(mispronounced_words=[], overall_score=1.0),
            argument=ArgumentAnalysis(has_claim=True, has_reasoning=False, has_evidence=False, argument_score=0.5, is_on_topic=True)
        )

        # Call multiple times with same inputs
        strategy1 = agent._select_strategy(analysis, sample_context)
        strategy2 = agent._select_strategy(analysis, sample_context)
        strategy3 = agent._select_strategy(analysis, sample_context)
        
        assert strategy1 == strategy2 == strategy3

    @pytest.mark.asyncio
    async def test_generate_response_strips_markdown(self, agent, sample_turn_analysis, sample_context, mock_anthropic_client):
        # Mock response with markdown
        mock_response = AsyncMock()
        mock_response.content = [Mock(text="**Bold text** and *italic* and `code` should be stripped.")]
        mock_anthropic_client.messages.create.return_value = mock_response

        response = await agent._generate_response(sample_turn_analysis, sample_context, CoachingStrategy.PROBE)
        
        assert "**" not in response
        assert "*" not in response
        assert "`" not in response
        assert "Bold text" in response

    @pytest.mark.asyncio
    async def test_generate_response_calls_anthropic_with_correct_params(self, agent, sample_turn_analysis, sample_context, mock_anthropic_client):
        mock_response = AsyncMock()
        mock_response.content = [Mock(text="Test response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        await agent._generate_response(sample_turn_analysis, sample_context, CoachingStrategy.CHALLENGE)

        # Verify the API was called
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        
        # Check that required parameters are present
        assert call_args.kwargs['model'] == 'claude-3-5-sonnet-20241022'
        assert call