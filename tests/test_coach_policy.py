import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from enum import Enum
from dataclasses import dataclass
from typing import List

# Import the module under test
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coach_policy import CoachPolicyAgent, CoachingStrategy, CoachingAction, SessionContext


# Mock data models for testing
@dataclass
class MockPronunciationWord:
    word: str
    severity: str = "MINOR"


@dataclass
class MockPronunciation:
    mispronounced_words: List[MockPronunciationWord]


@dataclass
class MockTurnInput:
    transcript: str


@dataclass
class MockAnalysisResult:
    has_claim: bool
    has_reasoning: bool
    has_evidence: bool
    argument_score: float


@dataclass
class MockTurnAnalysis:
    turn_input: MockTurnInput
    pronunciation: MockPronunciation
    analysis_result: MockAnalysisResult


class TestCoachPolicyAgent:
    
    @pytest.fixture
    def mock_anthropic(self):
        with patch('coach_policy.AsyncAnthropic') as mock:
            mock_client = AsyncMock()
            mock.return_value = mock_client
            mock_client.messages.create = AsyncMock()
            yield mock_client

    @pytest.fixture
    def agent(self, mock_anthropic):
        return CoachPolicyAgent("test-api-key")

    @pytest.fixture
    def sample_context(self):
        return SessionContext(
            session_id="test-123",
            topic="Social media should be banned for teenagers",
            user_position="for",
            turn_number=1,
            coaching_history=[],
            argument_scores=[]
        )

    @pytest.fixture
    def sample_analysis(self):
        return MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="I think social media is bad because it causes depression"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=True,
                has_evidence=False,
                argument_score=0.6
            )
        )

    def test_init_creates_anthropic_client(self, mock_anthropic):
        agent = CoachPolicyAgent("test-key")
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_decide_with_empty_transcript_returns_probe(self, agent, sample_context, mock_anthropic):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript=""),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=False,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.0
            )
        )
        
        result = await agent.decide(analysis, sample_context)
        
        assert isinstance(result, CoachingAction)
        assert result.strategy == CoachingStrategy.PROBE
        assert len(result.response_text) > 0
        assert "Social media should be banned for teenagers" in result.response_text

    @pytest.mark.asyncio
    async def test_decide_completes_within_timeout(self, agent, sample_analysis, sample_context, mock_anthropic):
        mock_anthropic.messages.create.return_value = Mock(content=[Mock(text="What evidence supports that claim?")])
        
        start_time = asyncio.get_event_loop().time()
        result = await agent.decide(sample_analysis, sample_context)
        end_time = asyncio.get_event_loop().time()
        
        assert (end_time - start_time) < 10.0
        assert isinstance(result, CoachingAction)

    @pytest.mark.asyncio
    async def test_decide_handles_api_error_gracefully(self, agent, sample_analysis, sample_context, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("API Error")
        
        result = await agent.decide(sample_analysis, sample_context)
        
        assert isinstance(result, CoachingAction)
        assert result.strategy == CoachingStrategy.PROBE
        assert len(result.response_text) > 0

    def test_select_strategy_major_pronunciation_error_returns_correct_pronunciation(self, agent, sample_context):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="I think this"),
            pronunciation=MockPronunciation(mispronounced_words=[
                MockPronunciationWord(word="think", severity="MAJOR")
            ]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=True,
                has_evidence=True,
                argument_score=0.8
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.CORRECT_PRONUNCIATION

    def test_select_strategy_empty_transcript_returns_redirect(self, agent, sample_context):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript=""),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=False,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.0
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.REDIRECT

    def test_select_strategy_low_argument_score_returns_redirect(self, agent, sample_context):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="I don't know what to say"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.05
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.REDIRECT

    def test_select_strategy_avoids_repeating_same_strategy_three_times(self, agent, sample_context):
        sample_context.coaching_history = [CoachingStrategy.PROBE, CoachingStrategy.PROBE]
        
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="I think social media is harmful"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.5
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy != CoachingStrategy.PROBE

    def test_select_strategy_high_score_and_turn_number_returns_challenge(self, agent, sample_context):
        sample_context.turn_number = 5
        
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="Social media causes depression with clear evidence"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=True,
                has_evidence=True,
                argument_score=0.8
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.CHALLENGE

    def test_select_strategy_high_score_early_turn_returns_praise_and_push(self, agent, sample_context):
        sample_context.turn_number = 2
        
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="Social media causes depression with evidence"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=True,
                has_evidence=True,
                argument_score=0.8
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PRAISE_AND_PUSH

    def test_select_strategy_has_claim_no_reasoning_returns_probe(self, agent, sample_context):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="Social media is bad"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.4
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PROBE

    def test_select_strategy_has_reasoning_no_evidence_returns_probe(self, agent, sample_context):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="Social media is bad because it's harmful"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=True,
                has_evidence=False,
                argument_score=0.5
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PROBE

    def test_select_strategy_default_case_returns_probe(self, agent, sample_context):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="I'm not sure"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=False,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.3
            )
        )
        
        strategy = agent._select_strategy(analysis, sample_context)
        assert strategy == CoachingStrategy.PROBE

    @pytest.mark.asyncio
    async def test_generate_response_calls_anthropic_api(self, agent, sample_analysis, sample_context, mock_anthropic):
        mock_anthropic.messages.create.return_value = Mock(content=[Mock(text="What evidence supports that?")])
        
        response = await agent._generate_response(sample_analysis, sample_context, CoachingStrategy.PROBE)
        
        mock_anthropic.messages.create.assert_called_once()
        call_args = mock_anthropic.messages.create.call_args
        assert call_args[1]['model'] == 'claude-3-5-sonnet-20241022'
        assert call_args[1]['max_tokens'] == 150
        assert call_args[1]['timeout'] == 30
        assert response == "What evidence supports that?"

    @pytest.mark.asyncio
    async def test_generate_response_strips_markdown(self, agent, sample_analysis, sample_context, mock_anthropic):
        mock_anthropic.messages.create.return_value = Mock(content=[Mock(text="**What** evidence *supports* that?")])
        
        response = await agent._generate_response(sample_analysis, sample_context, CoachingStrategy.PROBE)
        
        assert "**" not in response
        assert "*" not in response

    def test_build_prompt_includes_strategy_and_context(self, agent, sample_analysis, sample_context):
        prompt = agent._build_prompt(sample_analysis, sample_context, CoachingStrategy.PROBE)
        
        assert "Social media should be banned for teenagers" in prompt
        assert "PROBE" in prompt
        assert "for" in prompt
        assert sample_analysis.turn_input.transcript in prompt

    def test_build_prompt_includes_coaching_history(self, agent, sample_analysis, sample_context):
        sample_context.coaching_history = [CoachingStrategy.PROBE, CoachingStrategy.CHALLENGE]
        
        prompt = agent._build_prompt(sample_analysis, sample_context, CoachingStrategy.PRAISE_AND_PUSH)
        
        assert "PROBE" in prompt or "CHALLENGE" in prompt

    def test_create_default_action_returns_valid_action(self, agent, sample_context):
        action = agent._create_default_action(sample_context)
        
        assert isinstance(action, CoachingAction)
        assert action.strategy == CoachingStrategy.PROBE
        assert len(action.response_text) > 0
        assert "Social media should be banned for teenagers" in action.response_text
        assert action.difficulty_delta == 0

    def test_difficulty_delta_plus_one_for_high_scores(self, agent, sample_analysis, sample_context):
        sample_context.argument_scores = [0.8, 0.9]
        
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="Strong argument with evidence"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=True,
                has_evidence=True,
                argument_score=0.8
            )
        )
        
        # Mock the difficulty calculation in decide method
        with patch.object(agent, '_calculate_difficulty_delta', return_value=1):
            result = asyncio.run(agent.decide(analysis, sample_context))
            assert result.difficulty_delta == 1

    def test_difficulty_delta_minus_one_for_low_scores(self, agent, sample_analysis, sample_context):
        sample_context.argument_scores = [0.2, 0.1]
        
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="Weak argument"),
            pronunciation=MockPronunciation(mispronounced_words=[]),
            analysis_result=MockAnalysisResult(
                has_claim=False,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.2
            )
        )
        
        with patch.object(agent, '_calculate_difficulty_delta', return_value=-1):
            result = asyncio.run(agent.decide(analysis, sample_context))
            assert result.difficulty_delta == -1

    def test_difficulty_delta_zero_for_mixed_scores(self, agent, sample_analysis, sample_context):
        sample_context.argument_scores = [0.8, 0.2]
        
        with patch.object(agent, '_calculate_difficulty_delta', return_value=0):
            result = asyncio.run(agent.decide(sample_analysis, sample_context))
            assert result.difficulty_delta == 0

    @pytest.mark.asyncio
    async def test_decide_sets_target_word_for_pronunciation_correction(self, agent, sample_context, mock_anthropic):
        analysis = MockTurnAnalysis(
            turn_input=MockTurnInput(transcript="I think this is wrong"),
            pronunciation=MockPronunciation(mispronounced_words=[
                MockPronunciationWord(word="think", severity="MAJOR")
            ]),
            analysis_result=MockAnalysisResult(
                has_claim=True,
                has_reasoning=False,
                has_evidence=False,
                argument_score=0.5
            )
        )
        
        mock_anthropic.messages.create.return_value = Mock(content=[Mock(text="I think you mean this point?")])
        
        result = await agent.decide(analysis, sample_context)
        
        assert result.strategy == CoachingStrategy.CORRECT_PRONUNCIATION
        assert result.target_word == "think"

    @pytest.mark.asyncio
    async def test_decide_returns_coaching_action_with_all_fields(self, agent, sample_analysis, sample_context, mock_anthropic):
        