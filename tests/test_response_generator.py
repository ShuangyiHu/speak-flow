import pytest
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

from response_generator import (
    ResponseGenerator,
    CoachingStrategy,
    ToneMode,
    CoachingAction,
    ResponseRequest,
    GeneratedResponse,
    STRATEGY_TONE_MAPPING,
    FALLBACK_RESPONSES,
    MAX_RESPONSE_WORDS,
    ESTIMATED_WORDS_PER_SECOND,
    MAX_RESPONSE_TIME_SECONDS
)


@pytest.fixture
def sample_coaching_action():
    return CoachingAction(
        strategy=CoachingStrategy.CLARIFY_POSITION,
        target_skill="critical thinking",
        confidence_score=0.8
    )


@pytest.fixture
def sample_request(sample_coaching_action):
    return ResponseRequest(
        coaching_action=sample_coaching_action,
        topic="climate change",
        user_position="renewable energy is too expensive",
        prior_responses=["What makes you think that?", "Can you be more specific?"],
        turn_number=3
    )


@pytest.fixture
def mock_anthropic_client():
    with patch('response_generator.AsyncAnthropic') as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        
        # Mock the messages.create method
        mock_message = AsyncMock()
        mock_message.content = [AsyncMock()]
        mock_message.content[0].text = '{"text": "What specific costs concern you most?", "follow_up_prompt": "Can you provide examples?"}'
        
        client_instance.messages.create = AsyncMock(return_value=mock_message)
        yield client_instance


class TestResponseGeneratorInit:
    def test_init_with_default_env_vars(self, mock_anthropic_client):
        with patch.dict(os.environ, {}, clear=True):
            generator = ResponseGenerator()
            assert generator._model == "claude-sonnet-4-5"
    
    def test_init_with_custom_model(self, mock_anthropic_client):
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-3-opus"}, clear=True):
            generator = ResponseGenerator()
            assert generator._model == "claude-3-opus"
    
    def test_init_creates_anthropic_client(self, mock_anthropic_client):
        with patch('response_generator.AsyncAnthropic') as mock_client:
            ResponseGenerator()
            mock_client.assert_called_once_with(
                api_key=None,
                timeout=10.0
            )


class TestGenerateResponseHappyPath:
    @pytest.mark.asyncio
    async def test_successful_generation(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthropic_client
        
        response = await generator.generate_response(sample_request)
        
        assert isinstance(response, GeneratedResponse)
        assert response.text == "What specific costs concern you most?"
        assert response.tone == ToneMode.SOCRATIC
        assert response.follow_up_prompt == "Can you provide examples?"
        assert response.estimated_speaking_seconds > 0
    
    @pytest.mark.asyncio
    async def test_response_within_time_limit(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthropic_client
        
        start_time = asyncio.get_event_loop().time()
        await generator.generate_response(sample_request)
        end_time = asyncio.get_event_loop().time()
        
        assert end_time - start_time < MAX_RESPONSE_TIME_SECONDS
    
    @pytest.mark.asyncio
    async def test_tone_mapping_for_all_strategies(self, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthropic_client
        
        for strategy in CoachingStrategy:
            action = CoachingAction(strategy, "test_skill", 0.5)
            request = ResponseRequest(
                coaching_action=action,
                topic="test topic",
                user_position="test position",
                prior_responses=[],
                turn_number=1
            )
            
            response = await generator.generate_response(request)
            expected_tone = STRATEGY_TONE_MAPPING[strategy]
            assert response.tone == expected_tone


class TestGenerateResponseEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_request_raises_value_error(self, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthropic_client
        
        invalid_request = ResponseRequest(
            coaching_action=None,
            topic="test",
            user_position="test",
            prior_responses=[],
            turn_number=1
        )
        
        with pytest.raises(ValueError, match="Request must include coaching_action"):
            await generator.generate_response(invalid_request)
    
    @pytest.mark.asyncio
    async def test_api_timeout_returns_fallback(self, sample_request):
        generator = ResponseGenerator()
        
        # Mock client that times out
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = asyncio.TimeoutError()
        generator._client = mock_client
        
        response = await generator.generate_response(sample_request)
        
        # Should return fallback response
        expected_fallback = FALLBACK_RESPONSES[CoachingStrategy.CLARIFY_POSITION]
        assert response.text == expected_fallback
        assert response.tone == ToneMode.SOCRATIC
    
    @pytest.mark.asyncio
    async def test_api_exception_returns_fallback(self, sample_request):
        generator = ResponseGenerator()
        
        # Mock client that raises exception
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        generator._client = mock_client
        
        response = await generator.generate_response(sample_request)
        
        # Should return fallback response
        expected_fallback = FALLBACK_RESPONSES[CoachingStrategy.CLARIFY_POSITION]
        assert response.text == expected_fallback
    
    @pytest.mark.asyncio
    async def test_empty_response_returns_fallback(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        
        # Mock empty response
        mock_message = AsyncMock()
        mock_message.content = [AsyncMock()]
        mock_message.content[0].text = '{"text": "", "follow_up_prompt": null}'
        mock_anthropic_client.messages.create.return_value = mock_message
        generator._client = mock_anthropic_client
        
        response = await generator.generate_response(sample_request)
        
        expected_fallback = FALLBACK_RESPONSES[CoachingStrategy.CLARIFY_POSITION]
        assert response.text == expected_fallback
    
    @pytest.mark.asyncio
    async def test_repetitive_response_returns_fallback(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        
        # Mock response that's identical to prior response
        mock_message = AsyncMock()
        mock_message.content = [AsyncMock()]
        mock_message.content[0].text = '{"text": "What makes you think that?", "follow_up_prompt": null}'
        mock_anthropic_client.messages.create.return_value = mock_message
        generator._client = mock_anthropic_client
        
        response = await generator.generate_response(sample_request)
        
        expected_fallback = FALLBACK_RESPONSES[CoachingStrategy.CLARIFY_POSITION]
        assert response.text == expected_fallback


class TestResponseConstraints:
    @pytest.mark.asyncio
    async def test_response_word_count_under_limit(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthropic_client
        
        response = await generator.generate_response(sample_request)
        word_count = len(response.text.split())
        assert word_count <= MAX_RESPONSE_WORDS
    
    @pytest.mark.asyncio
    async def test_response_is_one_to_two_sentences(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthropic_client
        
        response = await generator.generate_response(sample_request)
        sentence_count = response.text.count('.') + response.text.count('!') + response.text.count('?')
        assert 1 <= sentence_count <= 2


class TestRepetitionDetection:
    def test_check_repetition_identical_text(self):
        generator = ResponseGenerator()
        prior_responses = ["What do you think?", "Can you explain?"]
        new_text = "What do you think?"
        
        assert generator._check_repetition(new_text, prior_responses) is True
    
    def test_check_repetition_similar_text(self):
        generator = ResponseGenerator()
        prior_responses = ["What do you think about this?"]
        new_text = "What do you think about that?"
        
        assert generator._check_repetition(new_text, prior_responses) is True
    
    def test_check_repetition_different_text(self):
        generator = ResponseGenerator()
        prior_responses = ["What do you think?", "Can you explain?"]
        new_text = "Tell me more about your reasoning."
        
        assert generator._check_repetition(new_text, prior_responses) is False
    
    def test_check_repetition_empty_prior_responses(self):
        generator = ResponseGenerator()
        prior_responses = []
        new_text = "What do you think?"
        
        assert generator._check_repetition(new_text, prior_responses) is False
    
    def test_check_repetition_empty_new_text(self):
        generator = ResponseGenerator()
        prior_responses = ["What do you think?"]
        new_text = ""
        
        assert generator._check_repetition(new_text, prior_responses) is True


class TestStrategyToneMapping:
    def test_map_strategy_to_tone_all_strategies(self):
        generator = ResponseGenerator()
        
        for strategy, expected_tone in STRATEGY_TONE_MAPPING.items():
            actual_tone = generator._map_strategy_to_tone(strategy)
            assert actual_tone == expected_tone
    
    def test_map_strategy_to_tone_socratic_strategies(self):
        generator = ResponseGenerator()
        socratic_strategies = [
            CoachingStrategy.CLARIFY_POSITION,
            CoachingStrategy.STRENGTHEN_ARGUMENT,
            CoachingStrategy.EXPLORE_IMPLICATIONS,
            CoachingStrategy.FIND_EVIDENCE
        ]
        
        for strategy in socratic_strategies:
            tone = generator._map_strategy_to_tone(strategy)
            assert tone == ToneMode.SOCRATIC
    
    def test_map_strategy_to_tone_challenging_strategy(self):
        generator = ResponseGenerator()
        tone = generator._map_strategy_to_tone(CoachingStrategy.COUNTER_ARGUMENT)
        assert tone == ToneMode.CHALLENGING
    
    def test_map_strategy_to_tone_affirming_strategy(self):
        generator = ResponseGenerator()
        tone = generator._map_strategy_to_tone(CoachingStrategy.ACKNOWLEDGE_GOOD_POINT)
        assert tone == ToneMode.AFFIRMING


class TestFallbackResponses:
    def test_create_fallback_response_all_strategies(self):
        generator = ResponseGenerator()
        
        for strategy in CoachingStrategy:
            response = generator._create_fallback_response(strategy, ToneMode.SOCRATIC)
            
            expected_text = FALLBACK_RESPONSES[strategy]
            assert response.text == expected_text
            assert response.tone == ToneMode.SOCRATIC
            assert response.follow_up_prompt is None
            assert response.estimated_speaking_seconds > 0
    
    def test_fallback_response_preserves_tone(self):
        generator = ResponseGenerator()
        
        for tone in ToneMode:
            response = generator._create_fallback_response(
                CoachingStrategy.CLARIFY_POSITION, 
                tone
            )
            assert response.tone == tone


class TestSpeakingTimeEstimation:
    def test_estimate_speaking_time_calculation(self):
        generator = ResponseGenerator()
        
        # Test with known word count
        text = "This is a test sentence with exactly eight words."
        expected_time = 8 / ESTIMATED_WORDS_PER_SECOND
        actual_time = generator._estimate_speaking_time(text)
        
        assert actual_time == expected_time
    
    def test_estimate_speaking_time_empty_text(self):
        generator = ResponseGenerator()
        
        actual_time = generator._estimate_speaking_time("")
        assert actual_time == 0.0
    
    def test_estimate_speaking_time_single_word(self):
        generator = ResponseGenerator()
        
        actual_time = generator._estimate_speaking_time("Hello")
        expected_time = 1 / ESTIMATED_WORDS_PER_SECOND
        assert actual_time == expected_time


class TestClaudeResponseParsing:
    def test_parse_claude_response_clean_json(self):
        generator = ResponseGenerator()
        
        raw_response = '{"text": "What do you think?", "follow_up_prompt": null}'
        parsed = generator._parse_claude_response(raw_response)
        
        assert parsed["text"] == "What do you think?"
        assert parsed["follow_up_prompt"] is None
    
    def test_parse_claude_response_with_markdown_fences(self):
        generator = ResponseGenerator()
        
        raw_response = '```json\n{"text": "What do you think?", "follow_up_prompt": "Tell me more"}\n```'
        parsed = generator._parse_claude_response(raw_response)
        
        assert parsed["text"] == "What do you think?"
        assert parsed["follow_up_prompt"] == "Tell me more"
    
    def test_parse_claude_response_with_whitespace(self):
        generator = ResponseGenerator()
        
        raw_response = '  \n  {"text": "What do you think?"}  \n  '
        parsed = generator._parse_claude_response(raw_response)
        
        assert parsed["text"] == "What do you think?"
    
    def test_parse_claude_response_invalid_json_raises_error(self):
        generator = ResponseGenerator()
        
        raw_response = '{"text": "What do you think?", invalid}'
        
        with pytest.raises(json.JSONDecodeError):
            generator._parse_claude_response(raw_response)


class TestPromptGeneration:
    def test_build_generation_prompt_includes_all_context(self, sample_request):
        generator = ResponseGenerator()
        tone = ToneMode.SOCRATIC
        
        prompt = generator._build_generation_prompt(sample_request, tone)
        
        assert sample_request.topic in prompt
        assert sample_request.user_position in prompt
        assert sample_request.coaching_action.strategy.value in prompt
        assert tone.value in prompt
        assert str(sample_request.turn_number) in prompt
        assert "What makes you think that?" in prompt  # prior response
    
    def test_build_generation_prompt_empty_prior_responses(self, sample_request):
        generator = ResponseGenerator()
        sample_request.prior_responses = []
        tone = ToneMode.CHALLENGING
        
        prompt = generator._build_generation_prompt(sample_request, tone)
        
        assert "None" in prompt
        assert tone.value in prompt


class TestAcceptanceCriteria:
    @pytest.mark.asyncio
    async def test_returns_response_within_two_seconds(self, sample_request, mock_anthropic_client):
        generator = ResponseGenerator()
        generator._client = mock_anthrop