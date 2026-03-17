import os
import json
import time
import re
import asyncio
from typing import List
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio
from shared_types import WordError, PronunciationResult, ErrorSeverity

@pytest.fixture
def mock_anthropic_client():
    """Mock AsyncAnthropic client for testing"""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '{"error_description": "Test error", "correction_tip": "Test tip", "model_sentence": "Test sentence"}'
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def sample_word_errors():
    """Sample WordError objects for testing"""
    return [
        WordError(word="think", expected_ipa="/θɪŋk/", actual_ipa="/sɪŋk/", severity=ErrorSeverity.HIGH),
        WordError(word="very", expected_ipa="/vɛri/", actual_ipa="/wɛri/", severity=ErrorSeverity.MEDIUM),
        WordError(word="cat", expected_ipa="/kæt/", actual_ipa="/kɛt/", severity=ErrorSeverity.LOW),
        WordError(word="the", expected_ipa="/ðə/", actual_ipa="/də/", severity=ErrorSeverity.HIGH),
        WordError(word="love", expected_ipa="/lʌv/", actual_ipa="/lɔv/", severity=ErrorSeverity.MEDIUM),
    ]

@pytest.fixture
def sample_pronunciation_result():
    """Sample PronunciationResult with no errors"""
    return PronunciationResult(
        mispronounced_words=[],
        fluency_score=0.8
    )

@pytest.fixture  
def sample_pronunciation_result_with_errors(sample_word_errors):
    """Sample PronunciationResult with errors"""
    return PronunciationResult(
        mispronounced_words=sample_word_errors,
        fluency_score=0.6
    )

@pytest.fixture
def coach():
    """PronunciationCoach instance for testing"""
    from pronunciation_coach import PronunciationCoach
    return PronunciationCoach("test-api-key")

def test_module_imports():
    """Test that module imports work correctly"""
    from pronunciation_coach import PronunciationCoach, PronunciationFeedback, WordCorrection
    assert PronunciationCoach is not None
    assert PronunciationFeedback is not None
    assert WordCorrection is not None

@pytest_asyncio.async_test
async def test_generate_feedback_empty_errors(coach, sample_pronunciation_result):
    """Test generate_feedback with empty mispronounced_words returns has_errors=False and no LLM call"""
    with patch.object(coach, '_client') as mock_client:
        result = await coach.generate_feedback(
            sample_pronunciation_result,
            "This is a test transcript",
            "technology"
        )
        
        assert result.has_errors is False
        assert len(result.corrections) == 0
        assert result.drill_sentence == ""
        assert result.fluency_comment is not None
        assert result.overall_message == "Your pronunciation was clear this turn. Keep it up!"
        assert result.latency_ms >= 0
        
        # Verify no LLM calls were made
        mock_client.messages.create.assert_not_called()

@pytest_asyncio.async_test
async def test_generate_feedback_with_five_errors_returns_max_three(coach, sample_pronunciation_result_with_errors):
    """Test generate_feedback with 5 WordErrors returns at most 3 corrections"""
    with patch.object(coach, '_client') as mock_client:
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"error_description": "Test error", "correction_tip": "Test tip", "model_sentence": "Test sentence"}'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        result = await coach.generate_feedback(
            sample_pronunciation_result_with_errors,
            "I think this is very good",
            "conversation"
        )
        
        assert len(result.corrections) <= 3
        assert result.has_errors is True

@pytest_asyncio.async_test
async def test_generate_feedback_returns_within_timeout(coach, sample_pronunciation_result_with_errors):
    """Test generate_feedback returns within 3 seconds"""
    with patch.object(coach, '_client') as mock_client:
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"error_description": "Test error", "correction_tip": "Test tip", "model_sentence": "Test sentence"}'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        start_time = time.time()
        result = await coach.generate_feedback(
            sample_pronunciation_result_with_errors,
            "I think this is very good",
            "conversation"
        )
        end_time = time.time()
        
        assert (end_time - start_time) < 3.0
        assert result is not None
        assert result.latency_ms >= 0

@pytest_asyncio.async_test
async def test_corrections_sorted_by_severity(coach):
    """Test corrections are sorted by severity (HIGH before MEDIUM before LOW)"""
    errors = [
        WordError(word="low", expected_ipa="/loʊ/", actual_ipa="/lo/", severity=ErrorSeverity.LOW),
        WordError(word="high", expected_ipa="/haɪ/", actual_ipa="/hai/", severity=ErrorSeverity.HIGH),
        WordError(word="medium", expected_ipa="/midiəm/", actual_ipa="/midium/", severity=ErrorSeverity.MEDIUM),
    ]
    pronunciation_result = PronunciationResult(mispronounced_words=errors, fluency_score=0.5)
    
    with patch.object(coach, '_client') as mock_client:
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"error_description": "Test error", "correction_tip": "Test tip", "model_sentence": "Test sentence"}'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        result = await coach.generate_feedback(
            pronunciation_result,
            "This is a test transcript",
            "testing"
        )
        
        # Check that HIGH severity comes before MEDIUM which comes before LOW
        severities = [correction.severity for correction in result.corrections]
        expected_order = [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM, ErrorSeverity.LOW]
        assert severities == expected_order

@pytest_asyncio.async_test
async def test_drill_sentence_non_empty_when_errors_present(coach, sample_pronunciation_result_with_errors):
    """Test drill_sentence is non-empty when errors list is non-empty"""
    with patch.object(coach, '_client') as mock_client:
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"error_description": "Test error", "correction_tip": "Test tip", "model_sentence": "Test sentence"}'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        # Mock drill sentence response
        mock_drill_response = MagicMock()
        mock_drill_content = MagicMock()
        mock_drill_content.text = "This is a practice sentence with target sounds."
        mock_drill_response.content = [mock_drill_content]
        mock_client.messages.create.side_effect = [mock_response, mock_response, mock_response, mock_response, mock_response, mock_drill_response]
        
        result = await coach.generate_feedback(
            sample_pronunciation_result_with_errors,
            "I think this is very good",
            "conversation"
        )
        
        assert result.drill_sentence != ""
        assert len(result.drill_sentence) > 0

@pytest_asyncio.async_test
async def test_generate_corrections_concurrent_processing(coach, sample_word_errors):
    """Test _generate_corrections runs concurrently using asyncio.gather"""
    with patch.object(coach, '_generate_single_correction') as mock_single:
        from pronunciation_coach import WordCorrection
        mock_correction = WordCorrection(
            word="test",
            error_description="Test error",
            correction_tip="Test tip",
            model_sentence="Test sentence",
            severity=ErrorSeverity.MEDIUM
        )
        mock_single.return_value = mock_correction
        
        errors = sample_word_errors[:3]  # Take first 3
        result = await coach._generate_corrections(errors, "test transcript")
        
        # Verify all corrections were processed
        assert len(result) == 3
        assert mock_single.call_count == 3
        
        # Verify all calls were made with correct parameters
        for i, error in enumerate(errors):
            mock_single.assert_any_call(error, "test transcript")

@pytest_asyncio.async_test
async def test_generate_single_correction_fallback_on_exception(coach):
    """Test _generate_single_correction returns fallback on LLM timeout or API error"""
    error = WordError(word="test", expected_ipa="/tɛst/", actual_ipa="/test/", severity=ErrorSeverity.HIGH)
    
    with patch.object(coach, '_client') as mock_client:
        # Simulate API exception
        mock_client.messages.create.side_effect = Exception("API Error")
        
        result = await coach._generate_single_correction(error, "test transcript")
        
        assert result.word == "test"
        assert result.error_description == "Pronunciation needs attention"
        assert result.correction_tip == "Practice this word slowly and listen to native speaker recordings."
        assert result.model_sentence == "Please practice saying 'test' carefully."
        assert result.severity == ErrorSeverity.HIGH

@pytest_asyncio.async_test
async def test_generate_feedback_no_exception_propagation(coach):
    """Test no exception propagates from generate_feedback - always returns PronunciationFeedback"""
    pronunciation_result = PronunciationResult(mispronounced_words=[], fluency_score=0.8)
    
    with patch.object(coach, '_client') as mock_client:
        # Simulate total failure
        mock_client.messages.create.side_effect = Exception("Total API Failure")
        
        result = await coach.generate_feedback(pronunciation_result, "test", "test")
        
        # Should still return a valid PronunciationFeedback object
        from pronunciation_coach import PronunciationFeedback
        assert isinstance(result, PronunciationFeedback)
        assert result.latency_ms >= 0

@pytest_asyncio.async_test 
async def test_generate_feedback_timeout_handling(coach, sample_pronunciation_result_with_errors):
    """Test generate_feedback handles timeout gracefully"""
    with patch.object(coach, '_client') as mock_client:
        # Simulate timeout by making the call hang
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(3.0)  # Longer than 2.8s timeout
            return MagicMock()
        
        mock_client.messages.create.side_effect = slow_response
        
        result = await coach.generate_feedback(
            sample_pronunciation_result_with_errors,
            "test transcript", 
            "test topic"
        )
        
        # Should return fallback response
        assert result is not None
        assert result.has_errors is False
        assert result.corrections == []

@pytest_asyncio.async_test
async def test_word_correction_tip_never_empty(coach):
    """Test each WordCorrection.correction_tip is never None or empty string"""
    error = WordError(word="test", expected_ipa="/tɛst/", actual_ipa="/test/", severity=ErrorSeverity.HIGH)
    
    with patch.object(coach, '_client') as mock_client:
        # Test with valid response
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"error_description": "Test error", "correction_tip": "Test tip", "model_sentence": "Test sentence"}'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        result = await coach._generate_single_correction(error, "test transcript")
        assert result.correction_tip is not None
        assert result.correction_tip != ""
        
        # Test with exception (fallback)
        mock_client.messages.create.side_effect = Exception("API Error")
        result = await coach._generate_single_correction(error, "test transcript")
        assert result.correction_tip is not None
        assert result.correction_tip != ""

@pytest_asyncio.async_test
async def test_anthropic_client_initialization():
    """Test Anthropic client is initialized with correct timeout"""
    from pronunciation_coach import PronunciationCoach
    
    with patch('pronunciation_coach.AsyncAnthropic') as mock_anthropic:
        coach = PronunciationCoach("test-key")
        mock_anthropic.assert_called_once_with(api_key="test-key", timeout=30.0)

@pytest_asyncio.async_test
async def test_fluency_score_comments(coach):
    """Test fluency comments are generated based on fluency_score"""
    # Test high fluency score
    high_fluency_result = PronunciationResult(mispronounced_words=[], fluency_score=0.8)
    result = await coach.generate_feedback(high_fluency_result, "test", "test")
    assert "clear" in result.fluency_comment.lower() or "good" in result.fluency_comment.lower()
    
    # Test low fluency score
    low_fluency_result = PronunciationResult(mispronounced_words=[], fluency_score=0.5)
    result = await coach.generate_feedback(low_fluency_result, "test", "test")
    assert "smooth" in result.fluency_comment.lower() or "pauses" in result.fluency_comment.lower()

@pytest_asyncio.async_test
async def test_drill_sentence_generation_with_phonemes(coach):
    """Test drill sentence generation includes target phonemes"""
    errors = [
        WordError(word="think", expected_ipa="/θɪŋk/", actual_ipa="/sɪŋk/", severity=ErrorSeverity.HIGH)
    ]
    
    with patch.object(coach, '_client') as mock_client:
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Think about the methods we use for thinking through problems."
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        result = await coach._generate_drill_sentence(errors, "education")
        
        assert result != ""
        assert len(result) > 0
        
        # Verify the prompt included the phoneme
        call_args = mock_client.messages.create.call_args
        prompt = call_args[1]['messages'][0]['content']
        assert "/θɪŋk/" in prompt