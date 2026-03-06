import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import subprocess
import json

from turn_analyzer import (
    TurnAnalyzer, TurnInput, ArgumentResult, WordError, 
    PronunciationResult, TurnAnalysis, ANALYSIS_TIMEOUT_SECONDS
)


@pytest.fixture
def sample_turn_input():
    """Create a sample TurnInput for testing."""
    return TurnInput(
        transcript="I believe that renewable energy is better than fossil fuels because it reduces carbon emissions and creates sustainable jobs.",
        session_id="test_session_123",
        turn_number=1,
        topic="Renewable Energy vs Fossil Fuels",
        user_position="pro",
        audio_path=Path("/tmp/test_audio.wav"),
        prior_turns=[]
    )


@pytest.fixture
def empty_turn_input():
    """Create a TurnInput with empty transcript."""
    return TurnInput(
        transcript="",
        session_id="test_session_empty",
        turn_number=1,
        topic="Test Topic",
        user_position="pro",
        audio_path=Path("/tmp/empty_audio.wav"),
        prior_turns=[]
    )


@pytest.fixture
def question_turn_input():
    """Create a TurnInput with a question (no claim)."""
    return TurnInput(
        transcript="What do you think about renewable energy?",
        session_id="test_session_question",
        turn_number=1,
        topic="Renewable Energy",
        user_position="pro",
        audio_path=Path("/tmp/question_audio.wav"),
        prior_turns=[]
    )


@pytest.fixture
def opinion_only_turn_input():
    """Create a TurnInput with opinion but no reasoning."""
    return TurnInput(
        transcript="I just like solar power better.",
        session_id="test_session_opinion",
        turn_number=1,
        topic="Renewable Energy",
        user_position="pro",
        audio_path=Path("/tmp/opinion_audio.wav"),
        prior_turns=[]
    )


@pytest.fixture
def well_structured_turn_input():
    """Create a TurnInput with well-structured CRE argument."""
    return TurnInput(
        transcript="Solar energy is superior to coal power because it produces zero emissions during operation and according to the International Energy Agency, solar costs have dropped 85% since 2010 making it economically viable.",
        session_id="test_session_structured",
        turn_number=1,
        topic="Energy Sources",
        user_position="pro",
        audio_path=Path("/tmp/structured_audio.wav"),
        prior_turns=[]
    )


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = AsyncMock()
    
    # Mock stream context manager
    stream_mock = AsyncMock()
    chunk_mock = Mock()
    chunk_mock.type = "content_block_delta"
    chunk_mock.delta = Mock()
    
    # Create async iterator for stream chunks
    async def async_iter():
        chunk_mock.delta.text = json.dumps({
            "has_claim": True,
            "has_reasoning": True,
            "has_evidence": False,
            "logical_gaps": [],
            "vocabulary_flags": [],
            "argument_score": 0.75,
            "summary": "Good argument with claim and reasoning"
        })
        yield chunk_mock
    
    stream_mock.__aiter__ = async_iter
    stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    stream_mock.__aexit__ = AsyncMock(return_value=None)
    
    client.messages.stream.return_value = stream_mock
    return client


@pytest.fixture
def mock_anthropic_client_no_claim():
    """Create a mock Anthropic client that returns no claim response."""
    client = AsyncMock()
    
    stream_mock = AsyncMock()
    chunk_mock = Mock()
    chunk_mock.type = "content_block_delta"
    chunk_mock.delta = Mock()
    
    async def async_iter():
        chunk_mock.delta.text = json.dumps({
            "has_claim": False,
            "has_reasoning": False,
            "has_evidence": False,
            "logical_gaps": ["No clear position stated"],
            "vocabulary_flags": [],
            "argument_score": 0.2,
            "summary": "Question format, no argumentative claim"
        })
        yield chunk_mock
    
    stream_mock.__aiter__ = async_iter
    stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    stream_mock.__aexit__ = AsyncMock(return_value=None)
    
    client.messages.stream.return_value = stream_mock
    return client


@pytest.fixture
def mock_anthropic_client_high_score():
    """Create a mock Anthropic client that returns high argument score."""
    client = AsyncMock()
    
    stream_mock = AsyncMock()
    chunk_mock = Mock()
    chunk_mock.type = "content_block_delta"
    chunk_mock.delta = Mock()
    
    async def async_iter():
        chunk_mock.delta.text = json.dumps({
            "has_claim": True,
            "has_reasoning": True,
            "has_evidence": True,
            "logical_gaps": [],
            "vocabulary_flags": [],
            "argument_score": 0.85,
            "summary": "Excellent CRE argument with clear claim, solid reasoning, and supporting evidence"
        })
        yield chunk_mock
    
    stream_mock.__aiter__ = async_iter
    stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    stream_mock.__aexit__ = AsyncMock(return_value=None)
    
    client.messages.stream.return_value = stream_mock
    return client


@pytest.fixture
def mock_anthropic_client_low_score():
    """Create a mock Anthropic client that returns low argument score."""
    client = AsyncMock()
    
    stream_mock = AsyncMock()
    chunk_mock = Mock()
    chunk_mock.type = "content_block_delta"
    chunk_mock.delta = Mock()
    
    async def async_iter():
        chunk_mock.delta.text = json.dumps({
            "has_claim": False,
            "has_reasoning": False,
            "has_evidence": False,
            "logical_gaps": ["Opinion without reasoning", "No supporting evidence"],
            "vocabulary_flags": [],
            "argument_score": 0.3,
            "summary": "Opinion-only statement without reasoning or evidence"
        })
        yield chunk_mock
    
    stream_mock.__aiter__ = async_iter
    stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    stream_mock.__aexit__ = AsyncMock(return_value=None)
    
    client.messages.stream.return_value = stream_mock
    return client


@pytest.fixture
def mock_subprocess_success():
    """Mock successful subprocess call for MFA."""
    result = Mock()
    result.returncode = 0
    result.stdout = "MFA alignment completed successfully"
    result.stderr = ""
    return result


@pytest.fixture
def mock_subprocess_failure():
    """Mock failed subprocess call for MFA."""
    result = Mock()
    result.returncode = 1
    result.stdout = ""
    result.stderr = "MFA alignment failed"
    return result


class TestTurnAnalyzer:
    """Test suite for TurnAnalyzer class."""

    def test_init(self):
        """Test TurnAnalyzer initialization."""
        analyzer = TurnAnalyzer()
        assert analyzer.anthropic_client is not None

    @pytest.mark.asyncio
    async def test_analyze_happy_path(self, sample_turn_input, mock_anthropic_client, mock_subprocess_success):
        """Test successful analysis of a normal turn."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            result = await analyzer.analyze(sample_turn_input)
        
        assert isinstance(result, TurnAnalysis)
        assert result.turn_input == sample_turn_input
        assert isinstance(result.argument, ArgumentResult)
        assert isinstance(result.pronunciation, PronunciationResult)
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_analyze_performance_requirement(self, sample_turn_input, mock_anthropic_client):
        """Test that analysis completes within 3 seconds for 30-word input."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        # Create 30-word input
        thirty_word_input = TurnInput(
            transcript=" ".join(["word"] * 30),
            session_id="perf_test",
            turn_number=1,
            topic="Performance Test",
            user_position="pro",
            audio_path=Path("/tmp/perf_test.wav"),
            prior_turns=[]
        )
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            start_time = datetime.utcnow()
            result = await analyzer.analyze(thirty_word_input)
            end_time = datetime.utcnow()
            
            total_time_ms = (end_time - start_time).total_seconds() * 1000
            
        assert total_time_ms < 3000
        assert result.latency_ms < 3000

    @pytest.mark.asyncio
    async def test_analyze_with_claim_detection(self, sample_turn_input, mock_anthropic_client):
        """Test that has_claim=True when transcript contains clear position statement."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            result = await analyzer.analyze(sample_turn_input)
        
        assert result.argument.has_claim is True

    @pytest.mark.asyncio
    async def test_analyze_question_no_claim(self, question_turn_input, mock_anthropic_client_no_claim):
        """Test that has_claim=False when transcript is a question."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client_no_claim
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            result = await analyzer.analyze(question_turn_input)
        
        assert result.argument.has_claim is False

    @pytest.mark.asyncio
    async def test_analyze_high_argument_score(self, well_structured_turn_input, mock_anthropic_client_high_score):
        """Test that argument_score > 0.7 for well-structured CRE argument."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client_high_score
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            result = await analyzer.analyze(well_structured_turn_input)
        
        assert result.argument.argument_score > 0.7

    @pytest.mark.asyncio
    async def test_analyze_low_argument_score(self, opinion_only_turn_input, mock_anthropic_client_low_score):
        """Test that argument_score < 0.4 for opinion-only turn with no reasoning."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client_low_score
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            result = await analyzer.analyze(opinion_only_turn_input)
        
        assert result.argument.argument_score < 0.4

    @pytest.mark.asyncio
    async def test_analyze_empty_transcript(self, empty_turn_input):
        """Test that empty transcript returns default TurnAnalysis without exception."""
        analyzer = TurnAnalyzer()
        
        result = await analyzer.analyze(empty_turn_input)
        
        assert isinstance(result, TurnAnalysis)
        assert result.argument.has_claim is False
        assert result.argument.argument_score == 0.0
        assert result.pronunciation.fluency_score == 0.0
        assert result.latency_ms == 0

    @pytest.mark.asyncio
    async def test_analyze_stub_mfa_clean_result(self, sample_turn_input, mock_anthropic_client):
        """Test that mispronounced_words is empty list when USE_STUB_MFA=True."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
            result = await analyzer.analyze(sample_turn_input)
        
        assert result.pronunciation.mispronounced_words == []
        assert result.pronunciation.fluency_score == 0.8

    @pytest.mark.asyncio
    async def test_analyze_real_mfa_success(self, sample_turn_input, mock_anthropic_client, mock_subprocess_success):
        """Test successful real MFA analysis."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "false"}):
            with patch('asyncio.to_thread', return_value=mock_subprocess_success):
                with patch.object(analyzer, '_parse_mfa_output') as mock_parse:
                    mock_parse.return_value = PronunciationResult(
                        mispronounced_words=[],
                        fluency_score=0.9,
                        target_phonemes=["θ", "ð"]
                    )
                    
                    result = await analyzer.analyze(sample_turn_input)
        
        assert isinstance(result.pronunciation, PronunciationResult)
        assert result.pronunciation.fluency_score == 0.9

    @pytest.mark.asyncio
    async def test_analyze_real_mfa_failure(self, sample_turn_input, mock_anthropic_client, mock_subprocess_failure):
        """Test MFA failure handling."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        with patch.dict(os.environ, {"USE_STUB_MFA": "false"}):
            with patch('asyncio.to_thread', return_value=mock_subprocess_failure):
                result = await analyzer.analyze(sample_turn_input)
        
        assert result.pronunciation.fluency_score == 0.5
        assert result.pronunciation.mispronounced_words == []

    @pytest.mark.asyncio
    async def test_analyze_timeout(self, sample_turn_input, mock_anthropic_client):
        """Test timeout handling."""
        analyzer = TurnAnalyzer()
        analyzer.anthropic_client = mock_anthropic_client
        
        # Mock a slow operation that exceeds timeout
        async def slow_analyze_argument(turn_input):
            await asyncio.sleep(ANALYSIS_TIMEOUT_SECONDS + 1)
            return ArgumentResult(
                has_claim=True, has_reasoning=True, has_evidence=True,
                logical_gaps=[], vocabulary_flags=[], argument_score=0.8,
                summary="Slow analysis"
            )
        
        with patch.object(analyzer, '_analyze_argument', slow_analyze_argument):
            with patch.dict(os.environ, {"USE_STUB_MFA": "true"}):
                with pytest.raises(asyncio.TimeoutError):
                    await analyzer.analyze(sample_turn_input)

    @pytest.mark.asyncio
    async def test_