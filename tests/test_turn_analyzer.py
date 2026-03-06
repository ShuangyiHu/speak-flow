import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from turn_analyzer import TurnAnalyzer, TurnInput, ArgumentResult, WordError, PronunciationResult, TurnAnalysis

@pytest.fixture
def turn_analyzer():
    return TurnAnalyzer()

@pytest.fixture
def sample_turn_input():
    return TurnInput(
        transcript="I believe social media is harmful because it reduces face-to-face interaction and studies show increased anxiety rates among heavy users.",
        session_id="test-session-123",
        turn_number=1,
        topic="Social media impact on society",
        user_position="against",
        audio_path="/path/to/audio.wav",
        prior_turns=[]
    )

@pytest.fixture
def empty_turn_input():
    return TurnInput(
        transcript="",
        session_id="test-session-123",
        turn_number=1,
        topic="Social media impact on society",
        user_position="against",
        audio_path="/path/to/audio.wav",
        prior_turns=[]
    )

@pytest.fixture
def question_turn_input():
    return TurnInput(
        transcript="What do you think about social media?",
        session_id="test-session-123",
        turn_number=1,
        topic="Social media impact on society",
        user_position="against",
        audio_path="/path/to/audio.wav",
        prior_turns=[]
    )

@pytest.fixture
def opinion_only_turn_input():
    return TurnInput(
        transcript="I just don't like social media.",
        session_id="test-session-123",
        turn_number=1,
        topic="Social media impact on society",
        user_position="against",
        audio_path="/path/to/audio.wav",
        prior_turns=[]
    )

@pytest.fixture
def mock_anthropic_response_strong_argument():
    return {
        "content": [
            {
                "text": """
                {
                    "has_claim": true,
                    "has_reasoning": true,
                    "has_evidence": true,
                    "logical_gaps": [],
                    "vocabulary_flags": ["face-to-face"],
                    "argument_score": 0.85,
                    "summary": "Strong argument with clear claim, reasoning, and evidence"
                }
                """
            }
        ]
    }

@pytest.fixture
def mock_anthropic_response_weak_argument():
    return {
        "content": [
            {
                "text": """
                {
                    "has_claim": false,
                    "has_reasoning": false,
                    "has_evidence": false,
                    "logical_gaps": ["no supporting reasons provided"],
                    "vocabulary_flags": [],
                    "argument_score": 0.3,
                    "summary": "Weak argument with only opinion"
                }
                """
            }
        ]
    }

@pytest.fixture
def mock_anthropic_response_question():
    return {
        "content": [
            {
                "text": """
                {
                    "has_claim": false,
                    "has_reasoning": false,
                    "has_evidence": false,
                    "logical_gaps": [],
                    "vocabulary_flags": [],
                    "argument_score": 0.0,
                    "summary": "Question format, no claim present"
                }
                """
            }
        ]
    }

@pytest.fixture
def mock_mfa_result_with_errors():
    return [
        WordError(word="social", expected_ipa="/ˈsoʊʃəl/", actual_ipa="/ˈsəʊʃəl/", severity="minor"),
        WordError(word="interaction", expected_ipa="/ˌɪntərˈækʃən/", actual_ipa="/ˌɪntərˈekʃən/", severity="moderate")
    ]

@pytest.mark.asyncio
async def test_analyze_happy_path_strong_argument(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument, mock_mfa_result_with_errors):
    """Test successful analysis of a well-structured argument with pronunciation errors"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = mock_mfa_result_with_errors
            
            start_time = datetime.now()
            result = await turn_analyzer.analyze(sample_turn_input)
            end_time = datetime.now()
            
            assert isinstance(result, TurnAnalysis)
            assert result.turn_input == sample_turn_input
            assert result.argument.has_claim is True
            assert result.argument.has_reasoning is True
            assert result.argument.has_evidence is True
            assert result.argument.argument_score > 0.7
            assert len(result.pronunciation.mispronounced_words) == 2
            assert result.pronunciation.fluency_score is not None
            assert result.latency_ms < 3000
            assert mock_client.messages.create.call_count == 1

@pytest.mark.asyncio
async def test_analyze_question_input(turn_analyzer, question_turn_input, mock_anthropic_response_question):
    """Test analysis of question format input - should have has_claim=False"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_question))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            result = await turn_analyzer.analyze(question_turn_input)
            
            assert result.argument.has_claim is False
            assert result.argument.has_reasoning is False
            assert result.argument.has_evidence is False
            assert result.argument.argument_score == 0.0

@pytest.mark.asyncio
async def test_analyze_weak_argument(turn_analyzer, opinion_only_turn_input, mock_anthropic_response_weak_argument):
    """Test analysis of opinion-only turn - should have argument_score < 0.4"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_weak_argument))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            result = await turn_analyzer.analyze(opinion_only_turn_input)
            
            assert result.argument.has_claim is False
            assert result.argument.argument_score < 0.4
            assert "no supporting reasons provided" in result.argument.logical_gaps

@pytest.mark.asyncio
async def test_analyze_empty_transcript(turn_analyzer, empty_turn_input):
    """Test empty transcript returns default TurnAnalysis without raising exception"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            result = await turn_analyzer.analyze(empty_turn_input)
            
            assert isinstance(result, TurnAnalysis)
            assert result.turn_input == empty_turn_input
            assert result.argument.has_claim is False
            assert result.argument.has_reasoning is False
            assert result.argument.has_evidence is False
            assert result.argument.argument_score == 0.0
            assert len(result.pronunciation.mispronounced_words) == 0
            assert mock_client.messages.create.call_count == 0

@pytest.mark.asyncio
async def test_analyze_with_stub_mfa_enabled(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test that stub MFA returns empty mispronounced_words when USE_STUB_MFA=True"""
    with patch.dict(os.environ, {'USE_STUB_MFA': 'True'}):
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
            
            result = await turn_analyzer.analyze(sample_turn_input)
            
            assert len(result.pronunciation.mispronounced_words) == 0
            assert result.pronunciation.fluency_score is not None

@pytest.mark.asyncio
async def test_analyze_performance_under_3_seconds(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test that analysis completes within 3 seconds for 30-word input"""
    # Create a 30-word transcript
    thirty_word_input = TurnInput(
        transcript="I strongly believe that social media platforms are fundamentally harmful to society because they reduce meaningful face-to-face human interactions and numerous psychological studies have consistently shown increased anxiety depression rates",
        session_id="test-session-123",
        turn_number=1,
        topic="Social media impact on society",
        user_position="against",
        audio_path="/path/to/audio.wav",
        prior_turns=[]
    )
    
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            start_time = datetime.now()
            result = await turn_analyzer.analyze(thirty_word_input)
            end_time = datetime.now()
            
            latency_ms = (end_time - start_time).total_seconds() * 1000
            assert latency_ms < 3000
            assert result.latency_ms < 3000

@pytest.mark.asyncio
async def test_concurrent_analysis_execution(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test that argument and pronunciation analysis run concurrently"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            with patch('asyncio.gather') as mock_gather:
                mock_gather.return_value = [
                    ArgumentResult(has_claim=True, has_reasoning=True, has_evidence=True, 
                                 logical_gaps=[], vocabulary_flags=[], argument_score=0.85, summary="test"),
                    PronunciationResult(mispronounced_words=[], fluency_score=0.9, target_phonemes=[])
                ]
                
                await turn_analyzer.analyze(sample_turn_input)
                
                mock_gather.assert_called_once()

@pytest.mark.asyncio
async def test_mfa_uses_asyncio_to_thread(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test that MFA analysis uses asyncio.to_thread for non-blocking execution"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = []
            
            await turn_analyzer.analyze(sample_turn_input)
            
            mock_to_thread.assert_called()

@pytest.mark.asyncio
async def test_anthropic_client_initialization():
    """Test that Anthropic client is properly initialized"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        analyzer = TurnAnalyzer()
        mock_anthropic_class.assert_called_once()

@pytest.mark.asyncio
async def test_anthropic_api_call_parameters(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test that Anthropic API is called with correct parameters"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            await turn_analyzer.analyze(sample_turn_input)
            
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            assert call_args[1]['model'] == 'claude-3-5-sonnet-20241022'
            assert 'Claim-Reason-Evidence' in call_args[1]['messages'][0]['content']

@pytest.mark.asyncio
async def test_error_handling_anthropic_failure(turn_analyzer, sample_turn_input):
    """Test graceful handling of Anthropic API failures"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.return_value = []
            
            result = await turn_analyzer.analyze(sample_turn_input)
            
            # Should return default argument result on API failure
            assert result.argument.has_claim is False
            assert result.argument.argument_score == 0.0

@pytest.mark.asyncio
async def test_error_handling_mfa_failure(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test graceful handling of MFA analysis failures"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=Mock(**mock_anthropic_response_strong_argument))
        
        with patch('turn_analyzer.run_mfa_analysis') as mock_mfa:
            mock_mfa.side_effect = Exception("MFA Error")
            
            result = await turn_analyzer.analyze(sample_turn_input)
            
            # Should return default pronunciation result on MFA failure
            assert len(result.pronunciation.mispronounced_words) == 0
            assert result.pronunciation.fluency_score == 0.0

@pytest.mark.asyncio
async def test_timestamp_and_latency_calculation(turn_analyzer, sample_turn_input, mock_anthropic_response_strong_argument):
    """Test that timestamp and