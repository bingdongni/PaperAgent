"""
Unit tests for core configuration and LLM manager
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from paperagent.core.config import Settings
from paperagent.core.llm_manager import LLMManager, OpenAILLM, AnthropicLLM, OllamaLLM


class TestSettings:
    """Test configuration settings"""

    def test_default_settings(self):
        """Test default settings initialization"""
        settings = Settings()
        assert settings.app_name == "PaperAgent"
        assert settings.app_version == "1.0.0"
        assert settings.default_llm_provider in ["openai", "anthropic", "ollama"]

    def test_custom_settings(self):
        """Test custom settings"""
        settings = Settings(
            app_name="CustomPaperAgent",
            debug=True,
            default_llm_provider="ollama"
        )
        assert settings.app_name == "CustomPaperAgent"
        assert settings.debug is True
        assert settings.default_llm_provider == "ollama"

    def test_directory_creation(self, tmp_path):
        """Test that required directories are created"""
        settings = Settings(data_dir=str(tmp_path / "data"))
        assert (tmp_path / "data").exists()


class TestLLMManager:
    """Test LLM Manager"""

    def test_initialization_default(self):
        """Test default LLM manager initialization"""
        manager = LLMManager(provider="ollama")
        assert manager.provider == "ollama"
        assert manager.llm is not None

    def test_switch_provider(self):
        """Test switching LLM providers"""
        manager = LLMManager(provider="ollama")
        original_provider = manager.provider

        manager.switch_provider("ollama")
        assert manager.provider == "ollama"

    @patch('paperagent.core.llm_manager.openai')
    def test_openai_generate(self, mock_openai):
        """Test OpenAI text generation"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated text"))]
        mock_openai.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(api_key="test_key", model="gpt-4")
        result = llm.generate("Test prompt")

        assert result == "Generated text"
        mock_openai.chat.completions.create.assert_called_once()

    @patch('paperagent.core.llm_manager.Anthropic')
    def test_anthropic_generate(self, mock_anthropic_class):
        """Test Anthropic text generation"""
        # Mock the client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated text")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        llm = AnthropicLLM(api_key="test_key", model="claude-3")
        result = llm.generate("Test prompt")

        assert result == "Generated text"

    @patch('paperagent.core.llm_manager.requests.post')
    def test_ollama_generate(self, mock_post):
        """Test Ollama text generation"""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        llm = OllamaLLM(base_url="http://localhost:11434", model="llama3")
        result = llm.generate("Test prompt")

        assert result == "Generated text"
        mock_post.assert_called_once()

    @patch('paperagent.core.llm_manager.requests.post')
    def test_ollama_chat(self, mock_post):
        """Test Ollama chat completion"""
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "Chat response"}}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        llm = OllamaLLM(base_url="http://localhost:11434", model="llama3")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        result = llm.chat(messages)

        assert result == "Chat response"

    def test_invalid_provider(self):
        """Test error handling for invalid provider"""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMManager(provider="invalid_provider")


class TestPrompts:
    """Test prompt templates"""

    def test_prompt_formatting(self):
        """Test prompt template formatting"""
        from paperagent.core.prompts import LiteraturePrompts

        prompt = LiteraturePrompts.TOPIC_RECOMMENDATION
        formatted = prompt.format(
            field="Computer Science",
            keywords="machine learning, AI"
        )

        assert "Computer Science" in formatted
        assert "machine learning" in formatted
        assert "AI" in formatted

    def test_all_prompts_exist(self):
        """Test that all required prompts are defined"""
        from paperagent.core.prompts import (
            LiteraturePrompts,
            ExperimentPrompts,
            WritingPrompts,
            BossPrompts
        )

        # Check Literature prompts
        assert hasattr(LiteraturePrompts, 'TOPIC_RECOMMENDATION')
        assert hasattr(LiteraturePrompts, 'LITERATURE_SUMMARY')

        # Check Experiment prompts
        assert hasattr(ExperimentPrompts, 'EXPERIMENT_DESIGN')
        assert hasattr(ExperimentPrompts, 'DATA_ANALYSIS')

        # Check Writing prompts
        assert hasattr(WritingPrompts, 'PAPER_STRUCTURE')
        assert hasattr(WritingPrompts, 'SECTION_WRITING')

        # Check Boss prompts
        assert hasattr(BossPrompts, 'TASK_DECOMPOSITION')
        assert hasattr(BossPrompts, 'QUALITY_CHECK')
