import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM model settings
    llm_model_name: str = "microsoft/DialoGPT-medium"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Fine-tuning Settings
    fine_tuned_model_path: str = "./models/fine_tuned/"
    use_fine_tuned_model: bool = False
    fine_tune_data_path: str = "./data/fine_tune_data/"
    
    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "story_examples"
    chroma_persist_directory: str = "./chroma_db"

    # Dataset Settings
    story_dataset_path: str = "./data/story_dataset.csv"
    max_rag_results: int = 5
    similarity_threshold: float = 0.7

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    reload: bool = True
    
    # CORS settings
    allowed_origins: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security settings
    secret_key: str = "your-secret-key-change-in-production"
    
    # Story settings
    max_story_length: int = 1000
    min_story_length: int = 100
    max_characters_per_story: int = 5
    max_character_name_length: int = 30
    
    # Prompt Engineering Settings
    prompt_templates_path: str = "./data/prompts/"
    use_dynamic_prompts: bool = True
    context_window_size: int = 4096
    
    # Performance Settings
    batch_size: int = 1
    max_concurrent_requests: int = 10
    response_timeout: int = 60  # seconds
    
    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_response_caching: bool = True

    # Quality Control Settings
    content_filter_enabled: bool = True
    min_story_sentences: int = 3
    max_repetition_ratio: float = 0.3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""
    
    def get_chroma_settings(self) -> dict:
        # Return to chromaDB settings 
        return {
            "host": self.chroma_host,
            "port": self.chroma_port,
            "collection_name": self.chroma_collection_name,
            "persist_directory": self.chroma_persist_directory
        }
    
    def get_llm_settings(self) -> dict:
        # Return to LLM settings
        return {
            "model_name": self.llm_model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "use_fine_tuned": self.use_fine_tuned_model,
            "fine_tuned_path": self.fine_tuned_model_path
        }

    def get_prompt_settings(self) -> dict:
        # Return prompt engineering settings
        return {
            "templates_path": self.prompt_templates_path,
            "use_dynamic": self.use_dynamic_prompts,
            "context_window": self.context_window_size
        }

    def get_rag_settings(self) -> dict:
        # Return RAG settings
        return {
            "dataset_path": self.story_dataset_path,
            "max_results": self.max_rag_results,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model_name
        }
    
    def get_quality_settings(self) -> dict:
        # Return quality control settings
        return {
            "content_filter": self.content_filter_enabled,
            "min_sentences": self.min_story_sentences,
            "max_repetition": self.max_repetition_ratio,
            "min_length": self.min_story_length,
            "max_length": self.max_story_length
        }
    
    def is_development(self) -> bool:
        # in a development environment?
        return self.debug
    
    def is_production(self) -> bool:
        # in a production environment?
        return not self.debug
    
    def validate_settings(self) -> bool:
        # Validate all settings
        try:
            # API port check
            if not (1 <= self.api_port <= 65535):
                raise ValueError("API port must be between 1-65535")
            
            # Temperature check
            if not (0.0 <= self.temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0-2.0")
            
            # Top-p check
            if not (0.0 <= self.top_p <= 1.0):
                raise ValueError("Top-p must be between 0.0-1.0")
            
            # Token check
            if self.max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
            
            # Story length checks
            if self.min_story_length >= self.max_story_length:
                raise ValueError("Min story length must be less than max story length")
            
            # File path checks
            required_dirs = [
                os.path.dirname(self.story_dataset_path),
                self.prompt_templates_path,
                self.chroma_persist_directory
            ]
            
            for dir_path in required_dirs:
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
            
            # Fine-tuned model path check
            if self.use_fine_tuned_model and not os.path.exists(self.fine_tuned_model_path):
                raise ValueError(f"Fine-tuned model path does not exist: {self.fine_tuned_model_path}")
            
            return True
            
        except Exception as e:
            print(f"Settings validation error: {e}")
            return False
        
    def get_model_info(self) -> dict:
        # Get current model configuration info
        return {
            "base_model": self.llm_model_name,
            "embedding_model": self.embedding_model_name,
            "fine_tuned_enabled": self.use_fine_tuned_model,
            "fine_tuned_path": self.fine_tuned_model_path if self.use_fine_tuned_model else None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

# Global settings instance
settings = Settings()

# Validate settings on import
if not settings.validate_settings():
    print("Settings validation failed! Please check your configuration.")
else:
    print(f"Settings loaded successfully")
    print(f"Model: {settings.llm_model_name}")
    print(f"Fine-tuned: {'Yes' if settings.use_fine_tuned_model else 'No'}")
    print(f"RAG Dataset: {settings.story_dataset_path}")