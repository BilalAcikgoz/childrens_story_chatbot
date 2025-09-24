import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5432/story_chatbot"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_db: str = "story_chatbot"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    
    # LLM model settings
    llm_model_name: str = "microsoft/DialoGPT-medium"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "story_examples"
    chroma_persist_directory: str = "./chroma_db"
    
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
    access_token_expire_minutes: int = 30
    
    # Story settings
    max_story_length: int = 1000
    min_story_length: int = 10
    max_characters_per_story: int = 5
    max_character_name_length: int = 30
    
    # Rate limiting
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100
    
    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 saat
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Prefix of environment variables
        env_prefix = ""

    def get_database_url(self) -> str:
        # Create to database URL
        if hasattr(self, '_database_url_cache'):
            return self._database_url_cache
            
        if self.database_url and self.database_url != "postgresql://postgres:password@localhost:5432/story_chatbot":
            self._database_url_cache = self.database_url
        else:
            self._database_url_cache = f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        
        return self._database_url_cache
    
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
            "top_p": self.top_p
        }
    
    def is_development(self) -> bool:
        # in a development environment?
        return self.debug
    
    def is_production(self) -> bool:
        # in a production environment?
        return not self.debug
    
    def validate_settings(self) -> bool:
        # Validate settings
        try:
            # Database URL check
            if not self.get_database_url():
                raise ValueError("Database URL cannot be empty")
            
            # Port check
            if not (1 <= self.api_port <= 65535):
                raise ValueError("API port must be between 1-65535")
            
            if not (1 <= self.postgres_port <= 65535):
                raise ValueError("PostgreSQL port must be between 1-65535")
            
            # Temperature check
            if not (0.0 <= self.temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0-2.0")
            
            # Token check
            if self.max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
            
            return True
            
        except Exception as e:
            print(f"Setting validation error: {e}")
            return False

# Global settings instance
settings = Settings()

# Validate settings
if not settings.validate_settings():
    raise ValueError("Invalid settings detected!")