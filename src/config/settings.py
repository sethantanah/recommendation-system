from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Configuration
    PASSWORD: str
    DATABASE_NAME: str
    
    SOURCE_COLLECTION: str
    VECTOR_COLLECTION: str

    class Config:
        frozen = True
        env_file = ".env"
        case_sensitive = True
