import enum
from typing import Annotated, Any

from pydantic import (
    AnyUrl,
    BeforeValidator,
    HttpUrl,
    MySQLDsn,
    SecretStr,
    computed_field,
    model_validator,
)
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Environment(str, enum.Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    DOMAIN: str = "localhost"
    ENVIRONMENT: Environment = Environment.LOCAL

    SESSION_COOKIE_NAME: str = "session"
    # 90 days
    SESSION_COOKIE_MAX_AGE: int = 3600 * 24 * 90
    SESSION_COOKIE_SECURE: bool = False

    BROWSER_ID_COOKIE_NAME: str = "bid"
    BROWSER_ID_COOKIE_MAX_AGE: int = 3600 * 24 * 365 * 2

    # LANGFUSE_HOST: str
    # LANGFUSE_PUBLIC_KEY: str
    # LANGFUSE_SECRET_KEY: str

    @computed_field  # type: ignore[misc]
    @property
    def server_host(self) -> str:
        # Use HTTPS for anything other than local development
        if self.ENVIRONMENT == Environment.LOCAL:
            return f"http://{self.DOMAIN}"
        return f"https://{self.DOMAIN}"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    PROJECT_NAME: str = "SunDB.AI"
    SENTRY_DSN: HttpUrl | None = None

    LOCAL_FILE_STORAGE_PATH: str = "/Users/apple/Desktop/Tsinghua/Research/tidb-xuwu.ai/shared/data"


    PGDB_HOST: str = "localhost"
    PGDB_PORT: int = 5432
    PGDB_USER: str = "myuser"
    PGDB_PASSWORD: str = "mypassword"
    PGDB_DATABASE: str = "mydb"
    PGDB_SSL: bool = False

    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # TODO: move below config to `option` table, it should be configurable by staff in console
    SUNDB_AI_CHAT_ENDPOINT: str = "http://localhost:3000/api/v1/chats"
    SUNDB_AI_API_KEY: SecretStr | None = None

    COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH: str | None = None
    COMPLIED_PREREQUISITE_ANALYSIS_PROGRAM_PATH: str | None = None

    # CAUTION: Do not change EMBEDDING_DIMS after initializing the database.
    # Changing the embedding dimensions requires recreating the database and tables.
    # The default EMBEDDING_DIMS and EMBEDDING_MAX_TOKENS are set for the OpenAI text-embedding-3-small model.
    # If using a different embedding model, adjust these values according to the model's specifications.
    # For example:
    #   maidalun1020/bce-embedding-base_v1: EMBEDDING_DIMS=768   EMBEDDING_MAX_TOKENS=512
    EMBEDDING_DIMS: int = 1536
    EMBEDDING_MAX_TOKENS: int = 7000

    @computed_field  # type: ignore[misc]
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+psycopg2://{self.PGDB_USER}:{self.PGDB_PASSWORD}@{self.PGDB_HOST}:{self.PGDB_PORT}/{self.PGDB_DATABASE}?sslmode=disable"

    @computed_field  # type: ignore[misc]
    @property
    def SQLALCHEMY_ASYNC_DATABASE_URI(self) -> str:
        return f"postgresql+asyncpg://{self.PGDB_USER}:{self.PGDB_PASSWORD}@{self.PGDB_HOST}:{self.PGDB_PORT}/{self.PGDB_DATABASE}?ssl=disable"

    @model_validator(mode="after")
    def _validate_secrets(self) -> Self:
        secret = self.SECRET_KEY
        if not secret:
            raise ValueError(
                f"Please set a secret key using the SECRET_KEY environment variable."
            )

        min_length = 32
        if len(secret.encode()) < min_length:
            message = (
                "The SECRET_KEY is too short, "
                f"please use a longer secret, at least {min_length} characters."
            )
            raise ValueError(message)
        return self


settings = Settings()  # type: ignore
