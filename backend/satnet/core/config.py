from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path

_ENV_PATHS = (
    ".env",
    str(Path(__file__).resolve().parents[3] / ".env"),
)


class Settings(BaseSettings):
    app_name: str = "SatNet"
    environment: str = "development"
    database_url: str = "sqlite:///./satnet.db"
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "satnet"
    mysql_user: str = "satnet"
    mysql_password: str = ""
    tle_source_url: str = "https://celestrak.org/NORAD/elements/gp.php"
    default_safety_radius_km: float = Field(25.0, gt=0)
    default_time_step_seconds: int = Field(60, ge=1)
    max_simulation_hours: int = Field(48, ge=1, le=168)
    ml_enabled: bool = False
    ml_model_path: str | None = None
    cors_origins: str = "http://localhost:5173"
    mongodb_uri: str = "mongodb://localhost:27017"
    jwt_secret: str = "satnet-dev-secret-change-me-in-production"
    model_config = SettingsConfigDict(env_file=_ENV_PATHS, extra="ignore")
    @property
    def cors_origin_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()
