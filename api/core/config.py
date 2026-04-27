from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bucket: str = ""
    model_save_dir: str = "saved_models"
    label_encoder_path: str = "preprocessed_data/label_encoder.pkl"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
