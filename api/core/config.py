from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # S3
    bucket: str = ""
    model_save_dir: str = "saved_models"
    label_encoder_path: str = "preprocessed_data/label_encoder.pkl"

    # ECS — leave all empty to run training locally as subprocesses (dev mode).
    # Set these in .env (or as EC2 environment variables) for the live demo.
    aws_region: str = "us-east-1"
    ecs_cluster: str = ""               # e.g. "bias-detector-cluster"
    ecs_task_definition: str = "bias-train-task"
    ecs_container_name: str = "train"
    ecs_subnet_ids: str = ""            # comma-separated: "subnet-abc,subnet-def"
    ecs_security_group_ids: str = ""    # comma-separated: "sg-abc,sg-def"
    log_group: str = "/ecs/bias-train"
    log_stream_prefix: str = "ecs"      # must match awslogs-stream-prefix in task def

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
