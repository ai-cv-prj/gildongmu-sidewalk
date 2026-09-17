"""
file_path: src/config.py

YAML 설정 파일을 읽어 파인튜닝 코드에서 사용할 값으로 제공한다.
"""

from pathlib import Path

import yaml


# 프로젝트 경로
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "mask2former_surface1.yaml"


# YAML 설정 불러오기
def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    YAML 설정 파일을 읽고 필수 설정 그룹이 있는지 확인한다.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required_sections = {"paths", "model", "labels", "training"}
    missing_sections = required_sections - set(config or {})
    if missing_sections:
        missing_names = ", ".join(sorted(missing_sections))
        raise ValueError(f"필수 설정이 없습니다: {missing_names}")
    return config


# 설정 그룹 준비
CONFIG = load_config()
PATH_CONFIG = CONFIG["paths"]
MODEL_CONFIG = CONFIG["model"]
LABEL_CONFIG = CONFIG["labels"]
TRAINING_CONFIG = CONFIG["training"]

# 데이터와 출력 경로
DATA_DIR = PROJECT_DIR / PATH_CONFIG["data_dir"]
OUTPUT_ROOT = PROJECT_DIR / PATH_CONFIG["output_root"]

# 사전학습 모델
MODEL_NAME = MODEL_CONFIG["name"]

# 데이터셋 라벨
ID2LABEL = {
    int(label_id): label_name
    for label_id, label_name in LABEL_CONFIG["id_to_label"].items()
}
LABEL2ID = {label_name: label_id for label_id, label_name in ID2LABEL.items()}
POLYGON_LABEL_IDS = {
    (label, attribute): LABEL2ID[class_name]
    for class_name in ("walkable", "crosswalk")
    for label, attributes in LABEL_CONFIG[class_name].items()
    for attribute in attributes
}

# 학습 설정
IMAGE_SIZE = int(TRAINING_CONFIG["image_size"])
BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
EPOCHS = int(TRAINING_CONFIG["epochs"])
EARLY_STOPPING_PATIENCE = int(TRAINING_CONFIG["early_stopping_patience"])
if EARLY_STOPPING_PATIENCE < 1:
    raise ValueError("early_stopping_patience는 1 이상이어야 합니다.")
LEARNING_RATE = float(TRAINING_CONFIG["learning_rate"])
SEED = int(TRAINING_CONFIG["seed"])
