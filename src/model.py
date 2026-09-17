"""
file_path: src/model.py

Mask2Former 전처리기와 모델의 생성 및 로딩을 담당한다.
"""

from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)

from src.config import ID2LABEL, IMAGE_SIZE, LABEL2ID, MODEL_NAME


# 이미지 전처리기 생성
def create_processor():
    """
    이미지와 정답 마스크를 Mask2Former가 받을 수 있는 형태로 바꿔주는 전처리기를 만든다.
    """
    return AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        do_resize=True,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        do_reduce_labels=False,
    )


# 파인튜닝 모델 생성
def create_model(device):
    """
    설정의 클래스 수에 맞는 Mask2Former 모델을 생성하는 함수.
    """
    model_config = Mask2FormerConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = len(ID2LABEL)
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        config=model_config,
        ignore_mismatched_sizes=True,
    ).to(device)
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


# 저장된 모델 로딩
def load_model(model_dir, device):
    """
    저장된 파인튜닝 모델을 불러와 추론 모드로 설정한다.
    """
    return (
        Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
        .to(device)
        .eval()
    )
