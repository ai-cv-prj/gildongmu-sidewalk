"""
file_path: tests/test_video_inference.py

파인튜닝 모델을 불러와 보행가능 영역은 초록색, 횡단보도는 핑크색으로 표시한다.
"""

from pathlib import Path

import cv2
import torch
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


# 기본 경로
PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "outputs" / "run"
SAMPLE_DIR = PROJECT_DIR / "data" / "samples" / "sample1"
LABEL_COLORS = {
    "walkable": (0, 255, 0),
    "crosswalk": (180, 105, 255),  # OpenCV BGR 순서
}


# 최근 모델 탐색
def find_latest_model():
    """
    outputs/run에서 가장 최근에 저장된 파인튜닝 모델 폴더를 반환한다.
    """
    model_dirs = [
        model_dir
        for model_dir in RUNS_DIR.glob("*/model")
        if (model_dir / "config.json").is_file()
    ]
    if not model_dirs:
        raise FileNotFoundError(
            "저장된 모델이 없습니다. "
            "먼저 python -m scripts.run_finetune을 실행하세요."
        )
    return max(model_dirs, key=lambda model_dir: model_dir.stat().st_mtime)


# 샘플 영상 탐색
def find_sample_videos(sample_dir=SAMPLE_DIR):
    """
    지정 폴더 바로 아래의 모든 MP4 경로를 이름순으로 반환
    """
    sample_dir = Path(sample_dir).resolve()
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"샘플 폴더가 없습니다: {sample_dir}")
    video_paths = sorted(
        path for path in sample_dir.glob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not video_paths:
        raise FileNotFoundError(f"샘플 MP4 영상이 없습니다: {sample_dir}")
    return video_paths


# 저장된 모델의 라벨 번호 조회
def get_segmentation_label_ids(model):
    """
    모델 설정에서 3개 클래스 번호를 조회하고 누락 여부를 확인하는 함수.
    """
    label_ids = {
        label_name: int(label_id)
        for label_id, label_name in model.config.id2label.items()
    }
    required = {"non_walkable", "walkable", "crosswalk"}
    if not required.issubset(label_ids):
        raise ValueError(
            "3클래스 가중치가 필요합니다. "
            "수정된 코드로 파인튜닝한 모델을 --model-dir로 지정하세요."
        )
    return label_ids


# 클래스별 반투명 색상 표시
def overlay_segmentation(frame, class_map, label_ids):
    """
    보행가능·횡단보도만 반투명으로 칠하고 나머지는 유지하는 함수.
    """
    result = frame.copy()
    for label_name, color in LABEL_COLORS.items():
        mask = class_map == label_ids[label_name]
        overlay = frame.copy()
        overlay[mask] = color
        blended = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)
        result[mask] = blended[mask]
    return result


# 한 프레임 추론
def segment_frame(frame, processor, model, device, label_ids):
    """
    한 프레임을 추론해 보행가능과 횡단보도를 구분하여 표시하는 함수.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_frame, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    class_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[frame.shape[:2]],
    )[0].cpu().numpy()
    return overlay_segmentation(frame, class_map, label_ids)


# 영상 전체 추론
def process_video(
    video_path,
    output_path,
    processor,
    model,
    device,
    label_ids,
):
    """
    입력 영상 전체의 보행가능·횡단보도를 표시한 MP4를 저장하는 함수.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"결과 영상을 생성할 수 없습니다: {output_path}")

    processed_frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = segment_frame(
                frame,
                processor,
                model,
                device,
                label_ids,
            )
            writer.write(result)
            processed_frames += 1
            print(
                f"\r영상 처리: {processed_frames}/{total_frames}",
                end="",
                flush=True,
            )
    finally:
        capture.release()
        writer.release()

    print(f"\n결과 영상 저장: {output_path}")


# 영상 추론 테스트 수행
def run_video_inference(model_dir=None, video_path=None, output_path=None, sample_dir=None):
    """
    지정 영상 또는 샘플 폴더의 모든 MP4 추론, 모델은 한 번 로딩
    """
    # 상대 경로는 현재 작업 폴더 기준
    if video_path is not None and sample_dir is not None:
        raise ValueError("--video-path와 --sample-dir은 동시에 지정할 수 없습니다.")
    model_dir = Path(model_dir).resolve() if model_dir is not None else find_latest_model()
    video_paths = (
        [Path(video_path).resolve()]
        if video_path is not None
        else find_sample_videos(sample_dir if sample_dir is not None else SAMPLE_DIR)
    )

    if not model_dir.is_dir():
        raise FileNotFoundError(f"모델 폴더가 없습니다: {model_dir}")
    if output_path is not None and len(video_paths) > 1:
        raise ValueError("--output-path는 영상 한 개 처리 시에만 지정할 수 있습니다.")

    # 실행별 기본 영상 폴더 생성
    video_dir = model_dir.parent / "video"
    if output_path is None:
        video_dir.mkdir(exist_ok=True)

    # 전체 입출력 경로 확인 후 모델 로딩
    output_paths = [
        Path(output_path).resolve() if output_path is not None
        else video_dir / f"{video.stem}_walkable.mp4"
        for video in video_paths
    ]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("결과 파일명이 겹칩니다. 입력 영상 이름을 구분해 주세요.")
    for video, output in zip(video_paths, output_paths):
        if output.suffix.lower() != ".mp4":
            raise ValueError(f"출력 확장자는 .mp4여야 합니다: {output}")
        if not video.is_file():
            raise FileNotFoundError(f"입력 영상이 없습니다: {video}")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"출력 폴더가 없습니다: {output.parent}")
        if output.exists():
            raise FileExistsError(f"결과 영상이 이미 있습니다: {output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
        .to(device)
        .eval()
    )
    label_ids = get_segmentation_label_ids(model)

    print(f"추론 장치: {device}")
    print(f"모델: {model_dir}")
    for index, (video, output) in enumerate(zip(video_paths, output_paths), start=1):
        print(f"입력 영상 [{index}/{len(video_paths)}]: {video}")
        process_video(video, output, processor, model, device, label_ids)
