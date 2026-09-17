"""
file_path: scripts/run_video_inference.py

파인튜닝된 모델의 샘플 영상 추론을 시작하는 실행 파일.

실행: 
python -m scripts.run_video_inference # 기본값 : 최근 모델 가중치, sample1 폴더
python -m scripts.run_video_inference --model-dir outputs/run/20260916_163531_e100_b4/model --sample-dir data/samples/sample2
"""

import argparse
from pathlib import Path

from tests.test_video_inference import run_video_inference


# 영상 추론 시작
def main():
    """
    경로 인자 입력 및 영상 추론 시작
    """
    parser = argparse.ArgumentParser(description="파인튜닝 모델의 영상 추론")
    parser.add_argument("--model-dir", type=Path, help="가중치와 설정이 들어 있는 model 폴더")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video-path", type=Path, help="입력 영상 한 개")
    inputs.add_argument("--sample-dir", type=Path, help="입력 MP4 폴더 (기본: data/samples/sample1)")
    parser.add_argument("--output-path", type=Path, help="영상 한 개의 결과 MP4 파일 (생략: 실행 폴더의 video/)")
    args = parser.parse_args()
    run_video_inference(
        model_dir=args.model_dir,
        video_path=args.video_path,
        output_path=args.output_path,
        sample_dir=args.sample_dir,
    )


if __name__ == "__main__":
    main()
