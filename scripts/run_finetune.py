"""
file_path: scripts/run_finetune.py

Mask2Former 파인튜닝을 시작하는 실행 파일.

실행: python -m scripts.run_finetune
"""

from src.finetuning import run_finetuning


# 파인튜닝 시작
def main():
    """
    파인튜닝 전체 과정을 시작한다.
    """
    run_finetuning()


if __name__ == "__main__":
    main()
