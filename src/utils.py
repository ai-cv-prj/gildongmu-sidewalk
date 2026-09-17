"""
file_path: src/utils.py

파인튜닝 과정에서 공통으로 사용하는 보조 기능을 제공한다.
"""

import random
import shutil
import sys
import unicodedata

import numpy as np
import torch

from src.config import SEED


# 터미널 진행 상황 한 줄 출력
def print_progress(message, finished=False):
    """
    터미널에서는 한 줄을 지우고 갱신하며, 로그 파일에는 완료 결과만 남긴다.
    """
    if not sys.stdout.isatty():
        if finished:
            print(message, flush=True)
        return

    # 완료 로그는 클래스별 평가값을 자르지 않고 전체 출력
    if finished:
        print("\r\033[2K" + message, flush=True)
        return

    # 한글 표시 폭을 반영하고 마지막 열을 비워 자동 줄바꿈을 방지한다.
    limit = max(1, shutil.get_terminal_size().columns - 1)
    width = 0
    visible = ""
    for char in message:
        char_width = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if width + char_width > limit:
            break
        visible += char
        width += char_width
    print("\r\033[2K" + visible, end="\n" if finished else "", flush=True)


# 시간 표시 형식 변환
def format_duration(seconds):
    """
    초 단위 시간을 시:분:초 형식의 문자열로 변환한다.
    """
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# 난수 고정
def set_seed():
    """
    데이터 분리와 증강 결과를 재현할 수 있도록 난수를 고정한다.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
