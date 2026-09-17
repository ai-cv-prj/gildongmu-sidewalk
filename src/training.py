"""
file_path: src/training.py

Mask2Former 모델의 배치 이동과 한 epoch 학습을 담당한다.
"""

from time import perf_counter

import torch

from src.config import EPOCHS
from src.utils import format_duration, print_progress


# 배치 데이터를 장치로 이동
def move_batch_to_device(batch, device):
    """
    모델 입력과 정답 데이터를 GPU 또는 CPU로 이동한다.
    """
    return {
        "pixel_values": batch["pixel_values"].to(device),
        "pixel_mask": batch["pixel_mask"].to(device),
        "mask_labels": [mask.to(device) for mask in batch["mask_labels"]],
        "class_labels": [label.to(device) for label in batch["class_labels"]],
    }


# 한 epoch 학습
def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    """
    한 epoch 학습, 진행 시간 표시, 평균 loss 반환
    """
    model.train()
    optimizer.zero_grad()   # 기울기 초기화
    total_loss = 0.0        # 누적 loss
    epoch_started_at = perf_counter()

    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    # 배치 단위 순회
    for step, batch in enumerate(loader, start=1):
        model_inputs = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp,):
            outputs = model(**model_inputs) # 모델 예측 
            loss = outputs.loss             # loss 계산

        scaler.scale(loss).backward()   # loss 스케일링 및 역전파
        scaler.step(optimizer)          # 매 배치 가중치 업데이트
        scaler.update()                 # loss 스케일 갱신
        optimizer.zero_grad()           # 기울기 초기화

        total_loss += loss.item() # 배치별 loss 합산

        # 진행 상황 표시
        elapsed = perf_counter() - epoch_started_at
        seconds_per_step = elapsed / step
        remaining = seconds_per_step * (len(loader) - step)
        print_progress(
            f"🔶 Epoch {epoch}/{EPOCHS} | {step}/{len(loader)} 배치 | "
            f"경과 시간={format_duration(elapsed)} | "
            f"현재 epoch 예상 남은 시간≈{format_duration(remaining)}"
        )

    # 평균 배치 loss 반환
    average_loss = total_loss / len(loader)
    return average_loss
