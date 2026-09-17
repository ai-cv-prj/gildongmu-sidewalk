"""
file_path: src/finetuning.py

데이터 준비, Mask2Former 파인튜닝, 최종 평가의 전체 흐름을 관리한다.
"""

from datetime import datetime
import gc
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from transformers.utils import logging as transformers_logging

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_ROOT,
)
from src.dataset import (
    WalkableDataset,
    collate_batch,
    load_data,
    split_data,
)
from src.evaluation import evaluate, format_class_ious
from src.model import create_model, create_processor, load_model
from src.training import train_one_epoch
from src.utils import format_duration, print_progress, set_seed


# 데이터 로더 생성
def create_data_loaders(processor, device):
    """
    전체 샘플을 분리하고 학습, 검증, 테스트 데이터 로더를 생성한다.
    """
    samples = load_data(DATA_DIR)
    train_samples, validation_samples, test_samples = split_data(samples)
    print(
        f"데이터: 학습 {len(train_samples)}, "
        f"검증 {len(validation_samples)}, 테스트 {len(test_samples)}"
    )

    train_dataset = WalkableDataset(train_samples, processor, augment=True)
    validation_dataset = WalkableDataset(validation_samples, processor)
    test_dataset = WalkableDataset(test_samples, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    return train_loader, validation_loader, test_loader


# 모델 파인튜닝
def finetune_model(
    model,
    processor,
    train_loader,
    validation_loader,
    device,
    model_output_dir,
):
    """
    최고 검증 mIoU 모델을 저장하고 개선이 멈추면 조기 종료하는 함수.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    use_fp16_scaler = (
        device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    best_miou = -1.0
    epochs_without_improvement = 0
    training_started_at = perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_started_at = perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
        )
        print_progress(f"🔶 Epoch {epoch}/{EPOCHS} | 검증 중...")
        _, validation_ious, validation_miou = evaluate(
            model,
            validation_loader,
            processor,
            device,
            "검증",
            verbose=False,
        )
        if validation_miou > best_miou:
            best_miou = validation_miou
            epochs_without_improvement = 0
            # 가중치 저장 중 라이브러리 진행 막대만 숨긴 뒤 기존 설정을 복원한다.
            progress_enabled = transformers_logging.is_progress_bar_enabled()
            transformers_logging.disable_progress_bar()
            try:
                model.save_pretrained(model_output_dir)
                processor.save_pretrained(model_output_dir)
            finally:
                if progress_enabled:
                    transformers_logging.enable_progress_bar()
            status = "가중치 저장"
        else:
            epochs_without_improvement += 1
            status = (
                f"개선 없음 {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )

        should_stop = epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        if should_stop:
            status += " (조기 종료)"
        epoch_elapsed = perf_counter() - epoch_started_at
        print_progress(
            f"🔶 Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | "
            f"검증 mIoU={validation_miou:.4f} | "
            f"{format_class_ious(validation_ious)} | "
            f"소요 시간={format_duration(epoch_elapsed)} | {status}",
            finished=True,
        )
        if should_stop:
            break

    training_elapsed = perf_counter() - training_started_at
    print(
        "파인튜닝 및 검증 총 소요 시간: "
        f"{format_duration(training_elapsed)}"
    )


# 메모리 정리
def clear_device_cache(device):
    """
    사용하지 않는 CPU와 GPU 캐시 메모리를 정리한다.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# 파인튜닝 전체 과정 수행
def run_finetuning():
    """
    데이터 준비, 파인튜닝, 테스트 데이터 평가를 순서대로 수행한다.
    """
    run_started_at = perf_counter()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"학습 데이터 폴더가 없습니다: {DATA_DIR}")
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"결과 폴더가 없습니다: {OUTPUT_ROOT}")

    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"e{EPOCHS}_b{BATCH_SIZE}"
    )
    run_output_dir = OUTPUT_ROOT / run_name
    model_output_dir = run_output_dir / "model"
    run_output_dir.mkdir()
    model_output_dir.mkdir()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")
    print("라벨: 0=보행불가능, 1=보행가능, 2=횡단보도")
    print("모델 선택·조기 종료 기준: 검증 mIoU (3개 클래스 IoU 평균)")

    processor = create_processor()
    train_loader, validation_loader, test_loader = create_data_loaders(
        processor,
        device,
    )
    model = create_model(device)
    finetune_model(
        model,
        processor,
        train_loader,
        validation_loader,
        device,
        model_output_dir,
    )
    del model
    clear_device_cache(device)

    best_model = load_model(model_output_dir, device)
    evaluate(best_model, test_loader, processor, device, "테스트")

    total_elapsed = perf_counter() - run_started_at
    print(f"전체 실행 소요 시간: {format_duration(total_elapsed)}")
