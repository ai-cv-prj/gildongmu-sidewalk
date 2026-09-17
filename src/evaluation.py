"""
file_path: src/evaluation.py

파인튜닝 모델의 loss, 클래스별 IoU와 mIoU를 계산한다.
"""

import torch

from src.config import ID2LABEL, IMAGE_SIZE
from src.training import move_batch_to_device
from src.utils import print_progress


# 클래스별 IoU 출력 형식
def format_class_ious(class_ious):
    """
    보행가능·횡단보도·보행불가능 순서로 IoU 문자열을 만드는 함수.
    """
    return (
        f"IoU(보행/횡단/불가)={class_ious['walkable']:.4f}/"
        f"{class_ious['crosswalk']:.4f}/{class_ious['non_walkable']:.4f}"
    )


# 모델 성능 평가
def evaluate(model, loader, processor, device, name, verbose=True):
    """
    전체 데이터의 클래스별 교집합·합집합으로 IoU와 3클래스 평균을 계산하는 함수.
    """
    model.eval()
    total_loss = 0.0
    intersections = {label_id: 0 for label_id in ID2LABEL}
    unions = {label_id: 0 for label_id in ID2LABEL}
    if len(loader) == 0:
        raise ValueError(f"{name} 데이터가 비어 있습니다.")

    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            model_inputs = move_batch_to_device(batch, device)
            outputs = model(**model_inputs)
            total_loss += outputs.loss.item()
            predictions = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(IMAGE_SIZE, IMAGE_SIZE)]
                * len(batch["semantic_maps"]),
            )

            for prediction, target in zip(predictions, batch["semantic_maps"]):
                prediction = prediction.cpu()
                target = target.cpu()
                for label_id in ID2LABEL:
                    predicted_class = prediction == label_id
                    target_class = target == label_id
                    intersections[label_id] += (predicted_class & target_class).sum().item()
                    unions[label_id] += (predicted_class | target_class).sum().item()

            if verbose:
                print_progress(f"{name}: {step}/{len(loader)}")

    average_loss = total_loss / len(loader)
    # 평가 전체에서 정답·예측이 모두 없는 클래스는 IoU 0으로 포함
    class_ious = {
        label_name: intersections[label_id] / unions[label_id] if unions[label_id] else 0.0
        for label_id, label_name in ID2LABEL.items()
    }
    miou = sum(class_ious.values()) / len(class_ious)
    if verbose:
        print_progress(
            f"{name} | loss={average_loss:.4f} | mIoU={miou:.4f} | "
            f"{format_class_ious(class_ious)}",
            finished=True,
        )
    return average_loss, class_ious, miou
