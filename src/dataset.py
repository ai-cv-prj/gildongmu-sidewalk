"""
file_path: src/dataset.py

XML과 이미지를 읽고 3클래스 Mask2Former 학습 데이터로 변환한다.
"""

import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import IMAGE_SIZE, LABEL2ID, POLYGON_LABEL_IDS, SEED


# XML 좌표 변환
def parse_points(points_text):
    """
    XML의 polygon 좌표 문자열을 OpenCV 좌표 배열로 변환한다.
    """
    return np.array(
        [
            [round(float(x)), round(float(y))]
            for x, y in (point.split(",") for point in points_text.split(";"))
        ],
        dtype=np.int32,
    )


# 데이터(원본이미지+정답마스크) 불러오기
def load_data(data_dir):
    """
    XML에서 이미지 경로와 polygon(정답) 정보를 불러온다.
    """
    samples = []
    for xml_path in sorted(data_dir.glob("*/*.xml")):
        for image_node in ET.parse(xml_path).getroot().findall("image"):
            samples.append(
                {
                    "group": xml_path.parent.name,
                    "image_path": xml_path.parent / image_node.get("name"),
                    "width": int(image_node.get("width")),
                    "height": int(image_node.get("height")),
                    "polygons": [
                        {
                            "label": polygon.get("label"),
                            "attribute": polygon.findtext(
                                "attribute[@name='attribute']", default=""
                            ).strip(),
                            "z_order": int(polygon.get("z_order", 0)),
                            "points": parse_points(polygon.get("points")),
                        }
                        for polygon in image_node.findall("polygon")
                    ],
                }
            )
    return samples


# 데이터 분리
def split_data(samples):
    """
    연속 촬영 이미지가 섞이지 않도록 폴더 단위로 8:1:1 분리한다.
    """
    groups = sorted({sample["group"] for sample in samples})
    random.Random(SEED).shuffle(groups)
    train_end = int(len(groups) * 0.8)
    validation_end = int(len(groups) * 0.9)
    train_groups = set(groups[:train_end])
    validation_groups = set(groups[train_end:validation_end])
    test_groups = set(groups[validation_end:])

    train_samples = [sample for sample in samples if sample["group"] in train_groups]
    validation_samples = [sample for sample in samples if sample["group"] in validation_groups]
    test_samples = [sample for sample in samples if sample["group"] in test_groups]
    return train_samples, validation_samples, test_samples


# 클래스별 정답 마스크 생성
def create_segmentation_mask(sample):
    """
    XML의 라벨과 속성을 3개 클래스 번호로 변환하는 함수.
    """
    mask = np.full(
        (sample["height"], sample["width"]),
        LABEL2ID["non_walkable"],
        dtype=np.uint8,
    )
    for polygon in sorted(sample["polygons"], key=lambda item: item["z_order"]):
        value = POLYGON_LABEL_IDS.get(
            (polygon["label"], polygon.get("attribute", "")),
            LABEL2ID["non_walkable"],
        )
        cv2.fillPoly(mask, [polygon["points"]], value)

    return cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


# Mask2Former 학습 데이터셋
class WalkableDataset(Dataset):
    """
    원본 이미지와 3클래스 정답 마스크를 Mask2Former 형식으로 제공한다.
    """

    # 데이터셋 초기화
    def __init__(self, samples, processor, augment=False):
        """
        사용할 샘플, 전처리기, 증강 여부를 저장한다.
        """
        self.samples = samples
        self.processor = processor
        self.augment = augment

    # 데이터 개수 반환
    def __len__(self):
        """
        데이터셋의 이미지 개수를 반환한다.
        """
        return len(self.samples)

    # 학습 데이터 한 개 반환
    def __getitem__(self, index):
        """
        이미지와 Mask2Former 형식의 정답 데이터를 반환한다.
        """
        sample = self.samples[index]
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            raise FileNotFoundError(
                f"이미지를 읽을 수 없습니다: {sample['image_path']}"
            )

        semantic_mask = create_segmentation_mask(sample)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            semantic_mask = np.ascontiguousarray(semantic_mask[:, ::-1])

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoded = self.processor(
            images=rgb_image,
            segmentation_maps=semantic_mask,
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"][0],
            "pixel_mask": encoded["pixel_mask"][0],
            "mask_labels": encoded["mask_labels"][0],
            "class_labels": encoded["class_labels"][0],
            "semantic_map": torch.from_numpy(semantic_mask.astype(np.int64)),
        }


# 가변 길이 정답 묶기
def collate_batch(items):
    """
    Mask2Former의 마스크 목록을 유지하면서 배치 데이터를 묶는다.
    """
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "pixel_mask": torch.stack([item["pixel_mask"] for item in items]),
        "mask_labels": [item["mask_labels"] for item in items],
        "class_labels": [item["class_labels"] for item in items],
        "semantic_maps": torch.stack([item["semantic_map"] for item in items]),
    }
