"""
file_path: tests/test_three_class_segmentation.py

3클래스 정답 마스크, IoU 평가, 조기종료와 영상 색상을 검증하는 모듈.
실제 학습이나 파일 저장 없이 작은 입력과 모의 모델을 사용한다.
"""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from src import dataset, evaluation, finetuning, model, utils
from src.config import ID2LABEL, LABEL2ID
from tests.test_video_inference import (
    LABEL_COLORS,
    get_segmentation_label_ids,
    overlay_segmentation,
)


# 사각형 정답 생성
def make_polygon(label, attribute, bounds, z_order):
    """
    테스트용 사각형 폴리곤을 만드는 함수.
    """
    x1, y1, x2, y2 = bounds
    return {
        "label": label,
        "attribute": attribute,
        "z_order": z_order,
        "points": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32),
    }


# 3클래스 처리 검증
class ThreeClassTests(unittest.TestCase):
    """
    라벨 변환부터 평가·저장·영상 표시까지 확인하는 테스트 모음.
    """

    # 라벨과 겹침 순서 확인
    def test_mask_labels_and_overlap(self):
        """
        차도·골목 횡단보도와 다른 폴리곤의 겹침 처리를 확인하는 테스트.
        """
        sample = {
            "width": 6,
            "height": 6,
            "polygons": [
                make_polygon("sidewalk", "blocks", (0, 0, 5, 5), 0),
                make_polygon("roadway", "crosswalk", (0, 0, 1, 5), 1),
                make_polygon("alley", "crosswalk", (4, 0, 5, 5), 2),
                make_polygon("roadway", "normal", (4, 4, 5, 5), 3),
                make_polygon("caution_zone", "manhole", (0, 4, 1, 5), 4),
            ],
        }
        with patch.object(dataset, "IMAGE_SIZE", 6):
            mask = dataset.create_segmentation_mask(sample)
        self.assertEqual(set(np.unique(mask)), {0, 1, 2})
        self.assertEqual(mask[0, 0], 2)
        self.assertEqual(mask[0, 4], 2)
        self.assertEqual(mask[3, 2], 1)
        self.assertEqual(mask[5, 5], 0)
        self.assertEqual(mask[5, 0], 1)

    # 정답이 없는 영역 확인
    def test_empty_polygons_keep_background(self):
        """
        폴리곤이 없는 이미지는 기존처럼 보행불가능 마스크인지 확인하는 테스트.
        """
        mask = dataset.create_segmentation_mask({"width": 4, "height": 4, "polygons": []})
        self.assertTrue(np.all(mask == LABEL2ID["non_walkable"]))

    # 데이터 전체 합산 IoU 확인
    def test_evaluation_aggregates_all_pixels(self):
        """
        배치 평균이 아닌 전체 교집합·합집합으로 IoU를 계산하는지 확인하는 테스트.
        """
        fake_model = Mock()
        fake_model.side_effect = [SimpleNamespace(loss=torch.tensor(2.0)), SimpleNamespace(loss=torch.tensor(4.0))]
        processor = Mock()
        processor.post_process_semantic_segmentation.side_effect = [
            [torch.tensor([[0, 1, 2, 2, 0, 1]])],
            [torch.tensor([[0, 0, 0, 0]])],
        ]
        loader = [
            {"semantic_maps": [torch.tensor([[0, 1, 2, 1, 2, 0]])]},
            {"semantic_maps": [torch.tensor([[0, 0, 0, 0]])]},
        ]
        with patch.object(evaluation, "move_batch_to_device", return_value={}):
            loss, scores, miou = evaluation.evaluate(fake_model, loader, processor, "cpu", "테스트", False)
        self.assertEqual(loss, 3.0)
        self.assertAlmostEqual(scores["non_walkable"], 5 / 7)
        self.assertAlmostEqual(scores["walkable"], 1 / 3)
        self.assertAlmostEqual(scores["crosswalk"], 1 / 3)
        self.assertAlmostEqual(miou, (5 / 7 + 1 / 3 + 1 / 3) / 3)

    # 합집합이 없는 클래스 확인
    def test_absent_class_is_zero_in_three_class_mean(self):
        """
        정답·예측이 모두 없는 클래스의 IoU를 0으로 포함하는지 확인하는 테스트.
        """
        fake_model = Mock(return_value=SimpleNamespace(loss=torch.tensor(0.0)))
        processor = Mock()
        processor.post_process_semantic_segmentation.return_value = [torch.tensor([[0, 1]])]
        loader = [{"semantic_maps": [torch.tensor([[0, 1]])]}]
        with patch.object(evaluation, "move_batch_to_device", return_value={}):
            _, scores, miou = evaluation.evaluate(fake_model, loader, processor, "cpu", "테스트", False)
        self.assertEqual(scores["crosswalk"], 0.0)
        self.assertAlmostEqual(miou, 2 / 3)

    # 모델 클래스 수 확인
    def test_model_uses_configured_class_count(self):
        """
        모델 생성 시 3개 클래스와 라벨 이름이 전달되는지 확인하는 테스트.
        """
        config = SimpleNamespace()
        fake_model = Mock(supports_gradient_checkpointing=False)
        with (
            patch.object(model.Mask2FormerConfig, "from_pretrained", return_value=config),
            patch.object(model.Mask2FormerForUniversalSegmentation, "from_pretrained") as factory,
        ):
            factory.return_value.to.return_value = fake_model
            model.create_model(torch.device("cpu"))
        self.assertEqual(config.num_labels, 3)
        self.assertEqual(config.id2label, ID2LABEL)

    # 색상과 원본 유지 확인
    def test_overlay_colors_and_unchanged_background(self):
        """
        모델 라벨 번호에 맞는 두 색상과 보행불가능 영역 보존을 확인하는 테스트.
        """
        fake_model = SimpleNamespace(config=SimpleNamespace(id2label={7: "non_walkable", 5: "walkable", 2: "crosswalk"}))
        label_ids = get_segmentation_label_ids(fake_model)
        frame = np.full((1, 3, 3), 80, dtype=np.uint8)
        original = frame.copy()
        result = overlay_segmentation(frame, np.array([[7, 5, 2]]), label_ids)
        np.testing.assert_array_equal(result[0, 0], original[0, 0])
        for index, name in ((1, "walkable"), (2, "crosswalk")):
            color = np.full((1, 1, 3), LABEL_COLORS[name], dtype=np.uint8)
            expected = cv2.addWeighted(original[:, index:index + 1], 0.45, color, 0.55, 0)
            np.testing.assert_array_equal(result[:, index:index + 1], expected)
        np.testing.assert_array_equal(frame, original)

    # 구형 가중치 안내 확인
    def test_two_class_weights_report_clear_error(self):
        """
        횡단보도 클래스가 없는 가중치에 명확한 오류를 표시하는 테스트.
        """
        fake_model = SimpleNamespace(config=SimpleNamespace(id2label={0: "non_walkable", 1: "walkable"}))
        with self.assertRaisesRegex(ValueError, "3클래스 가중치"):
            get_segmentation_label_ids(fake_model)

    # mIoU 기준 저장·조기종료 확인
    def test_early_stopping_and_saving_use_miou(self):
        """
        보행 IoU가 상승해도 mIoU가 낮으면 저장하지 않고 조기종료하는 테스트.
        """
        fake_model = torch.nn.Linear(1, 1)
        fake_model.save_pretrained = Mock()
        processor = Mock()
        scores = [(.9, .3, .6), (.85, .7, .85), (.95, .6, .7), (.96, .64, .74), (.94, .62, .72)]
        evaluations = [
            (0.0, dict(zip(("walkable", "crosswalk", "non_walkable"), values)), sum(values) / 3)
            for values in scores
        ]
        with (
            patch.object(finetuning, "EPOCHS", 10),
            patch.object(finetuning, "EARLY_STOPPING_PATIENCE", 3),
            patch.object(finetuning, "train_one_epoch", return_value=1.0),
            patch.object(finetuning, "evaluate", side_effect=evaluations) as evaluate_mock,
            patch.object(finetuning, "print_progress") as progress,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            finetuning.finetune_model(fake_model, processor, [], [], torch.device("cpu"), "unused")
        self.assertEqual(evaluate_mock.call_count, 5)
        self.assertEqual(fake_model.save_pretrained.call_count, 2)
        self.assertEqual(processor.save_pretrained.call_count, 2)
        self.assertIn("개선 없음 3/3 (조기 종료)", progress.call_args.args[0])

    # 완료 로그 잘림 방지 확인
    def test_finished_log_is_not_truncated(self):
        """
        터미널이 좁아도 완료 로그의 클래스별 점수가 사라지지 않는지 확인하는 테스트.
        """
        output = io.StringIO()
        message = evaluation.format_class_ious({"walkable": .8, "crosswalk": .6, "non_walkable": .9})
        with (
            patch("sys.stdout", output),
            patch.object(output, "isatty", return_value=True),
            patch.object(utils.shutil, "get_terminal_size", return_value=SimpleNamespace(columns=20)),
        ):
            utils.print_progress(message, finished=True)
        self.assertIn(message, output.getvalue())


if __name__ == "__main__":
    unittest.main()
