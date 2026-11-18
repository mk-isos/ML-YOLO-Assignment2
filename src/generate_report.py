#!/usr/bin/env python3
"""Generate the Korean assignment report strictly from measured artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PROJECT_ROOT / "results/metrics"
PLOTS_ROOT = PROJECT_ROOT / "results/plots"
PREDICTIONS_ROOT = PROJECT_ROOT / "results/predictions"
REPORTS_ROOT = PROJECT_ROOT / "reports"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
FINAL_REPORT_NAME = "머신러닝_과제2_2021041023_김문기.pdf"


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}시간 {minutes}분 {secs}초" if hours else f"{minutes}분 {secs}초"


def scaled_image(path: Path, max_width: float = 170 * mm, max_height: float = 105 * mm) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def page_footer(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setFont("Korean", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, 12 * mm, "충북대학교 머신러닝 과제 2 — YOLO Fine-tuning")
    canvas.drawRightString(190 * mm, 12 * mm, str(document.page))
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="김문기")
    parser.add_argument("--student-id", default="2021041023")
    parser.add_argument(
        "--github-url", default="https://github.com/mk-isos/ML-YOLO-Assignment2"
    )
    parser.add_argument("--output", type=Path, default=REPORTS_ROOT / FINAL_REPORT_NAME)
    args = parser.parse_args()

    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Korean font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont("Korean", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("KoreanBold", str(FONT_PATH)))

    dataset = read_json(METRICS_ROOT / "dataset_validation.json")
    environment = read_json(METRICS_ROOT / "environment.json")
    training = read_json(METRICS_ROOT / "training_summary.json")
    test = read_json(METRICS_ROOT / "test_metrics.json")
    tracking = read_json(METRICS_ROOT / "tracking_metadata.json")
    analysis = read_json(METRICS_ROOT / "additional_analysis.json")
    transfer_audit = read_json(METRICS_ROOT / "transfer_audit.json")
    pretrained_comparison = read_json(METRICS_ROOT / "pretrained_comparison.json")
    tracking_continuity = read_json(METRICS_ROOT / "tracking_continuity.json")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "KoreanTitle",
        parent=styles["Title"],
        fontName="KoreanBold",
        fontSize=24,
        leading=34,
        textColor=colors.HexColor("#17365D"),
        alignment=TA_CENTER,
        spaceAfter=14 * mm,
    )
    subtitle = ParagraphStyle(
        "KoreanSubtitle",
        parent=styles["Normal"],
        fontName="Korean",
        fontSize=12,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#444444"),
    )
    h1 = ParagraphStyle(
        "KoreanH1",
        parent=styles["Heading1"],
        fontName="KoreanBold",
        fontSize=17,
        leading=24,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=5 * mm,
        spaceAfter=4 * mm,
        keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "KoreanH2",
        parent=styles["Heading2"],
        fontName="KoreanBold",
        fontSize=13,
        leading=19,
        textColor=colors.HexColor("#2F5597"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "KoreanBody",
        parent=styles["BodyText"],
        fontName="Korean",
        fontSize=10.1,
        leading=16.2,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        spaceAfter=2.3 * mm,
    )
    small = ParagraphStyle(
        "KoreanSmall",
        parent=body,
        fontSize=8.7,
        leading=13.0,
        textColor=colors.HexColor("#555555"),
    )
    caption = ParagraphStyle(
        "KoreanCaption",
        parent=small,
        alignment=TA_CENTER,
        spaceBefore=1.5 * mm,
        spaceAfter=4 * mm,
    )
    bullet = ParagraphStyle(
        "KoreanBullet",
        parent=body,
        leftIndent=6 * mm,
        firstLineIndent=-3 * mm,
        bulletIndent=1 * mm,
        alignment=TA_LEFT,
    )
    compact_bullet = ParagraphStyle(
        "KoreanCompactBullet",
        parent=small,
        leftIndent=6 * mm,
        firstLineIndent=-3 * mm,
        bulletIndent=1 * mm,
        alignment=TA_LEFT,
        spaceAfter=1 * mm,
    )

    def p(text: str, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(text, style)

    def bullets(items: list[str]) -> list[Paragraph]:
        return [Paragraph(f"• {item}", bullet) for item in items]

    def compact_bullets(items: list[str]) -> list[Paragraph]:
        return [Paragraph(f"• {item}", compact_bullet) for item in items]

    def table(rows: list[list[object]], widths: list[float] | None = None) -> Table:
        converted = [
            [cell if isinstance(cell, Paragraph) else p(escape(str(cell)), small) for cell in row]
            for row in rows
        ]
        result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                    ("FONTNAME", (0, 0), (-1, -1), "Korean"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9EADBA")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return result

    story: list[object] = []
    story.extend(
        [
            Spacer(1, 35 * mm),
            p("YOLOv8 Fine-tuning을 이용한<br/>3개 부류 객체 탐지 및 추적", title),
            p("충북대학교 소프트웨어학부 · 머신러닝 과제 2", subtitle),
            Spacer(1, 25 * mm),
            table(
                [
                    ["항목", "내용"],
                    ["과목 / 담당", "머신러닝 / 이건명 교수"],
                    ["제출자", args.author],
                    ["학번", args.student_id],
                    ["실험 모델", "Ultralytics YOLOv8n (COCO 사전학습 → KITTI 3부류 fine-tuning)"],
                    ["작성일", datetime.now().strftime("%Y년 %m월 %d일")],
                ],
                [38 * mm, 122 * mm],
            ),
            Spacer(1, 35 * mm),
            p(
                "본 보고서의 수치와 그림은 저장된 학습 로그, 별도 테스트 분할 평가, 추가 진단 실험, "
                "추적 CSV 및 ffprobe 검증 결과에서 자동으로 불러왔다.",
                subtitle,
            ),
            PageBreak(),
        ]
    )

    story += [p("1. 과제 개요 및 목적", h1)]
    story += [
        p(
            "강의자료의 과제 원문은 ‘YOLO 모델을 3개 부류 데이터에 대해서 fine-tuning’하는 것이다. "
            "세부 제출 조건은 3개 부류의 100개 이상 이미지 수집, 학습, 탐지 및 추적, 약 5초 결과 영상, "
            "공개 GitHub의 모델·영상, 그리고 학습 과정·성능·경험을 포함한 결과보고서이다 "
            "(computer-vision-4.pdf, 49쪽). 50–51쪽은 라벨링 도구와 Ultralytics 실습 참고 링크이다. "
            "본 프로젝트는 단순 COCO 추론이 아니라, KITTI 라벨을 "
            "3개 부류로 재매핑하고 YOLOv8n의 사전학습 가중치에서 실제 추가 학습을 수행했다."
        ),
        p(
            "강의자료는 객체 탐지의 bounding box, IoU, NMS, YOLO 구조, 전이학습 및 precision/recall/mAP를 "
            "설명한다. 데이터 누수 방지용 장면 그룹화, SHA-256 검증, ByteTrack 사용, Apple MPS의 결정론 한계는 "
            "과제 명세를 충족하기 위해 추가 적용한 일반적인 머신러닝 엔지니어링 절차이다."
        ),
        p("강의자료 직접 요구사항과 충족 근거", h2),
        table(
            [
                ["49쪽 직접 요구사항", "실제 충족 근거"],
                ["3개 부류, 100개 이상 이미지", "car/pedestrian/cyclist; train 클래스별 이미지 4,908/1,076/794장"],
                ["3개 부류 모델 학습", "COCO 사전학습 YOLOv8n을 KITTI 3부류로 16 Epoch fine-tuning"],
                ["학습 부류 탐지 및 추적", "best.pt와 ByteTrack으로 세 클래스 검출·프레임 간 ID 추적"],
                ["약 5초 결과 동영상", "H.264 MP4 5.000초·150프레임·전체 디코딩 확인"],
                ["개인 GitHub에 모델·영상 공개", "mk-isos/ML-YOLO-Assignment2의 models/와 videos/"],
                ["과정·성능·링크·경험 보고", "본 보고서 3–13장과 독립 Test·추가 진단 결과"],
            ],
            [60 * mm, 108 * mm],
        ),
        p("최종 검증 목표", h2),
        *bullets(
            [
                "정확히 car / pedestrian / cyclist 3개 클래스만 정의하고 원본 클래스 ID를 감사한다.",
                "학습에 사용하지 않은 test 분할에서 Precision, Recall, mAP@0.5, mAP@0.5:0.95를 측정한다.",
                "학습된 best.pt로 프레임 간 ID가 유지되는 약 5초 MP4를 생성하고 재생·길이를 검증한다.",
                "코드, 설정, 로그, 가중치, 영상과 보고서가 서로 같은 실험을 가리키도록 한다.",
            ]
        ),
    ]

    story += [p("2. 데이터셋과 클래스 구성", h1)]
    splits = dataset["splits"]
    dataset_rows = [["분할", "이미지", "car 이미지/개체", "pedestrian 이미지/개체", "cyclist 이미지/개체"]]
    for split in ("train", "val", "test"):
        row = splits[split]
        dataset_rows.append(
            [
                split,
                row["images"],
                f"{row['class_images']['car']} / {row['class_instances']['car']}",
                f"{row['class_images']['pedestrian']} / {row['class_instances']['pedestrian']}",
                f"{row['class_images']['cyclist']} / {row['class_instances']['cyclist']}",
            ]
        )
    story += [
        p(
            "KITTI 2D 객체 탐지의 Ultralytics 변환본 7,481장을 사용했다. 원본 YAML과 전체 라벨을 검사한 결과 "
            "원본 클래스는 0 car, 1 van, 2 truck, 3 pedestrian, 4 person_sitting, 5 cyclist, 6 tram, 7 misc였다. "
            "이 가운데 0→0 car, 3→1 pedestrian, 5→2 cyclist로 조밀하게 재매핑하고 나머지는 제거했다. "
            "세 부류는 도로 장면에서 함께 등장해 탐지·추적 영상을 만들기 좋고, 자동차·보행자·자전거 이용자처럼 "
            "크기와 출현 빈도가 달라 모델의 한계를 분석하기에도 적합하다."
        ),
        table(dataset_rows, [19 * mm, 18 * mm, 42 * mm, 47 * mm, 42 * mm]),
        Spacer(1, 3 * mm),
        p("분할 및 누수 방지", h2),
        p(
            "배포된 탐지 데이터에는 촬영 시퀀스 ID가 없어 원본 train/val만 신뢰하면 유사 장면이 양쪽에 섞였다. "
            "실제로 가까운 프레임이 원본 분할을 가로지르는 사례를 확인했다. 이에 pHash, dHash, 축소 RGB 평균 절대오차와 "
            "상관계수를 함께 사용해 3,856개 장면 그룹을 구성하고, 같은 그룹 전체를 70/15/15로 배정했다. "
            "검증 결과 split 간 동일 SHA-256 이미지와 장면 그룹 중복은 0건이며 모든 라벨 좌표·클래스 ID가 유효했다."
        ),
        p(
            f"<b>한계:</b> {escape(str(dataset['known_limitation']))} 따라서 동일 주행의 서로 다른 시점처럼 "
            "시각적 유사도가 낮은 프레임까지 완전히 식별했다고 주장할 수는 없다."
        ),
    ]

    story += [p("3. 모델과 Fine-tuning 방법", h1)]
    story += [
        p(
            "강의자료의 YOLOv8 실습 흐름과 Ultralytics의 안정적인 학습·평가·추적 API를 고려해 YOLOv8n을 선택했다. "
            "M1 Pro 16 GB 환경과 7,481장 데이터에서 반복 실험 시간을 통제하기 위해 가장 작은 n 모델을 사용했다. "
            "YOLOv8은 한 번의 forward pass에서 다중 스케일 특징으로 bounding box와 클래스 확률을 예측하고, "
            "NMS로 중복 상자를 제거한다."
        ),
        p(
            "COCO 사전학습 가중치 yolov8n.pt를 불러온 뒤 출력 클래스를 80개에서 3개로 바꾸었다. "
            "설치된 Ultralytics 8.4.215의 DetectionTrainer는 대상 클래스 이름을 가중치 로드 전에 설정하고, "
            "DetectionModel._remap_cls_by_names가 이름이 정확히 일치하는 분류 출력 행을 옮긴다. 같은 버전과 원본 "
            "체크포인트로 로드만 재구성한 감사에서 ‘Remapped 1/3’과 세 탐지 스케일의 COCO car 행/바이어스가 "
            "KITTI car 행과 동일함을 확인했다. pedestrian과 cyclist는 COCO person/bicycle 행과 동일하지 않았다. "
            "따라서 ‘Transferred 355/355’ 문구 하나가 아니라 구현 감사와 텐서 비교를 근거로 car 출력 행 재사용을 판단했다."
        ),
        p(
            "그 후 3부류 데이터에서 box/class/DFL 손실을 계산하고 역전파와 AdamW 갱신을 실제 수행했다. "
            "16개 Epoch의 손실·검증 로그와 학습 전후 가중치 SHA-256이 서로 다르므로 기존 COCO 예측을 필터링한 것이 아니라 "
            "실제 fine-tuning이다. 전이 감사 과정은 저장된 best.pt를 수정하지 않았다."
        ),
        p("평가 정의", h2),
        *bullets(
            [
                "IoU = 예측 상자와 정답 상자의 교집합 면적 / 합집합 면적",
                "Precision = TP / (TP + FP), Recall = TP / (TP + FN)",
                "mAP@0.5는 IoU 0.5의 클래스별 AP 평균, mAP@0.5:0.95는 0.50–0.95 임계값 평균",
                "정식 수치는 Ultralytics val의 confidence sweep 결과이며, 사례 그림은 conf 0.25·IoU 0.5 매칭으로 별도 선별",
            ]
        ),
    ]

    story += [p("4. 학습 환경과 하이퍼파라미터", h1)]
    params = [
        ["구분", "설정 / 측정값"],
        ["하드웨어", "MacBook Pro 18,3 · Apple M1 Pro (8 CPU / 14 GPU) · 16 GB"],
        ["운영체제", environment["platform"]],
        ["Python / PyTorch", f"{environment['python']} / {environment['torch']}"],
        ["Ultralytics", environment["ultralytics"]],
        ["가속기", f"MPS built={environment['mps_built']}, available={environment['mps_available']}"],
        ["모델 / 입력", "YOLOv8n / 640×640"],
        ["Epoch(최대 / 실제) / batch", f"{training['configured_epochs']} / {training['completed_epochs']} / 8"],
        ["Optimizer", "AdamW, lr0=0.001, lrf=0.01, weight_decay=0.0005"],
        ["증강", "mosaic=1.0, HSV, horizontal flip=0.5, scale/translate; 마지막 10 Epoch mosaic 종료"],
        ["재현 설정", "seed=42, deterministic=True, patience=8"],
    ]
    story += [
        table(params, [42 * mm, 126 * mm]),
        Spacer(1, 3 * mm),
        p(
            f"MPS에서 convolution forward/backward를 실제 실행했으며 손실 {environment['mps_backward_loss']:.6f}, "
            f"초기 점검 시간 {environment['mps_backward_seconds']:.3f}초였다. 72장/30장의 1 Epoch smoke test를 먼저 통과한 뒤, "
            "전체 train/val 1 Epoch 벤치마크가 약 4분 3초여서 30 Epoch를 약 2시간으로 추정했다. 50 Epoch를 기계적으로 "
            "지정하지 않고, 30 Epoch와 early stopping patience 8을 선택했다."
        ),
        p(
            "PyTorch는 MPS의 scatter_reduce와 accumulate index_put 연산이 deterministic 구현이 아니라고 경고했다. "
            "따라서 seed와 deterministic 옵션을 고정했더라도 GPU 수준의 비트 단위 완전 재현은 보장하지 않는다. "
            "대신 환경 버전, 데이터 SHA-256, 분할 manifest, 설정과 최종 가중치 SHA-256을 기록했다."
        ),
    ]

    story += [p("5. 학습 과정과 검증 성능", h1)]
    story += [
        p(
            f"전체 학습은 {training['completed_epochs']} Epoch를 완료했고 실제 소요 시간은 "
            f"{format_duration(float(training['duration_seconds']))}였다. 검증 mAP@0.5:0.95가 가장 높은 Epoch는 "
            f"{training['best_epoch']}였으며, 그 시점의 검증 Precision={training['best_validation']['precision']:.4f}, "
            f"Recall={training['best_validation']['recall']:.4f}, mAP@0.5={training['best_validation']['map50']:.4f}, "
            f"mAP@0.5:0.95={training['best_validation']['map50_95']:.4f}였다. 최대 설정은 "
            f"{training['configured_epochs']} Epoch, patience는 8이었다. Ultralytics 8.4.215의 detection fitness는 "
            "Precision/Recall/mAP@0.5가 아니라 mAP@0.5:0.95에 가중치 1.0을 두므로 이 지표로 best.pt를 선택한다. "
            "Epoch 9-16에서 Epoch 8의 0.40616을 넘지 못해 연속 8회 개선 없음 조건을 만족했고 Epoch 16 뒤 조기 종료되었다."
        )
    ]
    training_curve = PLOTS_ROOT / "training_curves.png"
    if training_curve.exists():
        story += [scaled_image(training_curve), p("그림 1. Epoch별 train/validation loss와 검증 지표", caption)]
    loss = training["final_losses"]
    story += [
        table(
            [
                ["최종 손실", "Box", "Class", "DFL"],
                ["Train", f"{loss['train_box']:.4f}", f"{loss['train_cls']:.4f}", f"{loss['train_dfl']:.4f}"],
                ["Validation", f"{loss['val_box']:.4f}", f"{loss['val_cls']:.4f}", f"{loss['val_dfl']:.4f}"],
            ],
            [40 * mm, 40 * mm, 40 * mm, 40 * mm],
        ),
        p(
            "따라서 30은 허용한 최대 Epoch이고 16은 실제 완료 횟수이다. 최적 모델은 Epoch 8의 validation "
            "mAP@0.5:0.95 기준 best.pt이며, Epoch 16의 last.pt와 구별된다. 이후 test 평가는 best.pt만 사용했고 "
            "test 분할은 모델 선택과 조기 종료 판단에 사용하지 않았다."
        ),
    ]

    overall = test["overall"]
    story += [p("6. 독립 Test 분할 정량 평가", h1)]
    story += [
        table(
            [
                ["Test 이미지", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
                [
                    test["test_images"],
                    f"{overall['precision']:.4f}",
                    f"{overall['recall']:.4f}",
                    f"{overall['map50']:.4f}",
                    f"{overall['map50_95']:.4f}",
                ],
            ],
            [30 * mm, 34 * mm, 34 * mm, 34 * mm, 38 * mm],
        ),
        Spacer(1, 3 * mm),
    ]
    class_rows = [["클래스", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]]
    for name in ("car", "pedestrian", "cyclist"):
        values = test["per_class"][name]
        class_rows.append(
            [
                name,
                f"{values['precision']:.4f}",
                f"{values['recall']:.4f}",
                f"{values['map50']:.4f}",
                f"{values['map50_95']:.4f}",
            ]
        )
    story += [table(class_rows, [38 * mm, 32 * mm, 32 * mm, 32 * mm, 36 * mm])]
    lowest = min(test["per_class"], key=lambda name: test["per_class"][name]["map50_95"])
    story += [
        p(
            f"클래스별로는 {escape(lowest)}의 mAP@0.5:0.95가 가장 낮았다. 이는 train에서 cyclist 이미지가 "
            f"{splits['train']['class_images']['cyclist']}장으로 car {splits['train']['class_images']['car']}장보다 적고, "
            "원거리의 작은 객체·부분 가림·자전거와 탑승자의 복합 형태가 많기 때문으로 해석한다. 이는 정량 결과와 "
            "데이터 분포에 근거한 해석이며, 별도 원인 분리 실험을 수행한 것은 아니다."
        )
    ]
    confusion = PLOTS_ROOT / "test_confusion_matrix_normalized.png"
    pr_curve = PLOTS_ROOT / "test_PR_curve.png"
    if confusion.exists():
        story += [scaled_image(confusion, max_height=95 * mm), p("그림 2. Test 정규화 혼동행렬", caption)]
    if pr_curve.exists():
        story += [scaled_image(pr_curve, max_height=95 * mm), p("그림 3. Test 클래스별 Precision–Recall 곡선", caption)]

    story += [PageBreak(), p("7. 추가 진단 실험", h1)]
    story += [
        p(
            "추가 모델 학습 없이 사전학습 모델과 기존 best.pt의 추론 결과만 분석했다. Test에서 confidence 0.001 이상 예측을 한 번 생성하고 "
            "동일한 NMS 결과를 0.05, 0.10, 0.25, 0.50, 0.75로 필터링했다. 클래스가 같은 예측과 정답을 IoU 0.5에서 "
            "confidence 내림차순으로 일대일 매칭했다. 따라서 아래 값은 고정 threshold의 micro 지표이며, 클래스별 PR 곡선을 "
            "적분하는 공식 mAP와 구별된다. 평가 후 가중치 SHA-256이 학습 완료 시점과 같아 모델이 변경되지 않았음을 확인했다."
        ),
        p(
            f"추가 평가 시간은 {analysis['duration_seconds']:.1f}초였고, 가중치 SHA-256은 "
            f"{analysis['model_sha256']}이다.",
            small,
        ),
        p("7.1 Fine-tuning 전후의 제한적 비교", h2),
    ]
    pre = pretrained_comparison["pretrained"]["per_class"]
    post = pretrained_comparison["fine_tuned"]["per_class"]
    baseline_rows = [["클래스 / 모델", "Precision", "Recall", "F1"]]
    for class_name in ("car", "pedestrian"):
        baseline_rows.extend(
            [
                [f"{class_name} / COCO", f"{pre[class_name]['precision']:.4f}", f"{pre[class_name]['recall']:.4f}", f"{pre[class_name]['f1']:.4f}"],
                [f"{class_name} / Fine-tuned", f"{post[class_name]['precision']:.4f}", f"{post[class_name]['recall']:.4f}", f"{post[class_name]['f1']:.4f}"],
            ]
        )
    story += [
        p(
            "COCO와 KITTI의 라벨 의미가 완전히 같지 않으므로 공식 mAP를 직접 비교하지 않았다. 동일한 1,076장 Test와 "
            "imgsz 640, conf 0.25, NMS IoU 0.7, 정답 매칭 IoU 0.5에서 COCO car(id 2)를 KITTI car로, "
            "COCO person(id 0)을 KITTI pedestrian으로 제한 매핑해 고정 임계값 P/R/F1만 비교했다. KITTI cyclist는 "
            "탑승자와 자전거를 하나의 상자로 표시하지만 COCO는 person/bicycle을 분리하므로 비교에서 제외했다."
        ),
        table(baseline_rows, [58 * mm, 36 * mm, 36 * mm, 36 * mm]),
    ]
    baseline_plot = PROJECT_ROOT / pretrained_comparison["artifact"]
    if baseline_plot.exists():
        story += [
            scaled_image(baseline_plot, max_height=70 * mm),
            p("그림 4. 동일 Test·고정 임계값에서 COCO 사전학습 모델과 fine-tuned 모델의 제한적 비교", caption),
        ]
    story += [
        p(
            f"car는 F1 {pre['car']['f1']:.4f}에서 {post['car']['f1']:.4f}로 향상되었다. pedestrian은 "
            f"Precision이 {pre['pedestrian']['precision']:.4f}에서 {post['pedestrian']['precision']:.4f}로 높아졌지만 "
            f"Recall은 {pre['pedestrian']['recall']:.4f}에서 {post['pedestrian']['recall']:.4f}로 낮아졌고, F1은 "
            f"{pre['pedestrian']['f1']:.4f}에서 {post['pedestrian']['f1']:.4f}로 상승했다. 이는 제한 매핑·단일 confidence의 "
            "진단 결과이지 COCO 대비 공식 일반화 성능 향상을 증명하지 않는다. 특히 COCO person과 KITTI pedestrian의 "
            "포함 범위, KITTI에서 제외한 Van/Truck 때문에 두 모델의 오검출 의미도 완전히 같지 않다."
        ),
        p("7.2 Confidence threshold에 따른 Precision/Recall", h2),
    ]
    threshold_rows = [["Conf.", "Micro P", "Micro R", "Micro F1", "car R", "ped. R", "cyclist R"]]
    for result in analysis["confidence_thresholds"]:
        threshold_rows.append(
            [
                f"{result['threshold']:.2f}",
                f"{result['micro']['precision']:.4f}",
                f"{result['micro']['recall']:.4f}",
                f"{result['micro']['f1']:.4f}",
                f"{result['per_class']['car']['recall']:.4f}",
                f"{result['per_class']['pedestrian']['recall']:.4f}",
                f"{result['per_class']['cyclist']['recall']:.4f}",
            ]
        )
    story += [table(threshold_rows, [18 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm, 25 * mm, 27 * mm])]
    threshold_plot = PROJECT_ROOT / analysis["artifacts"]["confidence_plot"]
    if threshold_plot.exists():
        story += [
            scaled_image(threshold_plot, max_height=72 * mm),
            p("그림 5. Confidence threshold 변화에 따른 Test micro 지표와 클래스별 Recall", caption),
        ]
    sampled_best = max(analysis["confidence_thresholds"], key=lambda result: result["micro"]["f1"])
    low_threshold = analysis["confidence_thresholds"][0]
    high_threshold = analysis["confidence_thresholds"][-1]
    story += [
        p(
            f"표본 threshold 중 conf {sampled_best['threshold']:.2f}에서 micro F1={sampled_best['micro']['f1']:.4f}로 가장 높았다. "
            f"conf {low_threshold['threshold']:.2f}에서는 Recall={low_threshold['micro']['recall']:.4f}까지 올라가지만 "
            f"Precision={low_threshold['micro']['precision']:.4f}로 낮아졌다. 반대로 conf {high_threshold['threshold']:.2f}에서는 "
            f"Precision={high_threshold['micro']['precision']:.4f}이지만 Recall={high_threshold['micro']['recall']:.4f}에 그쳤다. "
            "즉 threshold 변경은 모델 성능 자체의 개선이 아니라 FP와 FN의 운영상 교환이며, 안전 중심 탐지라면 낮은 threshold, "
            "오경보 비용이 큰 경우라면 높은 threshold를 선택해야 한다."
        )
    ]
    comparison = analysis["confidence_comparison"]
    comparison_image = PROJECT_ROOT / comparison["output"]
    if comparison_image.exists():
        panel_text = ", ".join(
            f"conf {item['threshold']:.2f}: TP {item['tp']}, FP {item['fp']}, FN {item['fn']}"
            for item in comparison["panels"]
        )
        story += [
            scaled_image(comparison_image, max_height=126 * mm),
            p(
                f"그림 6. 동일 Test 이미지 {comparison['image']}의 threshold 비교 - {panel_text}. "
                "빨간 상자는 놓친 정답이다.",
                caption,
            ),
        ]

    story += [PageBreak(), p("7.3 객체 크기별 Recall", h2)]
    story += [
        p(
            "원본 KITTI 영상 좌표에서 bounding box 면적을 small(<32^2), medium(32^2 이상 96^2 미만), "
            "large(96^2 이상) 픽셀로 구분했다. 이는 COCO식 진단 구간을 빌린 것이며 KITTI 공식 평가 지표가 아니다. "
            "conf 0.25, IoU 0.5에서 각 크기의 ground truth가 검출된 비율을 계산했다. FP에 정답 크기를 부여하는 것은 "
            "모호하므로 크기별 Precision이나 AP를 임의로 만들지 않았다."
        )
    ]
    size_rows = [["클래스", "Small n / R", "Medium n / R", "Large n / R"]]
    for class_name in ("car", "pedestrian", "cyclist"):
        values = analysis["size_recall"]["per_class"][class_name]
        size_rows.append(
            [
                class_name,
                f"{values['small']['ground_truth']} / {values['small']['recall']:.4f}",
                f"{values['medium']['ground_truth']} / {values['medium']['recall']:.4f}",
                f"{values['large']['ground_truth']} / {values['large']['recall']:.4f}",
            ]
        )
    story += [table(size_rows, [36 * mm, 44 * mm, 44 * mm, 44 * mm])]
    size_plot = PROJECT_ROOT / analysis["artifacts"]["size_plot"]
    if size_plot.exists():
        story += [
            scaled_image(size_plot, max_height=83 * mm),
            p("그림 7. Test ground-truth 객체 크기별 Recall(conf 0.25, IoU 0.5)", caption),
        ]
    pedestrian_sizes = analysis["size_recall"]["per_class"]["pedestrian"]
    cyclist_sizes = analysis["size_recall"]["per_class"]["cyclist"]
    story += [
        p(
            f"pedestrian Recall은 small {pedestrian_sizes['small']['recall']:.4f}, medium "
            f"{pedestrian_sizes['medium']['recall']:.4f}, large {pedestrian_sizes['large']['recall']:.4f}였고, cyclist는 "
            f"각각 {cyclist_sizes['small']['recall']:.4f}, {cyclist_sizes['medium']['recall']:.4f}, "
            f"{cyclist_sizes['large']['recall']:.4f}였다. 두 클래스 모두 작은 객체에서 약 2%만 검출되었고 큰 객체에서는 "
            "약 73%까지 상승했다. 이 결과는 원거리의 작은 객체가 낮은 Recall과 강하게 연관됨을 정량적으로 지지한다. "
            "다만 크기 외의 가림, 자세, 장면 복잡도를 통제하지 않았으므로 크기만이 유일한 원인이라고 단정하지 않는다."
        )
    ]

    story += [PageBreak(), p("7.4 Validation-Test 성능 차이", h2)]
    validation = analysis["validation_recheck"]
    split_rows = [["클래스", "Val R", "Test R", "Val mAP50-95", "Test mAP50-95"]]
    split_rows.append(
        [
            "전체 평균",
            f"{validation['overall']['recall']:.4f}",
            f"{test['overall']['recall']:.4f}",
            f"{validation['overall']['map50_95']:.4f}",
            f"{test['overall']['map50_95']:.4f}",
        ]
    )
    for class_name in ("car", "pedestrian", "cyclist"):
        split_rows.append(
            [
                class_name,
                f"{validation['per_class'][class_name]['recall']:.4f}",
                f"{test['per_class'][class_name]['recall']:.4f}",
                f"{validation['per_class'][class_name]['map50_95']:.4f}",
                f"{test['per_class'][class_name]['map50_95']:.4f}",
            ]
        )
    story += [table(split_rows, [35 * mm, 30 * mm, 30 * mm, 37 * mm, 37 * mm])]
    val_dist = analysis["split_distribution"]["val"]
    test_dist = analysis["split_distribution"]["test"]
    story += [
        table(
            [
                ["분포", "Validation", "Test"],
                ["이미지 / 개체", f"{val_dist['images']} / {val_dist['instances']}", f"{test_dist['images']} / {test_dist['instances']}"],
                ["이미지당 개체 평균", f"{val_dist['instances_per_image']:.3f}", f"{test_dist['instances_per_image']:.3f}"],
                ["이미지당 개체 중앙값 / 90백분위", f"{val_dist['objects_per_image_median']:.1f} / {val_dist['objects_per_image_p90']:.1f}", f"{test_dist['objects_per_image_median']:.1f} / {test_dist['objects_per_image_p90']:.1f}"],
                ["pedestrian small / large 비율", f"{val_dist['per_class_size_instances']['pedestrian']['small'] / val_dist['class_instances']['pedestrian']:.1%} / {val_dist['per_class_size_instances']['pedestrian']['large'] / val_dist['class_instances']['pedestrian']:.1%}", f"{test_dist['per_class_size_instances']['pedestrian']['small'] / test_dist['class_instances']['pedestrian']:.1%} / {test_dist['per_class_size_instances']['pedestrian']['large'] / test_dist['class_instances']['pedestrian']:.1%}"],
                ["pedestrian box 면적 중앙값", f"{val_dist['per_class_median_area_pixels']['pedestrian']:.1f} px²", f"{test_dist['per_class_median_area_pixels']['pedestrian']:.1f} px²"],
            ],
            [67 * mm, 50 * mm, 50 * mm],
        ),
        p(
            f"재평가된 Validation mAP@0.5:0.95는 {validation['overall']['map50_95']:.4f}, Test는 "
            f"{test['overall']['map50_95']:.4f}로 차이는 {analysis['validation_minus_test']['overall']['map50_95']:.4f}였다. "
            "클래스 비중과 전체 small/medium/large 비중은 두 분할이 대체로 비슷했으므로 단순 클래스 구성만으로 차이를 "
            "충분히 설명하기 어렵다. 반면 Test는 이미지당 개체가 평균 4.948개로 Validation 2.458개의 약 두 배이고, "
            "중앙값과 90백분위도 각각 4/10 대 2/5로 더 조밀했다. 이는 가림과 중첩 가능성이 더 큰 분포임을 시사하지만, "
            "직접 occlusion 라벨을 분석한 것은 아니므로 인과관계로 단정하지 않는다."
        ),
        p(
            f"특히 Validation pedestrian은 small 0.0%, large 76.6%, box 면적 중앙값 "
            f"{val_dist['per_class_median_area_pixels']['pedestrian']:.1f}px²였지만 Test는 small 26.8%, large 14.5%, "
            f"중앙값 {test_dist['per_class_median_area_pixels']['pedestrian']:.1f}px²였다. 같은 모델의 pedestrian Recall이 "
            f"{validation['per_class']['pedestrian']['recall']:.4f}에서 {test['per_class']['pedestrian']['recall']:.4f}로, "
            f"mAP@0.5:0.95가 {validation['per_class']['pedestrian']['map50_95']:.4f}에서 "
            f"{test['per_class']['pedestrian']['map50_95']:.4f}로 낮아진 현상과 방향이 일치한다. 이는 보행자 크기 분포 이동을 "
            "주요 설명 후보로 강하게 지지하지만 통제 실험은 아니다."
        ),
        p(
            "cyclist는 오히려 Validation의 small 비율이 Test보다 높았는데도 Test 성능이 낮았다. 따라서 cyclist의 분할 간 "
            "차이는 객체 크기만으로 설명할 수 없다. 학습 표본 부족, 조밀한 Test 장면, 가림이나 자세 차이는 가능한 설명이지만 "
            "이번 진단으로 각각의 효과를 분리하지 못했으므로 가설로만 남긴다."
        ),
    ]

    story += [PageBreak(), p("8. 성공·실패 사례 분석", h1)]
    case_analysis = test["case_analysis"]
    aggregate = case_analysis["aggregate"]
    story += [
        p(
            f"사례 선별용 고정 임계값(confidence 0.25, IoU 0.5)에서 전체 TP={aggregate['tp']}, "
            f"FP={aggregate['fp']}, FN={aggregate['fn']}이었다. 이 값은 PR 곡선을 적분한 공식 mAP와 목적이 다르므로 "
            "공식 성능표를 대체하지 않는다. 성공 사례는 FP/FN이 없고 TP가 많은 순서, 실패 사례는 FN·FP가 큰 순서로 "
            "자동 선택했다. 실패 그림의 빨간 상자는 놓친 ground truth이다."
        )
    ]
    for case in case_analysis["selected_cases"]:
        path = PROJECT_ROOT / case["output"]
        if path.exists():
            label = "성공" if case["category"] == "success" else "실패"
            story += [
                scaled_image(path, max_height=78 * mm),
                p(
                    f"{label} 사례 {case['image']} — TP {case['tp']}, FP {case['fp']}, FN {case['fn']}",
                    caption,
                ),
            ]
    story += [
        p(
            "대표 실패는 작은 원거리 객체, 심한 가림, 한 객체의 중복 검출 또는 유사 외형에서 발생했다. 개선하려면 "
            "희소 클래스의 장면을 보강하고, 작은 객체에 유리한 입력 해상도·더 큰 모델·class-balanced sampling을 비교하며, "
            "크기별 Recall 진단을 class-balanced 학습 및 해상도 통제 실험으로 확장할 필요가 있다."
        )
    ]

    story += [PageBreak(), p("9. 5초 객체 추적 결과", h1)]
    source = tracking["source"]
    verification = tracking["verification"]
    per_frame = tracking["per_class_frame_count"]
    unique_ids = tracking["per_class_unique_track_ids"]
    continuity = tracking_continuity["per_class"]
    duration = float(tracking["ffprobe"]["format"]["duration"])
    story += [
        p(
            f"KITTI 탐지 이미지와 별개인 Pexels 연속 영상 ‘{escape(str(source['title']))}’를 사용했다. "
            f"제작자는 {escape(str(source['creator']))}이며, 원본은 {source['source_duration_seconds']:.3f}초, "
            "1920×1080이다. Pexels License는 무료 사용과 수정을 허용하고 출처 표기를 요구하지 않지만, "
            "재현성과 예의를 위해 원본 페이지와 제작자를 기록했다."
        ),
        p(
            "원본 첫 5초에는 자동차, 보행자, 자전거 이용자가 함께 등장한다. best.pt로 매 프레임을 탐지하고 "
            "Ultralytics ByteTrack에 persist=True로 전달해 track ID를 유지했다. 결과 영상에는 class, confidence, "
            "bounding box와 ID가 표시된다."
        ),
        table(
            [
                ["검증 항목", "결과"],
                ["출력", tracking["output"]],
                ["길이 / 프레임", f"{duration:.3f}초 / {tracking['decoded_frames']} 프레임"],
                ["codec / pixel format", f"{tracking['ffprobe']['streams'][0]['codec_name']} / {tracking['ffprobe']['streams'][0]['pix_fmt']}"],
                ["탐지 / ID 포함 탐지", f"{tracking['detections']} / {tracking['detections_with_track_id']}"],
                ["클래스별 검출 프레임", ", ".join(f"{name} {per_frame[name]}" for name in ("car", "pedestrian", "cyclist"))],
                ["클래스별 고유 ID 수", ", ".join(f"{name} {len(unique_ids[name])}" for name in ("car", "pedestrian", "cyclist"))],
                ["인접 프레임 ID 쌍", ", ".join(f"{name} {continuity[name]['consecutive_frame_pairs']}/{continuity[name]['all_observed_frame_pairs']}" for name in ("car", "pedestrian", "cyclist"))],
                ["검증", f"duration={verification['duration_valid']}, decode={verification['decode_valid']}, IDs={verification['tracking_ids_present']}"],
            ],
            [47 * mm, 121 * mm],
        ),
    ]
    tracking_frame = PROJECT_ROOT / tracking["representative_frame"]
    if tracking_frame.exists():
        story += [scaled_image(tracking_frame, max_height=95 * mm), p("그림 8. 학습 모델과 ByteTrack의 5초 영상 중 대표 프레임", caption)]
    story += [
        p(
            "CSV에서 동일 ID가 인접 프레임에 반복되는 쌍은 car 485/498, pedestrian 339/366, cyclist 309/343이었다. "
            "따라서 결과는 프레임별 독립 탐지만이 아니라 실제 ID 기반 추적을 포함한다. 그러나 모든 1,244개 검출에 ID가 "
            "있다는 사실은 ID가 항상 올바르게 유지되었다는 뜻이 아니다. 이 Pexels 영상에는 ground-truth track ID가 없으므로 "
            "IDF1, HOTA, identity switch 수를 측정하지 않았으며 ‘안정적인 추적 성능’을 정량적으로 주장하지 않는다."
        )
    ]

    story += [p("10. 수행 중 문제와 해결", h1)]
    story += bullets(
        [
            "원본 분할 누수: 가까운 장면이 train/val 양쪽에 존재함을 시각적으로 확인하고, perceptual hash와 픽셀 유사도로 장면 그룹을 만든 뒤 그룹 단위로 다시 분할했다.",
            "MPS 가용성: 단순 API 확인에 그치지 않고 convolution backward와 1 Epoch smoke test를 실행했다. 샌드박스 밖 실제 장치에서 MPS가 사용됨을 기록했다.",
            "학습 시간: 전체 1 Epoch 벤치마크를 먼저 수행해 약 4분을 측정하고 30 Epoch·batch 8을 선택했다.",
            "희소 클래스: cyclist의 출현 수가 car보다 작아 클래스별 지표와 실패 사례를 따로 보고했다. 존재하지 않는 균형 데이터를 주장하지 않았다.",
            "추적 입력: 탐지용 KITTI 정지 프레임을 이어 붙이지 않고 별도의 실제 연속 영상과 이용 조건을 확보했다.",
            "아티팩트 일치: 보고서 생성기가 JSON·CSV·그림을 직접 읽게 하여 수치 전사 오류를 줄였다.",
            "분할 성능 차이: 재학습 없이 confidence·크기·분할 분포를 측정해 확인된 연관성과 미검증 가설을 구분했다.",
            "전이학습 설명: 집계형 Transferred 문구에만 의존하지 않고 같은 버전 로더와 출력 텐서를 감사해 클래스별 재사용 범위를 확인했다.",
        ]
    )
    story += [p("11. 한계와 개선 방향", h1)]
    story += bullets(
        [
            "KITTI 변환본에 공식 촬영 시퀀스 ID가 없어 유사도 기반 그룹화로도 모든 동일 주행 장면을 완전히 분리했다고 보장할 수 없다.",
            "MPS 일부 연산의 비결정론 때문에 동일 설정 재실행 시 마지막 자리 수가 달라질 수 있다.",
            "YOLOv8n 단일 모델·단일 seed만 사용했다. YOLOv8s, 입력 960, class-balanced sampling의 통제 실험을 추가하면 선택 근거가 강해진다.",
            "객체 크기 분석은 고정 conf 0.25의 Recall 진단이다. 크기별 AP나 가림 수준별 성능을 측정한 것으로 해석하면 안 된다.",
            "COCO 사전학습 대비 비교는 car/person의 제한 매핑과 conf 0.25에서만 유효하며 공식 mAP 비교가 아니다. cyclist는 라벨 의미가 달라 제외했다.",
            "추적 영상은 5초의 한 장면이므로 다양한 날씨·가림·카메라 이동에 대한 tracker 안정성을 대표하지 않는다.",
            "Pexels 영상의 도메인은 KITTI와 달라 실제 도메인 이동 성능을 보여 주지만, 해당 영상의 정답 라벨이 없어 추적 정확도를 수치화하지 않았다.",
        ]
    )

    story += [p("12. 결론", h1)]
    story += [
        p(
            "KITTI의 8개 원본 클래스 구조를 검사하고 정확히 3개로 재매핑했으며, 장면 그룹 단위의 train/val/test를 구성했다. "
            f"Apple M1 Pro MPS에서 실제 fine-tuning을 수행해 best.pt를 얻었고, 독립 test에서 "
            f"mAP@0.5={overall['map50']:.4f}, mAP@0.5:0.95={overall['map50_95']:.4f}를 측정했다. "
            f"또한 별도 연속 영상에서 {duration:.3f}초 ByteTrack 영상을 생성하고 ffmpeg 전체 디코딩과 track ID 존재를 확인했다."
        ),
        p(
            "이 과제를 통해 높은 전체 지표만 제시하는 것보다 클래스 불균형, 데이터 누수, 평가 분할의 독립성, "
            "실제 장치 제약과 실패 사례를 함께 기록해야 결과를 신뢰할 수 있음을 경험했다. 추가 진단에서 threshold는 FP/FN의 "
            "교환을 만들 뿐 성능을 일방적으로 개선하지 않았고, small pedestrian/cyclist Recall이 약 2%임을 확인했다. "
            "제한적 학습 전후 비교에서는 car와 pedestrian F1이 상승했으나 pedestrian Recall은 낮아져 모든 지표가 일괄 개선된 것은 아니었다. 또한 "
            "Validation-Test 차이에는 보행자 크기 분포 이동과 Test 장면 밀도가 연관되어 있었지만 cyclist 차이는 크기만으로 "
            "설명되지 않았다. 다음 단계에서는 cyclist 보강, 입력 해상도와 모델 크기를 분리한 통제 실험을 수행하겠다."
        ),
    ]

    story += [p("13. GitHub 및 참고자료", h1)]
    story += [
        table(
            [
                ["항목", "위치 / URL"],
                ["GitHub 저장소", args.github_url],
                ["최종 모델", training["weight"]],
                ["5초 영상", tracking["output"]],
                ["테스트 지표", "results/metrics/test_metrics.json"],
                ["추가 진단", "results/metrics/additional_analysis.json"],
                ["학습 전후 비교", "results/metrics/pretrained_comparison.json"],
                ["전이 감사", "results/metrics/transfer_audit.json"],
                ["추적 연속성", "results/metrics/tracking_continuity.json"],
                ["보고서", str(output.relative_to(PROJECT_ROOT))],
            ],
            [43 * mm, 125 * mm],
        ),
        Spacer(1, 5 * mm),
        p("강의자료", h2),
        *compact_bullets(
            [
                "이건명, computer-vision-1.pdf — 컴퓨터비전 개론.",
                "이건명, computer-vision-2.pdf — 특징 추출과 전통적 컴퓨터비전.",
                "이건명, computer-vision-3.pdf — 객체 탐지, bounding box, IoU, NMS.",
                "이건명, computer-vision-4.pdf — YOLO, 전이학습, 성능 평가와 과제(특히 49–51쪽).",
            ]
        ),
        p("외부 자료 및 데이터", h2),
        *compact_bullets(
            [
                "KITTI Vision Benchmark Suite: https://www.cvlibs.net/datasets/kitti/ (academic use only 안내).",
                "Ultralytics KITTI YAML/변환본: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml",
                "Ultralytics YOLOv8: https://docs.ultralytics.com/models/yolov8/",
                "Ultralytics Train / Val / Track: https://docs.ultralytics.com/modes/train/, /val/, /track/",
                f"Pexels 원본 영상: {source['page_url']} — {source['creator']}, Pexels License.",
                f"Pexels License: {source['license_url']}",
                "PyTorch MPS backend note: https://docs.pytorch.org/docs/stable/notes/mps.html",
            ]
        ),
        p(
            f"최종 가중치 SHA-256: {training['weight_sha256']}<br/>"
            f"추적 영상 SHA-256: {tracking['output_sha256']}",
            small,
        ),
    ]

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="머신러닝 과제 2 - YOLOv8 Fine-tuning을 이용한 3개 부류 객체 탐지 및 추적",
        author=args.author,
        subject="충북대학교 머신러닝 과제 2",
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(output)


if __name__ == "__main__":
    main()
