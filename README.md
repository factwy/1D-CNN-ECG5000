# ECG5000 1D-CNN Classification

ECG5000 시계열 데이터셋을 기반으로 다양한 1D-CNN 모델을 학습하고, ONNX 및 TFLite로 변환·추론하는 프로젝트입니다.

---

## 목차

1. [데이터셋](#데이터셋)
2. [프로젝트 구조](#프로젝트-구조)
3. [Docker 환경 설정](#docker-환경-설정)
4. [모델 설명](#모델-설명)
5. [학습](#학습)
6. [추론](#추론)
7. [모델 요약](#모델-요약)
8. [실험 결과](#실험-결과)
9. [ONNX 변환 및 추론](#onnx-변환-및-추론)
10. [TFLite 변환 및 추론](#tflite-변환-및-추론)

---

## 데이터셋

**ECG5000** — UCR Time Series Archive 제공 심전도(ECG) 분류 데이터셋

| 항목 | 값 |
|---|---|
| Train 샘플 수 | 500 |
| Test 샘플 수 | 4,500 |
| 시계열 길이 | 140 |
| 클래스 수 | 5 |
| 클래스 분포 (Train) | C0: 292 / C1: 177 / C2: 10 / C3: 19 / C4: 2 |

데이터 파일은 아래 경로에 위치해야 합니다.

```
data/ECG5000/
├── ECG5000_TRAIN.txt
└── ECG5000_TEST.txt
```

---

## 프로젝트 구조

```
1D-CNN-ECG5000/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
│
├── train.py              # 학습 스크립트
├── inference.py          # PyTorch 추론
├── summary.py            # 모델 아키텍처 요약
├── train.sh              # 학습 실행 쉘 스크립트
│
├── models/
│   ├── __init__.py       # MODEL_REGISTRY / build_model()
│   ├── basecnn.py        # Base CNN
│   ├── vgg.py            # VGG
│   ├── inception.py      # InceptionTime
│   └── tcn.py            # TCN
│
├── onnx/
│   ├── export_onnx.py    # PyTorch → ONNX 변환
│   └── inference_onnx.py # ONNX 추론
│
├── tflite/
│   ├── export_tflite.py  # ONNX → TFLite 변환
│   └── inference_tflite.py # TFLite 추론
│
├── data/
│   └── ECG5000/          # 데이터셋 (볼륨 마운트)
└── outputs/              # 체크포인트, 플롯 (볼륨 마운트)
```

---

## Docker 환경 설정

### 요구사항

- Docker Engine 25+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 이미지 빌드 및 컨테이너 실행

```bash
# 빌드 + 백그라운드 실행
docker compose up -d --build

# 학습 컨테이너 접속
docker compose exec train bash

# 추론 컨테이너 접속
docker compose exec inference bash
```

### 볼륨 구성

| 호스트 경로 | 컨테이너 경로 | 용도 |
|---|---|---|
| `.` | `/workspace` | 소스 코드 (수정 즉시 반영) |
| `./data/ECG5000` | `/data/ECG5000` | 데이터셋 |
| `./outputs` | `/workspace/outputs` | 체크포인트, 플롯 |

> 소스 코드가 볼륨으로 마운트되어 있어 재빌드 없이 파일 수정이 즉시 반영됩니다.

---

## 모델 설명

모든 모델은 입력 shape `(B, 1, 140)` → 출력 `(B, num_classes)` 구조입니다.

### Base CNN (`base-cnn`)

심플한 3-블록 베이스라인 모델.

```
Conv(64, k=7) → BN → ReLU → Dropout → MaxPool
Conv(128, k=5) → BN → ReLU → Dropout → MaxPool
Conv(256, k=3) → BN → ReLU → Dropout
GAP → FC(128) → ReLU → Dropout → FC(num_classes)
```

### VGG (`vgg`)

VGG11 구조를 1D 시계열에 맞게 변형.

```
Block1 : Conv(64,  k=3) → BN → ReLU → Dropout → MaxPool
Block2 : Conv(128, k=3) → BN → ReLU → Dropout → MaxPool
Block3 : Conv(256, k=3) × 2 → BN → ReLU → Dropout → MaxPool
Block4 : Conv(512, k=3) × 2 → BN → ReLU → Dropout → MaxPool
Block5 : Conv(512, k=3) × 2 → BN → ReLU → Dropout → MaxPool
GAP → FC(512) → ReLU → FC(512) → ReLU → Dropout → FC(num_classes)
```

### InceptionTime (`inception`)

[InceptionTime](https://arxiv.org/abs/1909.04939) 기반. 다양한 커널 크기(k=9, 19, 39)의 병렬 합성곱으로 멀티스케일 특징 추출.

```
InceptionModule: Bottleneck(32) → [Branch k=9 / k=19 / k=39 / MaxPool] → Concat → BN → ReLU

Block1 → Block2 → Block3 → Residual(input) → ReLU
Block4 → Block5 → Block6 → Residual → ReLU
GAP → FC(64) → ReLU → Dropout → FC(num_classes)
```

### TCN (`tcn`)

[Temporal Convolutional Network](https://arxiv.org/abs/1803.01271) 기반. WeightNorm + Dilated Conv + Residual Connection.

```
TemporalModule(d=1, C=64)  → WeightNorm Dilated Conv × 2 + Residual
TemporalModule(d=2, C=64)
TemporalModule(d=4, C=64)
TemporalModule(d=8, C=64)
GAP → FC(32) → ReLU → Dropout → FC(num_classes)
```

---

## 학습

### 기본 실행

```bash
# 쉘 스크립트 (기본 하이퍼파라미터)
bash train.sh <model>

# 예시
bash train.sh base-cnn
bash train.sh vgg
bash train.sh inception
bash train.sh tcn
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | `vgg` | `base-cnn` / `vgg` / `inception` / `tcn` |
| `--data_dir` | `/data/ECG5000` | 데이터 경로 |
| `--output_dir` | `./outputs` | 결과 저장 경로 |
| `--epochs` | `50` | 학습 에포크 수 |
| `--lr` | `1e-3` | 학습률 (CosineAnnealing 스케줄러) |
| `--batch_size` | `64` | 배치 크기 |
| `--dropout` | `0.3` | Dropout 비율 |
| `--amp` | `True` | Automatic Mixed Precision |
| `--compile` | `False` | torch.compile() |

학습이 완료되면 `outputs/<model>/` 에 아래 파일이 저장됩니다.

```
outputs/<model>/
├── <model>_best.pth          # Best validation accuracy 체크포인트
├── <model>_training_curves.png
└── <model>_confusion_matrix.png
```

---

## 추론

### PyTorch 추론

```bash
python inference.py \
  --model      vgg \
  --model_path outputs/vgg/vgg_best.pth
```

---

## 모델 요약

```bash
# torchinfo 기반 레이어별 shape / 파라미터 수 출력
python summary.py --model vgg
python summary.py --model inception
```

---

## 실험 결과


### 포맷별 성능 요약

| 모델 | PyTorch (ONNX) | TFLite FP32 | TFLite INT8 | FP32 크기 | INT8 크기 | 압축률 |
|---|:---:|:---:|:---:|---:|---:|:---:|
| VGG | 92.84% | 92.84% | 92.82% | 13.76 MB | 3.50 MB | 1/4 |
| Inception | 94.24% | 94.24% | 94.22% | 1.80 MB | 0.49 MB | 1/4 |
| TCN | 91.73% | 91.73% | 91.73% | 0.35 MB | 0.11 MB | 1/3 |


---

### 상세 결과

#### VGG

<details>
<summary>PyTorch / ONNX / TFLite FP32 — Accuracy 0.9284</summary>

```
              precision    recall  f1-score   support

     Class 0       0.98      0.99      0.98      2627
     Class 1       0.86      0.99      0.92      1590
     Class 2       0.00      0.00      0.00        86
     Class 3       0.00      0.00      0.00       175
     Class 4       0.00      0.00      0.00        22

    accuracy                           0.93      4500
   macro avg       0.37      0.40      0.38      4500
weighted avg       0.87      0.93      0.90      4500
```

</details>

<details>
<summary>TFLite INT8 (3.50 MB) — Accuracy 0.9282</summary>

```
              precision    recall  f1-score   support

     Class 0       0.98      0.99      0.98      2627
     Class 1       0.86      0.99      0.92      1590
     Class 2       0.00      0.00      0.00        86
     Class 3       0.00      0.00      0.00       175
     Class 4       0.00      0.00      0.00        22

    accuracy                           0.93      4500
   macro avg       0.37      0.40      0.38      4500
weighted avg       0.87      0.93      0.90      4500
```

</details>

---

#### Inception

<details>
<summary>PyTorch / ONNX / TFLite FP32 — Accuracy 0.9424</summary>

```
              precision    recall  f1-score   support

     Class 0       0.98      1.00      0.99      2627
     Class 1       0.92      0.97      0.94      1590
     Class 2       0.71      0.23      0.35        86
     Class 3       0.54      0.41      0.46       175
     Class 4       0.00      0.00      0.00        22

    accuracy                           0.94      4500
   macro avg       0.63      0.52      0.55      4500
weighted avg       0.93      0.94      0.93      4500
```

</details>

<details>
<summary>TFLite INT8 (0.49 MB) — Accuracy 0.9422</summary>

```
              precision    recall  f1-score   support

     Class 0       0.98      1.00      0.99      2627
     Class 1       0.92      0.96      0.94      1590
     Class 2       0.71      0.23      0.35        86
     Class 3       0.53      0.41      0.46       175
     Class 4       0.00      0.00      0.00        22

    accuracy                           0.94      4500
   macro avg       0.63      0.52      0.55      4500
weighted avg       0.93      0.94      0.93      4500
```

</details>

---

#### TCN

<details>
<summary>PyTorch / ONNX / TFLite FP32 — Accuracy 0.9173</summary>

```
              precision    recall  f1-score   support

     Class 0       0.93      1.00      0.96      2627
     Class 1       0.89      0.95      0.92      1590
     Class 2       0.00      0.00      0.00        86
     Class 3       0.00      0.00      0.00       175
     Class 4       0.00      0.00      0.00        22

    accuracy                           0.92      4500
   macro avg       0.36      0.39      0.38      4500
weighted avg       0.86      0.92      0.89      4500
```

</details>

<details>
<summary>TFLite INT8 (0.11 MB) — Accuracy 0.9173</summary>

```
              precision    recall  f1-score   support

     Class 0       0.93      1.00      0.96      2627
     Class 1       0.89      0.95      0.92      1590
     Class 2       0.00      0.00      0.00        86
     Class 3       0.00      0.00      0.00       175
     Class 4       0.00      0.00      0.00        22

    accuracy                           0.92      4500
   macro avg       0.36      0.39      0.38      4500
weighted avg       0.86      0.92      0.89      4500
```

</details>

---

## ONNX 변환 및 추론

### 변환

```bash
python onnx/export_onnx.py \
  --model      vgg \
  --model_path outputs/vgg/vgg_best.pth
```

변환 후 자동으로 3단계 검증을 수행합니다.

```
1. torch.onnx.export()     → ONNX 파일 생성
2. onnx.checker             → 그래프 구조 유효성 검사
3. OnnxRuntime 추론         → PyTorch 출력과 수치 비교 (atol=1e-4)
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--opset` | `17` | ONNX opset 버전 |
| `--output_dir` | `/workspace/outputs/onnx` | 저장 경로 |

### ONNX 추론

```bash
python onnx/inference_onnx.py \
  --onnx_path outputs/onnx/vgg.onnx

# CPU 추론
python onnx/inference_onnx.py \
  --onnx_path outputs/onnx/vgg.onnx \
  --device cpu
```

---

## TFLite 변환 및 추론

### 변환 파이프라인

```
PyTorch (.pth)
    ↓  export_onnx.py
ONNX (.onnx)
    ↓  export_tflite.py  [onnx2tf → TFLiteConverter]
TFLite (.tflite)
```

### 변환

```bash
# FP32 (기본)
python tflite/export_tflite.py \
  --onnx_path outputs/onnx/vgg.onnx

# FP16 양자화 (크기 약 1/2)
python tflite/export_tflite.py \
  --onnx_path outputs/onnx/vgg.onnx \
  --quantize fp16

# INT8 PTQ - Dynamic Range Quantization (크기 약 1/4)
python tflite/export_tflite.py \
  --onnx_path outputs/onnx/vgg.onnx \
  --quantize int8
```

### 양자화 옵션 비교

| 옵션 | 가중치 | 활성화 | 보정 데이터 | 파일 크기 |
|---|---|---|---|---|
| `none` | FP32 | FP32 | 불필요 | 100% |
| `fp16` | FP16 | FP16 | 불필요 | ~50% |
| `int8` | INT8 | FP32 (동적) | 불필요 | ~25% |

### TFLite 추론

```bash
python tflite/inference_tflite.py \
  --tflite_path outputs/tflite/vgg.tflite

python tflite/inference_tflite.py \
  --tflite_path outputs/tflite/vgg_int8.tflite
```

---

## Requirements

주요 의존성 패키지 (`requirements.txt` 참고)

| 패키지 | 용도 |
|---|---|
| `torch` (Dockerfile 설치) | 학습 / PyTorch 추론 |
| `onnx` | ONNX 그래프 검증 |
| `onnxruntime-gpu` | ONNX 추론 (GPU) |
| `onnx2tf` | ONNX → TFLite 변환 |
| `tensorflow-cpu` | TFLite 변환 및 추론 |
| `torchinfo` | 모델 아키텍처 요약 |
| `scikit-learn` | 평가 지표 |
