# Fashion-MNIST CNN Classification

## 실습 설명

혼자 공부하는 머신러닝+딥러닝 책의 PyTorch 실습 코드를 따라 구현하며 Fashion-MNIST 데이터셋을 이용한 CNN 분류 과정을 학습했습니다.

먼저 책의 기본 실습 모델을 바탕으로 데이터 전처리, CNN 모델 구성, 학습 루프, 검증 손실 확인, 모델 저장과 로드, 테스트 정확도 평가 과정을 실습했습니다.

이후 기본 CNN 모델을 그대로 사용하는 데서 그치지 않고, 합성곱층을 추가하고 Batch Normalization을 적용한 개선 CNN 모델을 직접 구성하여 성능 변화를 비교했습니다.

이를 통해 기본 CNN 구조를 이해하고, 모델 구조 변경이 검증 손실과 테스트 정확도에 어떤 영향을 주는지 확인했습니다.

## 사용 데이터

Fashion-MNIST 데이터셋을 사용했습니다.

Fashion-MNIST는 28×28 크기의 흑백 의류 이미지 데이터셋이며, 총 10개의 클래스로 구성되어 있습니다.

- 훈련 데이터: 60,000장
- 테스트 데이터: 10,000장
- 클래스 수: 10개

## 사용 기술

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

## 데이터 전처리

PyTorch의 `Conv2d` 층은 입력 데이터를 `(배치 크기, 채널 수, 높이, 너비)` 형태로 받기 때문에 이미지 데이터를 다음과 같이 변환했습니다.

```python
train_scaled = train_input.reshape(-1, 1, 28, 28) / 255.0
```

또한 픽셀값을 0-255 범위에서 0-1 범위로 정규화했습니다.

훈련 데이터 중 20%를 검증 세트로 분리하여 모델 성능 확인에 사용했습니다.

```python
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled,
    train_target,
    test_size=0.2,
    random_state=42,
    stratify=train_target
)
```
## 기본 CNN 모델

기본 CNN 모델은 두 개의 합성곱 블록과 하나의 밀집층으로 구성했습니다.

```text
Input
→ Conv2d(1 → 32)
→ ReLU
→ MaxPool

→ Conv2d(32 → 64)
→ ReLU
→ MaxPool

→ Flatten
→ Linear(3136 → 100)
→ ReLU
→ Dropout
→ Linear(100 → 10)
```

## 개선 CNN 모델

개선 CNN 모델에서는 기존 모델보다 합성곱층을 더 깊게 구성하고, 각 합성곱층 뒤에 Batch Normalization을 추가했습니다.
```text
Input
→ Conv2d(1 → 32)
→ BatchNorm2d(32)
→ ReLU

→ Conv2d(32 → 32)
→ BatchNorm2d(32)
→ ReLU
→ MaxPool

→ Conv2d(32 → 64)
→ BatchNorm2d(64)
→ ReLU

→ Conv2d(64 → 64)
→ BatchNorm2d(64)
→ ReLU
→ MaxPool

→ Flatten
→ Linear(3136 → 128)
→ ReLU
→ Dropout
→ Linear(128 → 10)
```

개선 모델에서는 합성곱층을 추가하여 더 복잡한 이미지 특징을 학습할 수 있도록 했습니다.

또한 Batch Normalization을 적용해 각 층의 출력 분포를 안정화하고, 학습이 더 안정적으로 진행되도록 했습니다.

## 학습 방법

모델 학습에는 CrossEntropyLoss와 Adam optimizer를 사용했습니다.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```

훈련 과정에서는 매 epoch마다 훈련 손실과 검증 손실을 계산했습니다.
검증 손실이 가장 낮은 모델을 저장하고, 검증 손실이 일정 횟수 이상 개선되지 않으면 조기 종료하도록 구현했습니다.

```python
torch.save(model.state_dict(), 'best_cnn_model.pt')
```

개선 모델은 별도의 파일로 저장했습니다.
```python
torch.save(improved_model.state_dict(), 'best_improved_cnn_model.pt')
```

## 실험 결과

기본 CNN 모델과 개선 CNN 모델의 검증 손실을 비교했습니다.

## 결과 비교

| 모델 | 최고 검증 손실 | 조기 종료 에포크 | 테스트 정확도 |
|---|---:|---:|---:|
| 기본 CNN | 0.2221 | 11 | 0.9121 |
| 개선 CNN | 0.1872 | 12 | 0.9309 |

개선 CNN 모델은 기본 CNN 모델보다 검증 손실이 낮게 나타났습니다.

기본 CNN 최고 검증 손실: 0.2221

개선 CNN 최고 검증 손실: 0.1872

이를 통해 합성곱층 추가와 Batch Normalization 적용이 Fashion-MNIST 분류 성능 개선에 도움이 되었음을 확인했습니다.

## 결과 해석

기본 CNN 모델도 Fashion-MNIST 데이터셋에 대해 정상적으로 학습되었지만, 개선 CNN 모델은 더 낮은 검증 손실을 기록했습니다.

개선 모델에서는 합성곱층을 더 깊게 구성하여 이미지 특징을 더 다양하게 추출할 수 있었고, Batch Normalization을 통해 학습이 더 안정적으로 진행되었습니다.

다만 개선 모델에서도 후반부에는 훈련 손실은 계속 감소하지만 검증 손실은 더 이상 크게 개선되지 않는 모습이 나타났습니다.
따라서 조기 종료를 사용해 과적합이 심해지기 전에 가장 좋은 모델을 저장했습니다.

## 정리

이 프로젝트에서는 Fashion-MNIST 데이터셋을 이용해 PyTorch 기반 CNN 분류 모델을 구현했습니다.

기본 CNN 모델을 먼저 구현하고, 이후 합성곱층 추가와 Batch Normalization을 적용한 개선 CNN 모델을 만들어 성능을 비교했습니다.

실험 결과, 개선 CNN 모델은 기본 CNN 모델보다 낮은 검증 손실을 기록하여 유의미한 성능 개선을 보였습니다.

이를 통해 CNN의 기본 구조, PyTorch 학습 루프, 검증 손실 기반 모델 저장, 조기 종료, 모델 개선 실험 과정을 학습했습니다.
