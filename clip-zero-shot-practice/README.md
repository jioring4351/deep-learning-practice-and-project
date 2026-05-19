# CLIP Zero-Shot Image-Text Matching Practice

## 실습 설명

사전학습된 CLIP 모델을 사용하여 이미지와 텍스트 간의 유사도를 계산하는 VLM 입문 실습입니다.

이미지를 이미지 임베딩으로 변환하고, 텍스트 후보를 텍스트 임베딩으로 변환한 뒤 두 임베딩의 유사도를 비교하여 이미지와 가장 잘 맞는 텍스트를 선택했습니다.

이번 실습에서는 CLIP 모델을 직접 학습시키는 것이 아니라, 이미 학습된 모델을 활용하여 zero-shot image-text matching 과정을 확인했습니다.

## 사용 기술

- Python
- PyTorch
- Hugging Face Transformers
- CLIP
- PIL
- Requests

## 실습 목표

이번 실습의 목표는 VLM의 기본 구조를 이해하는 것입니다.

CLIP은 이미지와 텍스트를 각각 같은 임베딩 공간으로 변환한 뒤, 두 벡터의 유사도를 계산합니다.

이를 통해 별도의 추가 학습 없이도 텍스트 후보를 이용해 이미지를 분류하거나 설명할 수 있습니다.

## 실습 내용

### 1. 기본 이미지-텍스트 매칭

고양이 이미지 1장과 텍스트 후보 5개를 사용하여 이미지와 가장 잘 맞는 텍스트를 찾았습니다.

텍스트 후보는 다음과 같이 구성했습니다.

```text
a photo of a cat
a photo of a dog
a photo of a car
a photo of a cup of coffee
a photo of a person
```

CLIP은 이미지와 각 텍스트 후보의 유사도를 계산했고, 가장 높은 유사도를 가진 텍스트로 `a photo of a cat`을 선택했습니다.

이를 통해 CLIP이 이미지와 텍스트를 직접 비교하여 가장 적절한 설명을 선택할 수 있음을 확인했습니다.

### 2. Prompt 변화 실험

같은 고양이 이미지를 대상으로 텍스트 표현을 바꿨을 때 유사도가 어떻게 달라지는지 확인했습니다.

사용한 prompt는 다음과 같습니다.

```text
cat
a cat
a photo of a cat
a blurry photo of a cat
a photo of two cats
```

실험 결과, `a photo of two cats`의 유사도가 가장 높게 나타났습니다.

이는 이미지 안에 고양이가 두 마리 포함되어 있기 때문에, 단순히 `cat`이라고 표현하는 것보다 이미지 내용을 더 구체적으로 설명한 prompt가 더 높은 유사도를 얻은 것으로 해석할 수 있습니다.

이를 통해 CLIP의 결과는 prompt 표현 방식에 영향을 받을 수 있음을 확인했습니다.

### 3. 여러 이미지 테스트

고양이, 강아지, 자동차, 커피 이미지를 사용하여 각 이미지에 대해 top-1 예측을 확인했습니다.

텍스트 후보는 다음과 같이 구성했습니다.

```text
a photo of a cat
a photo of a dog
a photo of a car
a photo of a cup of coffee
```

실험 결과, 각 이미지에 대해 다음과 같은 예측이 나왔습니다.

| 실제 이미지 | CLIP 예측 텍스트 | 확률 |
|---|---|---:|
| cat | a photo of a cat | 0.9925 |
| dog | a photo of a dog | 0.9975 |
| car | a photo of a car | 0.9987 |
| coffee | a photo of a cup of coffee | 0.9998 |

여러 이미지 테스트에서도 CLIP은 각 이미지와 가장 잘 맞는 텍스트를 높은 확률로 선택했습니다.

이를 통해 CLIP이 별도의 추가 학습 없이도 이미지와 텍스트 후보를 비교하여 zero-shot 방식으로 분류할 수 있음을 확인했습니다.

### 4. 텍스트 후보 구성에 따른 한계 분석

마지막 실습에서는 정답인 `cat` 후보를 제외하고, 비슷하거나 애매한 텍스트 후보만 제공했을 때 CLIP이 어떤 텍스트를 선택하는지 확인했습니다.

사용한 후보는 다음과 같습니다.

```text
a photo of a dog
a photo of a fox
a photo of a tiger
a photo of a sofa
a photo of a pet
```

실험 결과는 다음과 같습니다.

| 텍스트 후보 | 확률 |
|---|---:|
| a photo of a dog | 0.0055 |
| a photo of a fox | 0.0002 |
| a photo of a tiger | 0.0220 |
| a photo of a sofa | 0.2530 |
| a photo of a pet | 0.7193 |

정답 후보인 `cat`이 없을 때 CLIP은 `a photo of a pet`을 가장 높은 유사도를 가진 텍스트로 선택했습니다.

이를 통해 CLIP은 이미지 자체에 대해 절대적인 정답을 출력하는 것이 아니라, 사용자가 제공한 텍스트 후보들 중 이미지와 가장 유사한 후보를 선택한다는 점을 확인했습니다.

## 결과 해석

이번 실습을 통해 CLIP이 이미지와 텍스트를 같은 임베딩 공간에서 비교하는 방식을 이해했습니다.

기본 이미지-텍스트 매칭에서는 고양이 이미지에 대해 `a photo of a cat`을 정확히 선택했습니다.

Prompt 변화 실험에서는 이미지 내용을 더 구체적으로 설명한 `a photo of two cats`가 가장 높은 유사도를 보였습니다.

여러 이미지 테스트에서는 고양이, 강아지, 자동차, 커피 이미지에 대해 모두 올바른 텍스트를 선택했습니다.

마지막 한계 분석에서는 정답 후보가 없을 경우 CLIP이 주어진 후보 중 가장 가까운 의미의 텍스트를 선택한다는 점을 확인했습니다.

## 정리

이 프로젝트에서는 사전학습된 CLIP 모델을 활용하여 VLM의 기본적인 이미지-텍스트 매칭 과정을 실습했습니다.

이미지와 텍스트를 각각 임베딩으로 변환하고, 두 임베딩의 유사도를 계산하여 가장 적절한 텍스트를 선택하는 과정을 확인했습니다.

또한 prompt 표현 방식과 텍스트 후보 구성에 따라 CLIP의 결과가 달라질 수 있음을 실험을 통해 확인했습니다.

이를 통해 VLM에서 prompt 설계와 후보 텍스트 구성의 중요성을 이해할 수 있었습니다.