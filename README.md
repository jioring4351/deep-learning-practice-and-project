# Deep Learning Practice and Project

딥러닝, VLM, 강화학습의 기초 개념을 직접 실습하며 정리한 포트폴리오입니다.

## 실습 목록

### 1. Fashion-MNIST CNN Practice

PyTorch를 사용해 Fashion-MNIST 이미지 분류 CNN 모델을 구현했습니다.

기본 CNN 모델을 학습한 뒤, 개선된 CNN 모델을 만들어 성능 변화를 확인했습니다.

### 2. CLIP Zero-Shot Practice

사전학습된 CLIP 모델을 활용해 이미지와 텍스트 후보 간의 유사도를 계산했습니다.

기본 이미지-텍스트 매칭, prompt 변화 실험, 여러 이미지 테스트를 진행했습니다.

### 3. DQN CartPole Practice

Gymnasium의 CartPole 환경에서 DQN을 구현했습니다.

기본 DQN 실습, 성능 개선 실험, 커스텀 보상 함수 적용 실험을 단계별로 진행했습니다.

## 학습 방식

구현 과정에서는 AI 도구의 도움을 받아 코드 구조를 빠르게 잡았습니다.

이후 각 셀을 직접 실행하면서 입력과 출력, 텐서 shape, 학습 과정, 결과 해석을 하나씩 확인했습니다.

특히 CNN의 학습 구조, CLIP의 이미지-텍스트 유사도 계산, DQN의 Replay Buffer와 Q-value 계산 과정을 중점적으로 복습했습니다.

## 정리

이 저장소는 완성된 연구 프로젝트라기보다는, AI/VLM/강화학습 연구를 따라가기 위해 필요한 기초 개념을 직접 구현하고 이해하기 위한 실습 기록입니다.