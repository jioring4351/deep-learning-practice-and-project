# DQN CartPole Practice

## 프로젝트 설명

강화학습의 기본 환경인 CartPole에서 DQN을 구현해본 실습 프로젝트입니다.

CartPole은 카트를 왼쪽 또는 오른쪽으로 움직여 막대가 넘어지지 않도록 오래 버티는 문제입니다. 이번 실습에서는 에이전트가 현재 상태를 관찰하고 행동을 선택하며, 환경으로부터 보상을 받아 점점 더 좋은 행동을 학습하는 과정을 구현했습니다.

DQN은 Q-value를 신경망으로 근사하는 강화학습 알고리즘입니다. CartPole의 state를 입력받아 왼쪽 행동과 오른쪽 행동의 Q-value를 출력하고, 더 높은 Q-value를 가진 행동을 선택하도록 학습합니다.

## 사용 기술

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

## 실습 목표

이번 실습의 목표는 강화학습의 기본 구조와 DQN의 학습 흐름을 이해하는 것입니다.

주요 학습 내용은 다음과 같습니다.

- 강화학습의 기본 구성 요소 이해
- CartPole 환경 생성 및 상태/행동 공간 확인
- DQN 모델 구현
- Q-value 개념 이해
- Replay Buffer 구현
- epsilon-greedy 행동 선택 구현
- target network를 이용한 DQN 학습
- 에피소드별 보상 변화 시각화

## 강화학습 핵심 개념

### Agent

환경 안에서 행동을 선택하는 주체입니다.

### Environment

에이전트가 상호작용하는 환경입니다.

### State

현재 환경의 상태입니다. CartPole에서는 다음 4개의 값으로 구성됩니다.

```text
카트 위치
카트 속도
막대 각도
막대 각속도
```

### Action

에이전트가 선택할 수 있는 행동입니다. CartPole에서는 두 가지 행동이 있습니다.

```text
0 → 왼쪽으로 이동
1 → 오른쪽으로 이동
```

### Reward

에이전트가 행동한 결과로 받는 보상입니다. CartPole에서는 막대가 넘어지지 않고 한 step 버틸 때마다 보상 1을 받습니다.

### Q-value

특정 state에서 특정 action을 선택했을 때 기대되는 미래 보상의 총합입니다.

DQN은 현재 state를 입력받아 각 action의 Q-value를 출력합니다.

```text
입력: state 4개
출력: [Q(left), Q(right)]
```

## 실습 내용

### 1. CartPole 환경 이해

Gymnasium의 `CartPole-v1` 환경을 생성하고, 초기 state와 상태 공간, 행동 공간을 확인했습니다.

CartPole의 상태 차원은 4이고, 행동 개수는 2입니다.

```text
상태 차원: 4
행동 개수: 2
```

또한 랜덤 행동을 실행해 `env.step(action)`이 반환하는 값을 확인했습니다.

```text
next_state
reward
terminated
truncated
info
```

이를 통해 강화학습에서 에이전트가 환경과 상호작용하는 기본 구조를 확인했습니다.

### 2. DQN 모델 구현

CartPole의 state를 입력받아 각 action의 Q-value를 출력하는 DQN 모델을 구현했습니다.

모델 구조는 다음과 같습니다.

```text
Input: state_dim = 4
↓
Linear(4 → 128)
↓
ReLU
↓
Linear(128 → 128)
↓
ReLU
↓
Linear(128 → 2)
↓
Output: Q-value for each action
```

출력값 2개는 각각 왼쪽 행동과 오른쪽 행동의 Q-value를 의미합니다.

```text
[Q(left), Q(right)]
```

처음에는 모델이 학습되지 않은 상태이므로 Q-value는 의미 있는 값이라기보다 초기 랜덤 가중치에서 나온 값입니다.

### 3. Replay Buffer 구현

DQN에서는 에이전트가 환경과 상호작용하면서 얻은 경험을 Replay Buffer에 저장합니다.

하나의 경험은 다음과 같은 transition 형태로 저장됩니다.

```text
(state, action, reward, next_state, done)
```

Replay Buffer는 저장된 경험 중 일부를 무작위로 뽑아 미니배치 학습에 사용합니다.

이를 통해 최근 경험에만 치우치지 않고, 다양한 경험을 섞어 더 안정적으로 학습할 수 있습니다.

이번 실습에서는 최대 10,000개의 경험을 저장할 수 있도록 Replay Buffer를 구현했습니다.

```python
replay_buffer = ReplayBuffer(capacity=10000)
```

### 4. epsilon-greedy 행동 선택

DQN에서는 항상 Q-value가 가장 높은 행동만 선택하지 않고, 일정 확률로 랜덤 행동을 선택합니다.

이를 epsilon-greedy 방식이라고 합니다.

```text
epsilon 확률
→ 랜덤 행동 선택

1 - epsilon 확률
→ Q-value가 가장 큰 행동 선택
```

학습 초반에는 epsilon을 크게 두어 다양한 행동을 탐험하고, 학습이 진행될수록 epsilon을 줄여 Q-value 기반 행동 선택을 늘립니다.

이번 실습에서는 다음과 같이 설정했습니다.

```python
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
```

### 5. DQN 학습

Replay Buffer에서 미니배치를 뽑아 DQN을 학습했습니다.

DQN 학습의 핵심은 현재 Q-value와 target Q-value의 차이를 줄이는 것입니다.

현재 Q-value는 `policy_net`이 현재 state에서 실제 선택했던 action에 대해 예측한 값입니다.

target Q-value는 reward와 next_state를 이용해 계산합니다.

```text
target Q-value
= reward + gamma × max Q(next_state)
```

에피소드가 종료된 경우에는 미래 보상을 더하지 않습니다.

이번 실습에서는 학습 안정성을 위해 두 개의 네트워크를 사용했습니다.

```text
policy_net
→ 실제로 학습되는 DQN

target_net
→ target Q-value 계산에 사용하는 고정 기준 네트워크
```

`policy_net`은 매 학습마다 업데이트되고, `target_net`은 일정 주기마다 `policy_net`의 가중치를 복사해 갱신했습니다.

### 6. 학습 결과 확인

200 에피소드 동안 DQN을 학습시켰습니다.

10 에피소드마다 최근 10개 에피소드의 평균 보상과 epsilon 값을 출력했습니다.

학습 결과 일부는 다음과 같습니다.

```text
Episode 10, Average Reward: 19.80, Epsilon: 0.9511
Episode 50, Average Reward: 22.80, Epsilon: 0.7783
Episode 90, Average Reward: 62.50, Epsilon: 0.6369
Episode 120, Average Reward: 81.70, Epsilon: 0.5480
Episode 160, Average Reward: 79.60, Epsilon: 0.4484
Episode 200, Average Reward: 69.00, Epsilon: 0.3670
```

초반에는 평균 보상이 약 20 수준이었지만, 학습이 진행되면서 일부 구간에서는 60~80 수준까지 상승했습니다.

이를 통해 DQN이 CartPole 환경에서 막대를 더 오래 버티는 방향으로 어느 정도 학습되고 있음을 확인했습니다.

다만 평균 보상이 안정적으로 계속 증가하지는 않았고, 중간에 하락하는 구간도 있었습니다. 이는 강화학습의 특성상 학습이 불안정할 수 있고, epsilon이 아직 높아 랜덤 행동이 남아 있었기 때문으로 볼 수 있습니다.

## 시각화

에피소드별 누적 보상과 최근 10개 에피소드의 이동 평균 보상을 그래프로 시각화했습니다.

이를 통해 학습이 진행되면서 보상이 어떻게 변하는지 확인했습니다.

```text
Episode Reward
→ 각 에피소드에서 받은 총 보상

Moving Average Reward
→ 최근 10개 에피소드 평균 보상
```

## 학습 성능 개선 방향

현재 DQN은 평균 보상이 어느 정도 상승했지만, 아직 안정적으로 높은 보상을 유지하지는 못했습니다. 학습을 더 잘되게 하기 위해 다음 5가지를 조정해볼 수 있습니다.

### 1. 에피소드 수 늘리기

`num_episodes` 값을 200에서 500 또는 1000으로 늘리면 더 많은 경험을 바탕으로 학습할 수 있습니다.

예시:

```python
num_episodes = 500
```

### 2. epsilon 감소 속도 조정하기

200 에피소드 이후에도 epsilon이 높게 남아 있으면 랜덤 행동이 많아져 보상이 흔들릴 수 있습니다. `epsilon_decay` 값을 조금 낮추면 랜덤 행동 비율을 더 빠르게 줄일 수 있습니다.

예시:

```python
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
```

### 3. 학습률 낮추기

학습률이 너무 크면 Q-value 학습이 불안정할 수 있습니다. `lr` 값을 낮추면 더 안정적으로 학습될 수 있습니다.

예시:

```python
optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
```

### 4. 초기 Replay Buffer 경험 늘리기

학습 전에 랜덤 경험을 더 많이 쌓아두면 초반 학습이 더 안정될 수 있습니다.

예시:

```python
for step in range(1000):
```

### 5. target_net 업데이트 주기 조정하기

`target_net`은 target Q-value 계산을 안정적으로 하기 위한 기준 모델입니다. 일정 주기마다 `policy_net`의 가중치를 `target_net`에 복사합니다.

예시:

```python
target_update = 10
```

## 정리

이번 프로젝트에서는 CartPole 환경을 이용해 DQN의 기본 흐름을 실습했습니다.

강화학습의 핵심 구성 요소인 state, action, reward, done을 확인하고, DQN 모델을 구현하여 각 행동의 Q-value를 예측하도록 했습니다.

또한 Replay Buffer를 사용해 경험을 저장하고, epsilon-greedy 방식으로 탐험과 활용을 조절했습니다.

마지막으로 target network를 이용해 DQN을 학습시키고, 에피소드별 누적 보상 변화를 시각화했습니다.

이를 통해 DQN의 전체 흐름인 환경 상호작용, 경험 저장, 미니배치 학습, Q-value 업데이트, target network 갱신 과정을 이해할 수 있었습니다.