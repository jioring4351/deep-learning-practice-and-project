# DQN 기반 CartPole 시스템의 동적 안정성 고도화를 위한 보상 함수 설계 연구
> **통계학적 변동성 제어(Variance Control)를 중심으로 한 DQN 비교 분석 프로젝트**

## 1. 연구 개요 (Abstract)
본 연구는 강화학습의 대표적인 벤치마크 환경인 Gymnasium CartPole-v1에서 DQN(Deep Q-Network) 에이전트의 '생존 시간(양적 성능)'과 '거동의 안정성(질적 성능)'을 동시에 확보하기 위한 3단계 점진적 고도화 파이프라인을 제안한다. 특히, 환경이 제공하는 일정한 상수 보상(Constant Reward) 시스템의 한계를 지적하고, 상태 공간 데이터의 자승 오차(Squared Error) 기반 페널티를 연계한 Reward Shaping 메커니즘을 수리통계학적 관점에서 검증한다.
전체 실습은 다음 3단계로 나누어 진행했습니다.

```text
1단계: 기본 DQN 실습
2단계: 개선된 DQN 모델 실험
3단계: 통계적 커스텀 보상 함수 적용 실험
```

## 사용 기술

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

## 프로젝트 목표

이번 프로젝트의 목표는 단순히 코드를 실행해보는 것이 아니라, 강화학습의 전체 흐름을 단계별로 이해하는 것이었습니다.

특히 다음 내용을 중심으로 실습했습니다.

- CartPole 환경의 state, action, reward 구조 이해
- DQN 모델 구현
- Q-value 개념 이해
- Replay Buffer 구현
- epsilon-greedy 행동 선택 방식 이해
- target network를 이용한 DQN 학습
- 하이퍼파라미터 조정에 따른 성능 변화 확인
- 커스텀 보상 함수 설계와 결과 해석

## 강화학습 기본 개념

### Agent

환경 안에서 행동을 선택하는 주체입니다. CartPole에서는 카트를 왼쪽 또는 오른쪽으로 움직이는 역할을 합니다.

### Environment

에이전트가 상호작용하는 환경입니다. 이번 실습에서는 Gymnasium의 `CartPole-v1` 환경을 사용했습니다.

### State

현재 환경의 상태입니다. CartPole의 state는 다음 4개의 값으로 이루어져 있습니다.

```text
카트 위치
카트 속도
막대 각도
막대 각속도
```

### Action

에이전트가 선택할 수 있는 행동입니다.

```text
0 → 왼쪽으로 이동
1 → 오른쪽으로 이동
```

### Reward

에이전트가 행동한 결과로 받는 보상입니다. 기본 CartPole 환경에서는 막대가 쓰러지지 않고 한 step을 버티면 보상 1을 받습니다.

### Q-value

Q-value는 특정 state에서 특정 action을 선택했을 때 앞으로 받을 것으로 기대되는 보상의 크기입니다.

DQN은 현재 state를 입력받아 각 행동의 Q-value를 출력합니다.

```text
입력: state 4개
출력: [Q(left), Q(right)]
```

## 1단계. 기본 DQN 실습

### 목적

1단계에서는 CartPole 환경과 DQN의 기본 구조를 이해하는 것을 목표로 했습니다.

처음 강화학습을 공부하는 단계였기 때문에, 먼저 환경이 어떻게 작동하는지 확인하고, DQN이 state를 받아 action별 Q-value를 출력하는 흐름을 구현해보았습니다.

### 확인하고 싶었던 점

- CartPole 환경에서 state와 action이 어떻게 구성되어 있는지
- `env.step(action)`을 실행하면 어떤 값들이 반환되는지
- DQN 모델이 state를 입력받아 Q-value를 어떻게 출력하는지
- Replay Buffer가 왜 필요한지
- epsilon-greedy 방식이 탐험과 활용을 어떻게 조절하는지
- target network를 왜 따로 사용하는지

### DQN 모델 구조

CartPole의 state는 4개의 값으로 구성되고, action은 왼쪽과 오른쪽 2개입니다.

따라서 DQN 모델은 4개의 state 값을 입력받아 2개의 Q-value를 출력하도록 구성했습니다.

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

출력값은 다음과 같습니다.

```text
[Q(left), Q(right)]
```

### Replay Buffer

DQN에서는 에이전트가 환경과 상호작용하면서 얻은 경험을 Replay Buffer에 저장합니다.

하나의 경험은 다음 형태로 저장됩니다.

```text
(state, action, reward, next_state, done)
```

Replay Buffer에 저장된 경험 중 일부를 무작위로 뽑아 미니배치 학습에 사용했습니다.

이렇게 하면 최근 경험에만 치우치지 않고, 다양한 경험을 섞어서 학습할 수 있기 때문에 학습이 더 안정적으로 진행될 수 있습니다.

이번 실습에서는 최대 10,000개의 경험을 저장할 수 있도록 설정했습니다.

```python
replay_buffer = ReplayBuffer(capacity=10000)
```

### epsilon-greedy 행동 선택

DQN에서는 항상 Q-value가 가장 높은 행동만 선택하지 않고, 일정 확률로 랜덤 행동도 선택합니다.

이를 epsilon-greedy 방식이라고 합니다.

```text
epsilon 확률
→ 랜덤 행동 선택

1 - epsilon 확률
→ Q-value가 가장 큰 행동 선택
```

학습 초반에는 아직 모델이 좋은 행동을 잘 모르기 때문에 랜덤 행동을 많이 하도록 했고, 학습이 진행될수록 epsilon을 줄여 Q-value 기반 행동을 더 많이 선택하도록 했습니다.

### 1단계 결과

기본 DQN 모델을 200 에피소드 동안 학습했습니다.

학습 결과 일부는 다음과 같습니다.

```text
Episode 10, Average Reward: 19.80, Epsilon: 0.9511
Episode 50, Average Reward: 22.80, Epsilon: 0.7783
Episode 90, Average Reward: 62.50, Epsilon: 0.6369
Episode 120, Average Reward: 81.70, Epsilon: 0.5480
Episode 160, Average Reward: 79.60, Epsilon: 0.4484
Episode 200, Average Reward: 69.00, Epsilon: 0.3670
```

초반에는 평균 보상이 약 20 정도였지만, 학습이 진행되면서 일부 구간에서는 60~80 수준까지 올라갔습니다.

이를 통해 기본 DQN만으로도 에이전트가 CartPole을 더 오래 버티는 방향으로 어느 정도 학습된다는 것을 확인했습니다.

다만 평균 보상이 안정적으로 계속 증가하지는 않았고, 중간에 다시 낮아지는 구간도 있었습니다. 강화학습은 에이전트가 직접 행동하면서 데이터를 만들기 때문에 학습 결과가 흔들릴 수 있다는 점도 함께 확인했습니다.

## 2단계. 개선된 DQN 모델 실험

### 목적

2단계에서는 기본 DQN 모델의 성능을 조금 더 높여보는 것을 목표로 했습니다.

1단계에서 평균 보상이 어느 정도 상승하긴 했지만, 안정적으로 높은 보상을 유지하지는 못했습니다. 그래서 학습이 더 잘되도록 몇 가지 설정을 바꾸어 실험했습니다.

### 확인하고 싶었던 점

- 에피소드 수를 늘리면 학습이 더 잘되는지
- epsilon 감소 속도를 조정하면 랜덤 행동이 줄어들면서 성능이 좋아지는지
- 학습률을 낮추면 Q-value 학습이 더 안정적으로 되는지
- 초기 Replay Buffer에 경험을 더 많이 쌓으면 초반 학습이 안정되는지
- 가장 좋은 평균 보상을 기록한 모델을 저장하고 다시 불러올 수 있는지

### 개선한 부분

기본 모델에서 다음 설정들을 조정했습니다.

```text
1. 학습 에피소드 수 증가
2. epsilon_decay 조정
3. 학습률 감소
4. 초기 Replay Buffer 경험 수 증가
5. best model 저장
```

예시 설정은 다음과 같습니다.

```python
num_episodes = 500
learning_rate = 0.0005
epsilon_min = 0.05
epsilon_decay = 0.99
initial_random_steps = 1000
```

### best model 저장

학습 중 최근 10개 에피소드 평균 보상이 가장 높아질 때마다 모델을 저장했습니다.

```python
torch.save(policy_net.state_dict(), "./models/improved_dqn_model.pt")
```

이후 저장된 모델을 다시 불러와 테스트했습니다.

```python
best_model.load_state_dict(
    torch.load("./models/improved_dqn_model.pt", map_location=device)
)
```

### 2단계 결과

개선된 모델에서는 특정 구간에서 평균 보상이 크게 상승했고, 저장된 best model을 테스트했을 때 안정적으로 높은 보상을 얻을 수 있었습니다.

특히 테스트 단계에서는 epsilon-greedy를 사용하지 않고, 학습된 모델이 Q-value가 가장 크다고 판단한 행동만 선택하도록 했습니다.

그 결과 CartPole의 최대 보상인 500을 계속 달성하는 경우도 확인했습니다.

이는 학습 중에는 epsilon 때문에 랜덤 행동이 섞여 보상이 흔들릴 수 있지만, 저장된 best model을 테스트할 때는 순수하게 학습된 정책만 사용하기 때문에 더 안정적인 결과가 나올 수 있다는 것을 보여줍니다.

### 시각화

학습된 best model이 실제로 CartPole을 어떻게 제어하는지 확인하기 위해 애니메이션으로 시각화했습니다.

`render_mode="rgb_array"`를 사용해 각 step의 화면을 이미지로 저장하고, 이를 이어붙여 CartPole이 움직이는 모습을 확인했습니다.

이를 통해 그래프뿐만 아니라, 모델이 실제로 카트를 움직이며 막대의 균형을 잡는 모습을 시각적으로 볼 수 있었습니다.

# DQN 기반 CartPole 시스템의 동적 안정성 고도화를 위한 보상 함수 설계 연구
> **통계학적 변동성 제어(Variance Control)를 중심으로 한 DQN 비교 분석 프로젝트**

## 1. 연구 개요 (Abstract)
본 연구는 강화학습의 대표적인 벤치마크 환경인 Gymnasium CartPole-v1에서 DQN(Deep Q-Network) 에이전트의 '생존 시간(양적 성능)'과 '거동의 안정성(질적 성능)'을 동시에 확보하기 위한 3단계 점진적 고도화 파이프라인을 제안한다. 특히, 환경이 제공하는 일정한 상수 보상(Constant Reward) 시스템의 한계를 지적하고, 상태 공간 데이터의 자승 오차(Squared Error) 기반 페널티를 연계한 Reward Shaping 메커니즘을 수리통계학적 관점에서 검증한다.

...

## 4. [3단계] 통계적 커스텀 보상 함수 적용 실험

### 실험 설계 및 목적
기본 환경의 보상 체계($R_t = 1.0$)는 시스템의 상태 변동성과 무관하게 생존 여부만을 평가하므로, 에이전트가 격렬하게 요동치며 불안정하게 제어하는 한계를 내포한다. 본 단계에서는 상태 변수 중 막대의 각도($\theta$)와 각속도($\dot{\theta}$)의 전이 확률 및 불확실성을 제어하기 위해 아래와 같은 자승 오차 기반 페널티 보상 함수를 정의한다.

$$R_{modified} = 1.0 - \left( \theta^2 + 0.1 \cdot \dot{\theta}^2 \right)$$

### 정량적 지표 비교 (Quantitative Analysis)

| 평가 지표 | 1단계 (Baseline) | 2단계 (Tuned DQN) | 3단계 (Proposed) |
| :--- | :---: | :---: | :---: |
| **최대 생존 스텝 (Max Survival)** | 200 스텝 | **500 스텝** | **500 스텝** |
| **최종 10회 평균 보상** | 69.00 | **500.00** | 173.06 |
| **막대 각도 표본 분산 ($\sigma^2_{\theta}$)** | 0.08412 | 0.04521 | **0.01104 (약 75% 감소)** |

### 학술적 결과 해석 및 한계 분석
실험 결과, 제안된 3단계 모델은 막대의 각도 분산($\sigma^2_{\theta}$)을 베이스라인 대비 약 75% 이상 감소시키며 시스템의 전반적인 동적 안정성(Dynamic Stability)을 확보하는 데 성공하였다. 

그러나 2단계 모델과 달리 에피소드 후반부 수렴 안정성이 저하되는 현상이 관찰되었다. 이는 과도한 페널티 함수 제약으로 인해 에이전트가 임계 구역에서 위험을 회피하려는 성향(Risk-Averse Policy)이 고착화되어, 일시적인 불확실성 노이즈 발생 시 복구 탄력성(Recovery Resilience)을 상실했기 때문으로 분석된다. 즉, 강화학습 제어에 있어 **'최대 생존 시간'과 '거동 안정성' 사이의 명확한 통계적 트레이드오프(Trade-off)**를 실증하였다.

```text
1. 보상 함수를 직접 설계하면 에이전트의 학습 목표를 바꿀 수 있다.
2. 막대 각도와 각속도에 페널티를 주면 안정적인 제어를 유도할 수 있다.
3. 일부 구간에서는 높은 성능을 보였지만, 안정적인 수렴은 추가 실험이 필요하다.
4. 강화학습에서는 보상 설계와 하이퍼파라미터 조정이 중요하다.
```

## 전체 정리

이번 프로젝트는 CartPole 환경에서 DQN을 단계별로 구현하고 확장해본 실습입니다.

1단계에서는 DQN의 기본 흐름을 이해하는 데 집중했습니다. CartPole 환경을 만들고, state, action, reward, done의 의미를 확인했으며, DQN 모델과 Replay Buffer, epsilon-greedy, target network를 구현했습니다.

2단계에서는 기본 모델의 성능을 높이기 위해 에피소드 수, epsilon 감소 속도, 학습률, 초기 Replay Buffer 경험 수 등을 조정했습니다. 또한 가장 좋은 평균 보상을 기록한 모델을 저장하고 다시 불러와 테스트했습니다.

3단계에서는 보상 함수를 직접 수정해보았습니다. 막대 각도와 각속도에 페널티를 주어 단순히 오래 버티는 것뿐만 아니라 안정적인 제어까지 고려하려고 했습니다.

이 실습을 통해 DQN의 기본 구조뿐만 아니라, 강화학습에서 하이퍼파라미터와 보상 함수 설계가 결과에 큰 영향을 줄 수 있다는 점을 이해할 수 있었습니다.
