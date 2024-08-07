from agents.mcts import MCTS
from environment.state import transition, dealer_deals

# 初始化游戏环境和MCTS策略
max_depth = 5
mcts = MCTS(max_depth)

# 假设T为总的游戏步数，batch_size为更新策略网络和价值函数的频率
T = 1000
batch_size = 10

for t in range(T):
    if t % batch_size == 0:
        mcts.value_function.update(mcts.dataset_v)
        mcts.policy_network.update(mcts.dataset_p)

    state = None  # 初始化状态
    player = 0   # 当前玩家

    if player != 0:  # 不是庄家回合
        expected_value = mcts.mcts_strategy(state, player, 0, max_depth)
        policy = mcts.policy_network.forward(state)
        action = np.random.choice(len(policy), p=policy)
        state = transition(state, action)
    else:
        state = transition(state, dealer_deals(state))
