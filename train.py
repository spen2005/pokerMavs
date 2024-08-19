from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import attach_hole_card_from_deck
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.deck import Deck
from agents.mcts import MCTS
from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def train(num_episodes=10000, num_players=6, update_interval=100):
    policy_network = PolicyNetwork(num_players=num_players)
    value_function = ValueFunction(num_players=num_players, num_hand_categories=117)
    
    policy_optimizer = optim.Adam(policy_network.parameters())
    value_optimizer = optim.Adam(value_function.parameters())

    emulator = Emulator()
    emulator.set_game_rule(player_num=num_players, max_round=1, small_blind_amount=5, ante_amount=0)
    mcts = MCTS(policy_network, value_function, num_players)

    policy_data = []
    value_data = []

    for episode in range(num_episodes):
        players_info = {
            f"player_{i}": {"name": f"player_{i}", "stack": 1000}
            for i in range(num_players)
        }
        initial_state = emulator.generate_initial_game_state(players_info)
        game_state, events = emulator.start_new_round(initial_state)

        # 為每個玩家分發手牌
        for player in game_state['table'].seats.players:
            game_state = attach_hole_card_from_deck(game_state, player.uuid)
        
        # print the hand cards of each player
        print("Players' hand cards:")
        for player in game_state['table'].seats.players:
            hand_cards = [f"{card.rank}-{card.suit}" for card in player.hole_card]
            print(f"{player.name}: {hand_cards}")

        while game_state['street'] != Const.Street.FINISHED:
            current_player = game_state['next_player']
            
            # 使用 MCTS 策略
            avg_expected_values, policy_input, new_policy, action, amount = mcts.mcts_strategy(game_state)
            
            # 應用動作
            game_state, events = emulator.apply_action(game_state, action, amount)
            
            # 收集訓練數據
            policy_data.append((policy_input, new_policy))
            value_data.append((policy_input, avg_expected_values))
        
        # 更新網絡
        if (episode + 1) % update_interval == 0:
            update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data)
            policy_data = []
            value_data = []

        if (episode + 1) % 1000 == 0:
            print(f"\nCompleted {episode + 1} episodes")

    print("\nTraining completed")

def update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data):
    policy_optimizer.zero_grad()
    policy_loss = 0
    for inputs, target in policy_data:
        output = policy_network(inputs)
        policy_loss += F.kl_div(output.log(), torch.tensor(target), reduction='batchmean')
    policy_loss /= len(policy_data)
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss = 0
    for inputs, target in value_data:
        output = value_function(*inputs)
        value_loss += F.mse_loss(output, torch.tensor(target))
    value_loss /= len(value_data)
    value_loss.backward()
    value_optimizer.step()

    print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

if __name__ == "__main__":
    train()