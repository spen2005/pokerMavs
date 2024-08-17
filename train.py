from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from agents.mcts import MCTS
from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
from environment.actions import ActionType, get_action_space, BettingStage
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pypokerengine.players import BasePokerPlayer

class MCTSPlayer(BasePokerPlayer):
    def __init__(self, policy_network, value_function, num_players=6, max_round=1, initial_stack=1000, small_blind_amount=5):
        super().__init__()
        self.mcts = MCTS(policy_network, value_function, num_players, max_round, initial_stack, small_blind_amount)
        self.policy_network = policy_network
        self.value_function = value_function
        self.last_action_info = None

    def declare_action(self, valid_actions, hole_card, round_state):
        game_state = self.convert_round_state_to_game_state(round_state, hole_card)
        
        print("\n--- Current Game State ---")
        print(f"Round: {game_state['round_count']}")
        print(f"Street: {game_state['street']}")
        print(f"Community Cards: {game_state['community_card']}")
        print(f"Pot: {game_state['pot']}")
        print(f"Next Player: {game_state['next_player']}")
        
        print("\nPlayers:")
        for i, seat in enumerate(game_state['seats']):
            print(f"Player {i}:")
            print(f"  Stack: {seat['stack']}")
            print(f"  State: {seat['state']}")
            if i == game_state['next_player']:
                print(f"  Hole Cards: {seat['hole_card']}")
            if 'amount' in seat:
                print(f"  Amount: {seat['amount']}")
            else:
                print(f"  Bet: {seat.get('bet', 'N/A')}")
        
        print("\nValid Actions:")
        for action in valid_actions:
            print(f"  {action}")
        
        # 使用 MCTS 策略
        avg_expected_values, old_policy, new_policy = self.mcts.mcts_strategy(game_state)
        
        print(f"\nMCTS Strategy:")
        print(f"Expected Values: {avg_expected_values}")
        print(f"Old Policy: {old_policy}")
        print(f"New Policy: {new_policy}")
        
        # 準備策略輸入
        policy_input = self.mcts.prepare_policy_input(game_state)
        
        # 根據新策略選擇動作
        action_probs = new_policy
        action_type = np.random.choice(len(action_probs), p=action_probs)
        action = self.mcts.get_action_from_type(action_type, game_state)
        
        print(f"\nChosen Action: {action['action']}, Amount: {action['amount']}")
        
        # 存儲這次動作的信息，以便在訓練時使用
        self.last_action_info = {
            'policy_input': policy_input,
            'new_policy': new_policy,
            'avg_expected_values': avg_expected_values
        }
        
        return action['action'].value, action['amount']

    def receive_game_start_message(self, game_info):
        self.game_info = game_info
        print("\n=== Game Start ===")
        print(f"Game Info: {game_info}")

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.seats = seats
        print(f"\n--- Round {round_count} Start ---")
        print(f"Hole Cards: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        print(f"\n--- {street.capitalize()} Start ---")
        print(f"Community Cards: {round_state['community_card']}")

    def receive_game_update_message(self, action, round_state):
        print(f"\nGame Update: {action}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        print("\n=== Round Result ===")
        print(f"Winners: {winners}")
        print(f"Hand Info: {hand_info}")
        print(f"Final Community Cards: {round_state['community_card']}")
        print("Final Stacks:")
        for seat in round_state['seats']:
            print(f"  Player {seat['uuid']}: {seat['stack']}")

    @staticmethod
    def convert_round_state_to_game_state(round_state, hole_card):
        game_state = {
            'next_player': round_state['next_player'],
            'round_count': round_state['round_count'],
            'small_blind_amount': round_state['small_blind_amount'],
            'street': round_state['street'],
            'community_card': round_state['community_card'],
            'pot': round_state['pot']['main']['amount'],
        }

        # 添加玩家信息，包括手牌
        seats = []
        for i, player in enumerate(round_state['seats']):
            player_info = player.copy()
            if i == round_state['next_player']:
                player_info['hole_card'] = hole_card
            else:
                player_info['hole_card'] = []  # 其他玩家的手牌未知
            seats.append(player_info)
        game_state['seats'] = seats

        return game_state

def train(num_games=1000, num_players=6, update_interval=5):
    policy_network = PolicyNetwork(num_players=num_players)
    value_function = ValueFunction(num_players=num_players, num_hand_categories=117)
    
    policy_optimizer = optim.Adam(policy_network.parameters())
    value_optimizer = optim.Adam(value_function.parameters())

    policy_data = []
    value_data = []

    max_round = 1
    initial_stack = 1000
    small_blind_amount = 5

    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
    players = [MCTSPlayer(policy_network, value_function, num_players, max_round, initial_stack, small_blind_amount) for _ in range(num_players)]
    for i, player in enumerate(players):
        config.register_player(name=f"player_{i}", algorithm=player)
    
    for game in range(num_games):
        print(f"\n--- Starting Game {game + 1} ---")
        game_result = start_poker(config, verbose=0)
        
        # 收集這局遊戲中所有玩家的動作信息
        for player in players:
            if player.last_action_info:
                policy_data.append((player.last_action_info['policy_input'], player.last_action_info['new_policy']))
                if player == players[0]:  # 只使用第一個玩家的價值數據
                    value_data.append((player.last_action_info['policy_input'], player.last_action_info['avg_expected_values']))
            player.last_action_info = None  # 重置動作信息

        print(f"\n--- Game {game + 1} Completed ---")
        print(f"Winner(s): {game_result['winners']}")
        for player in game_result['players']:
            print(f"Player {player['uuid']} final stack: {player['stack']}")

        if (game + 1) % update_interval == 0:
            update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data)
            policy_data = []
            value_data = []
            print(f"\nNetworks updated after game {game + 1}")

        if (game + 1) % 100 == 0:
            print(f"\nCompleted {game + 1} games")

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