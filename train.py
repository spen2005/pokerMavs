# train.py

import os
import torch
from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
from agents.mcts import MCTS
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.player import Player

class MCTSPlayer(Player):
    def __init__(self, mcts, player_id):
        super().__init__()
        self.mcts = mcts
        self.player_id = player_id

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.game_state_to_model_state(round_state, hole_card)
        action = self.mcts.mcts_strategy(state, self.player_id, 0, self.mcts.max_depth)
        return valid_actions[action]['action']  # Return action string like 'fold', 'call', 'raise'

    def game_state_to_model_state(self, round_state, hole_card):
        # 将游戏状态转换为模型可以处理的状态
        state = {
            "round_state": round_state,
            "hole_card": hole_card,
            # 添加更多需要的信息
        }
        return state

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def train_model(max_epochs, batch_size, model_save_path, policy_network_params, value_function_params):
    policy_network = PolicyNetwork(**policy_network_params)
    value_function = ValueFunction(**value_function_params)
    mcts = MCTS(max_depth=5, policy_network=policy_network, value_function=value_function)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")

        # Initialize game environment
        emulator = Emulator()
        emulator.set_game_rule(nb_player=6, max_round=10, small_blind_amount=10, ante_amount=1)

        # Register players
        players = [MCTSPlayer(mcts, i) for i in range(6)]
        for i, player in enumerate(players):
            emulator.register_player(f"player{i}", player)

        # Start game and collect data
        emulator.start_game()

        # Update networks
        if (epoch + 1) % batch_size == 0:
            print("Updating networks...")
            mcts.update_networks()

        # Save models
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(mcts.policy_network.state_dict(), os.path.join(model_save_path, "policy_network.pth"))
        torch.save(mcts.value_function.state_dict(), os.path.join(model_save_path, "value_function.pth"))
        print("Models saved.")

if __name__ == "__main__":
    max_epochs = 1000
    batch_size = 10
    model_save_path = "TrainedAgent"

    # Define parameters for PolicyNetwork and ValueFunction
    policy_network_params = {
        "card_types": 52,  # Example parameter, replace with actual parameters
        "state_dim": 100,  # Replace with actual state dimension
        "action_dim": 3,  # Number of possible actions (fold, call, raise)
        "embed_dim": 10  # Example embedding dimension, replace with actual parameter
    }
    value_function_params = {
        "input_size": 100,  # Example parameter, replace with actual parameters
        "hidden_size": 128,
        "output_size": 1
    }

    train_model(max_epochs, batch_size, model_save_path, policy_network_params, value_function_params)
