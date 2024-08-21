from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import attach_hole_card_from_deck
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.deck import Deck
from environment.actions import ActionType
from agents.mcts import MCTS
from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

# Create the "trained" folder if it doesn't exist
os.makedirs("trained", exist_ok=True)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(num_episodes=1000, num_players=6, update_interval=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_network = PolicyNetwork(num_players=num_players).to(device)
    value_function = ValueFunction(num_players=num_players, num_hand_categories=52).to(device)
    
    policy_optimizer = optim.Adam(policy_network.parameters())
    value_optimizer = optim.Adam(value_function.parameters())

    emulator = Emulator()
    emulator.set_game_rule(player_num=num_players, max_round=1, small_blind_amount=5, ante_amount=0)
    mcts = MCTS(policy_network, value_function, num_players)

    policy_data = []
    value_data = []

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/poker_training")

    each_players_stack_sum = np.zeros(num_players)

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
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

        previous_state = game_state
        each_player_pay_before_this_street = [0 for _ in range(num_players)]

        while game_state['street'] != Const.Street.FINISHED:
            current_player = game_state['next_player']

            if(game_state['street'] != previous_state['street']):
                each_player_pay_before_this_street = [1000-player.stack for player in game_state['table'].seats.players]
                print(each_player_pay_before_this_street)

            previous_state = game_state

            # 使用 MCTS 策略
            avg_expected_values, policy_input, value_input, new_policy, action, amount = mcts.mcts_strategy(game_state, each_player_pay_before_this_street, episode, int(100+0.1*episode))

            # 應用動作
            print(f"convert_action_type: {mcts.convert_action_type(action)}")
            game_state, events = emulator.apply_action(game_state, mcts.convert_action_type(action), amount)
            
            # output game state: community cards, pot size, street, all players' stack
            community_cards = game_state['table'].get_community_card()
            print(f"\nCommunity cards: {[f'{card.rank}-{card.suit}' for card in community_cards]}")
            
            # Calculate the total pot size manually
            total_pot = mcts.get_total_pot(game_state)
            print(f"Pot size: {total_pot}")
            
            print(f"Street: {game_state['street']}")
            print("Players' stack:")

            for player in game_state['table'].seats.players:
                print(f"{player.name}: {player.stack}")
                
            # 收集訓練數據
            policy_data.append((policy_input, new_policy))

            if(game_state['street'] != previous_state['street']):
                # print(f"value_input: {value_input}")
                value_data.append((value_input, avg_expected_values))
        
        # 计算每位选手的平均剩余筹码
        for i, player in enumerate(game_state['table'].seats.players):
            each_players_stack_sum[i] += player.stack
            average_stack = each_players_stack_sum[i] / (episode + 1)
            print("=============")
            print(f"player_{i}_average_stack: {average_stack}")
            print("=============")
            writer.add_scalar(f'Average_Stack/player_{i}', average_stack, episode + 1)

        # 更新網絡
        if (episode + 1) % update_interval == 0:
            policy_loss, value_loss = update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data)
            policy_data = []
            value_data = []

            # Log losses to TensorBoard
            writer.add_scalar('Loss/Policy', policy_loss, episode + 1)
            writer.add_scalar('Loss/Value', value_loss, episode + 1)

        if (episode + 1) % 1000 == 0:
            print(f"\nCompleted {episode + 1} episodes")

    # Save the trained models
    save_model(policy_network, "trained/trained_policy_network.pth")
    save_model(value_function, "trained/trained_value_function.pth")

    print("\nTraining completed")
    writer.close()

def update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data, epochs=150):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_network.to(device)
    value_function.to(device)
    
    for epoch in range(epochs):
        policy_optimizer.zero_grad()
        policy_loss = 0
        for inputs, target in policy_data:
            inputs, target = inputs.to(device), torch.tensor(target, dtype=torch.float32).to(device)
            output = policy_network(inputs)
            # mse loss
            policy_loss += F.mse_loss(output, target)
        policy_loss /= len(policy_data)
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss = 0
        for inputs, target in value_data:
            inputs, target = [i.to(device) for i in inputs], torch.tensor(target, dtype=torch.float32).to(device)
            output = value_function(*inputs)
            if target.dim() == 1:
                target = target.unsqueeze(0)  # Ensure target has the same shape as output
            value_loss += F.mse_loss(output, target)
        value_loss /= len(value_data)
        value_loss.backward()
        value_optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

    return policy_loss.item(), value_loss.item()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train()