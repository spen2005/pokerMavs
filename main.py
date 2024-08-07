from agents.mcts import MCTS
from pypokerengine.api.emulator import Emulator
from pypokerengine.players import BasePokerPlayer
import numpy as np

class MCTSPlayer(BasePokerPlayer):
    def __init__(self, max_depth=5):
        self.mcts = MCTS(max_depth)
        self.current_state = None

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.current_state is None:
            # Initialize state if needed
            self.current_state = round_state

        # Use MCTS to determine the best action
        action, action_value = self.mcts.mcts_strategy(self.current_state, self.get_player_id(), 0, self.mcts.max_depth)
        
        # Choose action based on the strategy
        if action not in [a["action"] for a in valid_actions]:
            action = valid_actions[np.random.choice(len(valid_actions))]["action"]

        return action, action_value

    def receive_game_start_message(self, game_info):
        print(f"Game started with info: {game_info}")

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(f"Round {round_count} started.")
        print(f"My hole cards: {hole_card}")
        print(f"Seats: {seats}")

        # 更新回合状态
        self.current_state = {
            'hole_card': hole_card,
            'seats': seats,
            'community_card': [],
            'pot': {'main': {'amount': 0}},
            'street': 'preflop'
        }

    def receive_street_start_message(self, street, round_state):
        print(f"Street {street} started.")
        self.current_state = round_state

    def receive_game_update_message(self, action, round_state):
        print(f"Action: {action}")
        self.current_state = round_state

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(f"Round result: {winners}")
        print(f"Hand info: {hand_info}")
        print(f"Round state: {round_state}")

    def get_player_id(self):
        # Placeholder method to return player's ID, modify based on actual implementation
        return 0

if __name__ == "__main__":
    # Initialize the game
    emulator = Emulator()
    emulator.set_game_rule(nb_player=6, max_round=10, small_blind_amount=10, ante_amount=1)
    
    # Register players
    players = [MCTSPlayer() for _ in range(6)]
    for i, player in enumerate(players):
        emulator.register_player("player{}".format(i), player)
    
    # Start the game
    game_result = emulator.start_game()
    print(game_result)
