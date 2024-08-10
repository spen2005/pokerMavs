from enum import Enum, auto

class Action(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
    ALL_IN = auto()

class PokerAction:
    def __init__(self, action_type, amount=0):
        self.action_type = action_type
        self.amount = amount

    def __str__(self):
        if self.action_type in [Action.FOLD, Action.CHECK]:
            return self.action_type.name
        else:
            return f"{self.action_type.name} {self.amount}"

def get_possible_actions(game_state, player_id):
    """
    Determine the possible actions for a player given the current game state.
    """
    possible_actions = []
    player = game_state['seats'][player_id]
    current_bet = max(seat['bet'] for seat in game_state['seats'].values())
    player_bet = player['bet']
    player_stack = player['stack']

    # Fold is always possible
    possible_actions.append(PokerAction(Action.FOLD))

    # Check if the player can check
    if player_bet == current_bet:
        possible_actions.append(PokerAction(Action.CHECK))
    
    # Call
    call_amount = current_bet - player_bet
    if call_amount > 0 and call_amount < player_stack:
        possible_actions.append(PokerAction(Action.CALL, call_amount))

    # Raise
    min_raise = game_state['min_raise']
    if player_stack > call_amount + min_raise:
        possible_actions.append(PokerAction(Action.RAISE, call_amount + min_raise))
        # You might want to add more raise options here, e.g., 2x, 3x min raise

    # All-in
    if player_stack > 0:
        possible_actions.append(PokerAction(Action.ALL_IN, player_stack))

    return possible_actions

def execute_action(game_state, player_id, action):
    """
    Execute the given action and return the updated game state.
    """
    new_state = game_state.copy()  # Create a deep copy of the game state
    player = new_state['seats'][player_id]

    if action.action_type == Action.FOLD:
        player['folded'] = True
    elif action.action_type == Action.CHECK:
        pass  # No change to the game state
    elif action.action_type in [Action.CALL, Action.RAISE, Action.ALL_IN]:
        player['bet'] += action.amount
        player['stack'] -= action.amount
        new_state['pot'] += action.amount

    # Update the current player
    new_state['current_player'] = (player_id + 1) % len(new_state['seats'])

    return new_state