import torch
import numpy as np
from pypokerengine.api.emulator import Emulator
from environment.actions import ActionType, get_action_space, BettingStage, PREFLOP_ACTIONS, POSTFLOP_ACTIONS
from .PHE import PokerHandEvaluator
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.hand_evaluator import HandEvaluator

class MCTS:
    def __init__(self, policy_network, value_function, num_players=6):
        self.policy_network = policy_network
        self.value_function = value_function
        self.num_players = num_players
        self.phe = PokerHandEvaluator()
        self.emulator = Emulator()

    def act(self, game_state, hand_strengths, public_strength):
        print("acting...")
        # 轉換遊戲狀態中的卡牌
        game_state = self.convert_game_state_cards(game_state)
        
        betting_stage = BettingStage.PREFLOP if game_state['street'] == 'preflop' else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)

        # 獲取策略網絡的輸出
        policy_input = self.prepare_policy_input(game_state, hand_strengths, public_strength)
        action_probs = self.policy_network(policy_input).detach().numpy().squeeze()

        # 創建動作類型到索引的映射
        action_to_index = {
            ActionType.FOLD: 0,
            ActionType.CALL: 1,
            ActionType.ALL_IN: 2,
            ActionType.RAISE: 3,
            ActionType.BET: 3 + len(PREFLOP_ACTIONS[ActionType.RAISE])
        }

        # 過濾無效動作
        valid_actions = self.get_valid_actions(game_state, action_space)
        valid_probs = []
        valid_actions_expanded = []
        for action in valid_actions:
            if action in [ActionType.FOLD, ActionType.CALL, ActionType.ALL_IN]:
                valid_probs.append(action_probs[action_to_index[action]])
                valid_actions_expanded.append(action)
            elif action == ActionType.RAISE and betting_stage == BettingStage.PREFLOP:
                raise_probs = action_probs[action_to_index[ActionType.RAISE]:action_to_index[ActionType.RAISE]+len(action_space[ActionType.RAISE])]
                valid_probs.extend(raise_probs)
                valid_actions_expanded.extend([ActionType.RAISE] * len(raise_probs))
            elif action == ActionType.BET and betting_stage == BettingStage.POSTFLOP:
                bet_probs = action_probs[action_to_index[ActionType.BET]:action_to_index[ActionType.BET]+len(action_space[ActionType.BET])]
                valid_probs.extend(bet_probs)
                valid_actions_expanded.extend([ActionType.BET] * len(bet_probs))

        valid_probs = np.array(valid_probs)
        valid_probs /= valid_probs.sum()  # 重新歸一化概率

        # 選擇動作
        chosen_action_index = np.random.choice(len(valid_probs), p=valid_probs)
        chosen_action = valid_actions_expanded[chosen_action_index]

        # 如果選擇的是 RAISE 或 BET，還需要選擇具體的金額
        if chosen_action == ActionType.RAISE and betting_stage == BettingStage.PREFLOP:
            raise_index = chosen_action_index - sum(1 for action in valid_actions_expanded[:chosen_action_index] if action != ActionType.RAISE)
            amount = action_space[ActionType.RAISE][raise_index]
        elif chosen_action == ActionType.BET and betting_stage == BettingStage.POSTFLOP:
            bet_index = chosen_action_index - sum(1 for action in valid_actions_expanded[:chosen_action_index] if action != ActionType.BET)
            amount = action_space[ActionType.BET][bet_index]
        else:
            amount = action_space[chosen_action]
            
        print(f"chosen action: {chosen_action}, amount: {amount}")
        return {'action': chosen_action, 'amount': amount}

    def get_valid_actions(self, game_state, action_space):
        print("getting valid actions...")
        player = game_state['seats'][game_state['next_player']]
        stack = player['stack']
        valid_actions = []

        for action in action_space:
            if action == ActionType.FOLD:
                valid_actions.append(action)
            elif action == ActionType.CALL:
                if stack > 0:
                    valid_actions.append(action)
            elif action == ActionType.RAISE:
                if stack >= self.get_action_amount(action, game_state):
                    valid_actions.append(action)
            elif action == ActionType.ALL_IN:
                if stack > 0:
                    valid_actions.append(action)
            elif action == ActionType.BET:
                if stack > 0:
                    valid_actions.append(action)

        return valid_actions

    def get_action_amount(self, action, game_state):
        print("getting action amount...")
        if action in [ActionType.FOLD, ActionType.CALL]:
            return 0
        elif action == ActionType.RAISE:
            return min(game_state['small_blind_amount'] * 2, game_state['seats'][game_state['next_player']]['stack'])
        elif action == ActionType.ALL_IN:
            return game_state['seats'][game_state['next_player']]['stack']
        elif action == ActionType.BET:
            return min(game_state['small_blind_amount'] * 2, game_state['seats'][game_state['next_player']]['stack'])
        else:
            raise ValueError(f"Invalid action: {action}")
        
    def mcts_strategy(self, game_state, num_samples=1000):
        print("computing mcts strategy...")
        if self.is_game_end(game_state):
            return self.calculate_payoff(game_state), None, None

        betting_stage = BettingStage.PREFLOP if game_state['street'] == 'preflop' else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)
        num_players = len(game_state['seats'])

        expected_values = np.zeros((len(action_space), num_players))

        # calculate hand strengths for each player
        hand_strengths = self.calculate_hand_strengths(game_state)
        # calculate public strength
        public_strength = self.calculate_public_strength(game_state)

        for _ in range(num_samples):
            simulated_state = self.emulator.apply_action(game_state, self.act(game_state,hand_strengths,public_strength))
            
            while not self.is_round_end(simulated_state):
                simulated_state = self.emulator.apply_action(simulated_state, self.act(simulated_state))

            value_input = self.prepare_value_input(simulated_state)
            payoffs = self.value_function.evaluate(*value_input)
            
            action_index = action_space.index(simulated_state['action_histories'][betting_stage][0]['action'])
            expected_values[action_index] += np.array(payoffs) / num_samples

        current_player = game_state['next_player']
        current_policy_input = self.prepare_policy_input(game_state, self.calculate_hand_strengths(game_state)[current_player])
        old_policy = self.policy_network(current_policy_input).detach().numpy()

        current_player_expected_values = expected_values[:, current_player]
        avg_expected_value = np.dot(old_policy, current_player_expected_values)
        regrets = current_player_expected_values - avg_expected_value

        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)
        if regret_sum > 0:
            new_policy = positive_regrets / regret_sum
        else:
            new_policy = np.ones_like(old_policy) / len(old_policy)

        avg_expected_values = np.dot(old_policy, expected_values)

        return avg_expected_values, old_policy, new_policy


    def prepare_policy_input(self, game_state, hand_strengths, public_strength):
        print("preparing policy input...")
        current_player_hand_strength = hand_strengths[game_state['next_player']]

        # 準備其他輸入
        round_num = game_state['round_count']
        my_position = game_state['next_player']
        active_players = sum(1 for player in game_state['seats'] if player['state'] == 'participating')
        player_status = [1 if player['state'] == 'participating' else 0 for player in game_state['seats']]
        player_bets = [player['stack'] for player in game_state['seats']]
        min_bet = game_state['small_blind_amount']
        max_bet = max(player['stack'] for player in game_state['seats'])

        # 將所有輸入組合成一個張量
        policy_input = torch.cat([
            torch.tensor(current_player_hand_strength.flatten(), dtype=torch.float32),
            torch.tensor(public_strength.flatten(), dtype=torch.float32),
            torch.tensor([round_num, my_position, active_players, min_bet, max_bet], dtype=torch.float32),
            torch.tensor(player_status, dtype=torch.float32),
            torch.tensor(player_bets, dtype=torch.float32)
        ])

        return policy_input.unsqueeze(0)  # 添加批次維度

    def calculate_hand_strengths(self, game_state):
        print("calculating hand strengths...")
        hand_strengths = []
        for player in game_state['seats']:
            hole_cards = player['hole_card']
            if hole_cards:  # 如果有已知的手牌
                community_cards = game_state['community_card']
                known_cards = self.convert_cards_to_phe_format(hole_cards + community_cards)
                strength_matrix = self.phe.monte_carlo_simulation(known_cards)
            else:  # 如果手牌未知
                strength_matrix = np.zeros((13, 9))  # 假設一個空的強度矩陣
            hand_strengths.append(strength_matrix)
        return hand_strengths

    def calculate_public_strength(self, game_state):
        print("calculating public strength...")
        community_cards = game_state['community_card']
        phe_format_cards = self.convert_cards_to_phe_format(community_cards)
        return self.phe.monte_carlo_simulation(phe_format_cards)

    def convert_cards_to_phe_format(self, cards):
        rank_map = {'A': '14', 'K': '13', 'Q': '12', 'J': '11', 'T': '10'}
        suit_map = {'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
        
        converted_cards = []
        for card in cards:
            # print card for debug
            if len(card) == 2:
                suit, rank = card[0], card[1]
            elif len(card) == 3:
                suit, rank = card[0], card[1:]
            else:
                raise ValueError(f"Invalid card format: {card}")
            
            new_rank = rank_map.get(rank.upper(), rank)
            new_suit = suit_map.get(suit.upper(), suit.upper())
            converted_cards.append(f"{new_rank}{new_suit}")
        
        print(converted_cards)  # 保留這行用於調試
        return converted_cards

    def convert_game_state_cards(self, game_state):
        new_game_state = game_state.copy()
        new_game_state['community_card'] = self.convert_cards_to_phe_format(game_state['community_card'])
        for i, seat in enumerate(new_game_state['seats']):
            if 'hole_card' in seat:
                new_game_state['seats'][i]['hole_card'] = self.convert_cards_to_phe_format(seat['hole_card'])
        return new_game_state

    def is_round_end(self, game_state):
        return game_state['street'] in ['preflop', 'flop', 'turn', 'river'] and \
               all(player['action_histories'][game_state['street']] for player in game_state['seats'])

    def is_game_end(self, game_state):
        return game_state['street'] == 'showdown'

    def calculate_payoff(self, game_state):
        payoffs = [0] * self.num_players
        
        # Ensure the game is at showdown stage
        if game_state['street'] != Const.Street.SHOWDOWN:
            raise ValueError("Game is not at showdown stage")

        # Get all players who have not folded
        active_players = [player for player in game_state['seats'] if player['state'] != 'folded']
        
        # Evaluate the hand strength of each player
        for player in active_players:
            player['hand_value'] = HandEvaluator.eval_hand(player['hole_card'], game_state['community_card'])

        # Sort players by hand strength
        active_players.sort(key=lambda p: p['hand_value'], reverse=True)

        # Distribute the main pot
        main_pot = game_state['pot']['main']['amount']
        self._assign_pot_to_winners(active_players, main_pot, payoffs)

        # Distribute side pots
        side_pots = game_state['pot'].get('side', [])
        for side_pot in side_pots:
            eligible_players = [p for p in active_players if p['stack'] > side_pot['amount']]
            self._assign_pot_to_winners(eligible_players, side_pot['amount'], payoffs)

        return payoffs

    def _assign_pot_to_winners(self, eligible_players, pot_amount, payoffs):
        if not eligible_players:
            return
        
        best_hand_value = eligible_players[0]['hand_value']
        winners = [p for p in eligible_players if p['hand_value'] == best_hand_value]
        
        win_amount = pot_amount // len(winners)
        for winner in winners:
            player_idx = next(i for i, p in enumerate(self.game_state['seats']) if p['uuid'] == winner['uuid'])
            payoffs[player_idx] += win_amount