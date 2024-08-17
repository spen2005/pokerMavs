import torch
import numpy as np
from pypokerengine.api.emulator import Emulator
from environment.actions import ActionType, get_action_space, BettingStage, PREFLOP_ACTIONS, POSTFLOP_ACTIONS
from .PHE import PokerHandEvaluator
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.table import Table
from pypokerengine.engine.player import Player
from pypokerengine.engine.deck import Deck
from pypokerengine.engine.card import Card
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.deck import Deck
from pypokerengine.engine.player import Player
from pypokerengine.engine.table import Table

class MCTS:
    def __init__(self, policy_network, value_function, num_players=6, max_round=1, initial_stack=1000, small_blind_amount=5, ante_amount=0):
        self.policy_network = policy_network
        self.value_function = value_function
        self.num_players = num_players
        self.phe = PokerHandEvaluator()
        
        # 設置遊戲配置
        self.config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
        
        # 初始化 Emulator
        self.emulator = Emulator()
        self.emulator.set_game_rule(num_players, max_round, small_blind_amount, ante_amount)
        
        # 創建初始遊戲狀態
        players_info = {
            f"player_{i}": {"name": f"player_{i}", "stack": initial_stack}
            for i in range(num_players)
        }
        initial_state = self.emulator.generate_initial_game_state(players_info)
        
        # 設置 Emulator 的初始狀態
        self.initial_state = initial_state

    def act(self, game_state, hand_strengths, public_strength):
        # 轉換遊戲狀態中的卡牌
        game_state = self.convert_game_state_cards(game_state)
        
        betting_stage = BettingStage.PREFLOP if game_state['street'] == Const.Street.PREFLOP else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)

        # 獲取當前玩家
        current_player = game_state['table'].seats.players[game_state['next_player']]
        
        # 獲取當前玩家的手牌強度
        current_player_index = game_state['next_player']
        current_player_hand_strengths = hand_strengths[current_player_index]
        
        # 打印調試信息
        print(f"Current player: {current_player.name}")
        
        # 獲取策略網絡的輸出
        policy_input = self.prepare_policy_input(game_state, current_player_hand_strengths, public_strength)
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
            if action in [ActionType.FOLD, ActionType.CALL]:
                valid_probs.append(action_probs[action_to_index[action]])
                valid_actions_expanded.append(action)
            elif action == ActionType.ALL_IN:
                valid_probs.append(action_probs[action_to_index[action]])
                valid_actions_expanded.append(ActionType.ALL_IN)
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

        # 計算動作金額
        current_player = game_state['table'].seats.players[game_state['next_player']]
        
        print(current_player.name)
        print("acting...")
        
        if chosen_action == ActionType.FOLD:
            amount = 0
        elif chosen_action == ActionType.CALL:
            amount = self.get_call_amount(game_state)
        elif chosen_action == ActionType.ALL_IN:
            amount = current_player.stack
        elif chosen_action in [ActionType.RAISE, ActionType.BET]:
            if betting_stage == BettingStage.PREFLOP:
                raise_index = chosen_action_index - sum(1 for action in valid_actions_expanded[:chosen_action_index] if action != ActionType.RAISE)
                amount = min(action_space[ActionType.RAISE][raise_index], current_player.stack)
            else:  # POSTFLOP
                bet_index = chosen_action_index - sum(1 for action in valid_actions_expanded[:chosen_action_index] if action != ActionType.BET)
                amount = min(action_space[ActionType.BET][bet_index], current_player.stack)
        else:
            raise ValueError(f"Unknown action type: {chosen_action}")

        print(f"chosen action: {chosen_action}, amount: {amount}")
        return {'action': self.convert_action_type(chosen_action), 'amount': amount}

    def get_valid_actions(self, game_state, action_space):

        player = game_state['table'].seats.players[game_state['next_player']]
        stack = player.stack
        valid_actions = []
        print(player.name)
        print("getting valid actions...")

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
            return min(game_state['small_blind_amount'] * 2, game_state['table'].seats.players[game_state['next_player']].stack)
        elif action == ActionType.ALL_IN:
            return game_state['table'].seats.players[game_state['next_player']].stack
        elif action == ActionType.BET:
            return min(game_state['small_blind_amount'] * 2, game_state['table'].seats.players[game_state['next_player']].stack)
        else:
            raise ValueError(f"Invalid action: {action}")
        
    def mcts_strategy(self, game_state, num_samples=1000):
        print("computing mcts strategy...")
        if self.is_game_end(game_state):
            return self.calculate_payoff(game_state), None, None

        betting_stage = BettingStage.PREFLOP if game_state['street'] == Const.Street.PREFLOP else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)
        num_players = len(game_state['table'].seats.players)

        expected_values = np.zeros((len(action_space), num_players))

        # calculate hand strengths for each player
        hand_strengths = self.calculate_hand_strengths(game_state)
        # calculate public strength
        public_strength = self.calculate_public_strength(game_state)

        for _ in range(num_samples):
            simulated_state = game_state.copy()
            
            action = self.act(simulated_state, hand_strengths, public_strength)
            simulated_state, _ = self.emulator.apply_action(simulated_state, action['action'], action['amount'])

            while not self.is_round_end(simulated_state):
                action = self.act(simulated_state, self.calculate_hand_strengths(simulated_state), self.calculate_public_strength(simulated_state))
                simulated_state, _ = self.emulator.apply_action(simulated_state, action['action'], action['amount'])

            value_input = self.prepare_value_input(simulated_state)
            payoffs = self.value_function.evaluate(*value_input)
            
            action_index = action_space.index(action['action'])
            expected_values[action_index] += np.array(payoffs) / num_samples

        current_player = game_state['next_player']
        current_player_hand_strength = hand_strengths[current_player]
        
        policy_input = self.prepare_policy_input(game_state, current_player_hand_strength, public_strength)

        old_policy = self.policy_network(policy_input).detach().numpy()

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
    
    def convert_to_pye_game_state(self, game_state):
        table = Table()
        
        # 設置基本信息
        table.dealer_btn = game_state.get('dealer_btn', 0)
        table.set_blind_pos(game_state.get('sb_pos', 0), game_state.get('bb_pos', 1))
        
        # 設置玩家
        for seat in game_state['seats']:
            player = Player(seat['uuid'], seat['stack'], seat['name'])
            player.hole_card = [Card.from_str(c) for c in seat.get('hole_card', [])]
            player.state = seat['state']
            table.seats.sitdown(player)
        
        # 設置社區牌
        for card in game_state['community_card']:
            table.add_community_card(Card.from_str(card))
        
        # 創建 PyPokerEngine 格式的遊戲狀態
        pye_game_state = {
            'round_count': game_state['round_count'],
            'small_blind_amount': game_state['small_blind_amount'],
            'street': game_state['street'],
            'next_player': game_state['next_player'],
            'table': table,
            'pot': {'main': {'amount': game_state['pot']}},
            'action_histories': {
                'preflop': [],
                'flop': [],
                'turn': [],
                'river': []
            }
        }
        
        # 設置牌組
        deck = Deck()
        deck.deck = [Card.from_str(c) for c in game_state.get('deck', [])]
        pye_game_state['deck'] = deck
        
        return pye_game_state

    def convert_from_pye_game_state(self, pye_game_state):
        if isinstance(pye_game_state, tuple):
            pye_game_state = pye_game_state[0]
        
        print("Debug - pye_game_state keys:", pye_game_state.keys())
        
        game_state = {
            'round_count': pye_game_state.get('round_count', 0),
            'small_blind_amount': pye_game_state.get('small_blind_amount', 0),
            'street': pye_game_state.get('street', 'preflop'),
            'next_player': pye_game_state.get('next_player', 0),
        }

        if 'table' in pye_game_state:
            table = pye_game_state['table']
            game_state['community_card'] = [str(card) for card in table.get_community_card()]
            game_state['dealer_btn'] = table.dealer_btn
            game_state['sb_pos'] = getattr(table, 'sb_pos', 0)
            game_state['bb_pos'] = getattr(table, 'bb_pos', 1)
            
            game_state['seats'] = []
            for player in table.seats.players:
                # 檢查玩家是否已經棄牌
                is_folded = self.check_if_player_folded(pye_game_state, player.uuid)
                
                seat = {
                    'uuid': player.uuid,
                    'stack': player.stack,
                    'name': player.name,
                    'hole_card': [str(card) for card in player.hole_card],
                    'state': 'folded' if is_folded else 'participating'
                }
                game_state['seats'].append(seat)
        else:
            print("Debug - 'table' not in pye_game_state")
            game_state['community_card'] = []
            game_state['seats'] = []

        if 'pot' in pye_game_state:
            if isinstance(pye_game_state['pot'], dict) and 'main' in pye_game_state['pot']:
                game_state['pot'] = pye_game_state['pot']['main'].get('amount', 0)
            else:
                game_state['pot'] = sum(player.stack for player in pye_game_state['table'].seats.players)
        else:
            print("Debug - 'pot' not in pye_game_state")
            game_state['pot'] = 0

        if 'deck' in pye_game_state:
            game_state['deck'] = [str(card) for card in pye_game_state['deck'].deck]
        else:
            print("Debug - 'deck' not in pye_game_state")
            game_state['deck'] = []

        return game_state

    def check_if_player_folded(self, pye_game_state, player_uuid):
        # 檢查動作歷史
        if 'action_histories' in pye_game_state:
            for street, actions in pye_game_state['action_histories'].items():
                for action in actions:
                    if action['uuid'] == player_uuid and action['action'] == 'fold':
                        return True
        
        # 檢查當前街段的行動序列
        if 'next_player' in pye_game_state and 'table' in pye_game_state:
            current_players = pye_game_state['table'].seats.players
            if player_uuid not in [p.uuid for p in current_players]:
                return True
        
        return False

    def prepare_policy_input(self, game_state, current_player_hand_strength, public_strength):
        print("preparing policy input...")
        # 準備其他輸入
        round_num = game_state['round_count']
        my_position = game_state['next_player']
        active_players = sum(1 for player in game_state['table'].seats.players if player.is_active())
        player_status = [1 if player.is_active() else 0 for player in game_state['table'].seats.players]
        player_bets = [player.stack for player in game_state['table'].seats.players]
        min_bet = game_state['small_blind_amount']
        max_bet = max(player.stack for player in game_state['table'].seats.players)

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
        for player in game_state['table'].seats.players:
            hole_cards = player.hole_card
            if hole_cards:  # 如果有已知的手牌
                community_cards = game_state['table'].get_community_card()
                known_cards = self.convert_cards_to_phe_format(hole_cards + community_cards)
                strength_matrix = self.phe.monte_carlo_simulation(known_cards)
            else:  # 如果手牌未知
                strength_matrix = np.zeros((13, 9))  # 假設一個空的強度矩陣
            hand_strengths.append(strength_matrix)
        return hand_strengths

    def calculate_public_strength(self, game_state):
        print("calculating public strength...")
        community_cards = game_state['table'].get_community_card()
        phe_format_cards = self.convert_cards_to_phe_format(community_cards)
        return self.phe.monte_carlo_simulation(phe_format_cards)

    def convert_cards_to_phe_format(self, cards):
        rank_map = {'A': '14', 'K': '13', 'Q': '12', 'J': '11', 'T': '10'}
        suit_map = {16: 'S', 8: 'H', 4: 'D', 2: 'C'}  # 使用二的次方表示花色
        
        converted_cards = []
        for card in cards:
            print(f"Original card: {card}")  # 調試輸出
            if isinstance(card, Card):
                rank = card.rank
                suit = card.suit
            else:
                # 如果不是 Card 對象，假設它是字符串
                if len(card) == 2:
                    suit, rank = card[0], card[1]
                elif len(card) == 3:
                    suit, rank = card[0], card[1:]
                else:
                    raise ValueError(f"Invalid card format: {card}")
            
            # 處理 rank
            if isinstance(rank, str):
                new_rank = rank_map.get(rank.upper(), rank)
            else:
                new_rank = str(rank)
            
            # 處理 suit
            if isinstance(suit, int):
                new_suit = suit_map.get(suit, str(suit))
            else:
                new_suit = suit.upper()
            
            converted_card = f"{new_rank}{new_suit}"
            converted_cards.append(converted_card)
        
        print("converted cards:", converted_cards)  # 調試輸出
        return converted_cards

    def convert_game_state_cards(self, game_state):
        new_game_state = game_state.copy()
        
        # 轉換公共牌
        if 'table' in new_game_state:
            community_card = new_game_state['table'].get_community_card()
            new_game_state['community_card'] = self.convert_cards_to_phe_format(community_card)
        else:
            new_game_state['community_card'] = []
        
        # 轉換玩家手牌
        if 'table' in new_game_state and hasattr(new_game_state['table'], 'seats'):
            for player in new_game_state['table'].seats.players:
                if player.hole_card:
                    player.hole_card = self.convert_cards_to_phe_format(player.hole_card)
        
        return new_game_state

    def is_round_end(self, game_state):
        print("Debug - game_state type:", type(game_state))
        if isinstance(game_state, dict):
            print("Debug - game_state keys:", game_state.keys())
        else:
            print("Debug - game_state is not a dictionary")
            return False
        
        if 'street' not in game_state:
            print("Debug - 'street' not in game_state")
            return False
        
        if game_state['street'] not in ['preflop', 'flop', 'turn', 'river']:
            print(f"Debug - Unexpected street: {game_state['street']}")
            return False
        
        if 'table' not in game_state:
            print("Debug - 'table' not in game_state")
            return False
        
        # 檢查是否所有玩家都已經行動
        active_players = [player for player in game_state['table'].seats.players if player.is_active()]
        if len(active_players) <= 1:
            return True
        
        # 檢查是否所有玩家的下注金額相等
        bet_amounts = [player.stack for player in active_players]
        return len(set(bet_amounts)) == 1

    def is_game_end(self, game_state):
        if isinstance(game_state, tuple):
            game_state = game_state[0]
        return game_state['street'] == Const.Street.SHOWDOWN

    def calculate_payoff(self, game_state):
        payoffs = [0] * self.num_players
        
        # Ensure the game is at showdown stage
        if game_state['street'] != Const.Street.SHOWDOWN:
            raise ValueError("Game is not at showdown stage")

        # Get all players who have not folded
        active_players = [player for player in game_state['table'].seats.players if player.is_active()]
        
        # Evaluate the hand strength of each player
        for player in active_players:
            player.hand_value = HandEvaluator.eval_hand(player.hole_card, game_state['table'].get_community_card())

        # Sort players by hand strength
        active_players.sort(key=lambda p: p.hand_value, reverse=True)

        # Distribute the main pot
        main_pot = game_state['pot']['main']['amount']
        self._assign_pot_to_winners(active_players, main_pot, payoffs)

        # Distribute side pots
        side_pots = game_state['pot'].get('side', [])
        for side_pot in side_pots:
            eligible_players = [p for p in active_players if p.stack > side_pot['amount']]
            self._assign_pot_to_winners(eligible_players, side_pot['amount'], payoffs)

        return payoffs

    def _assign_pot_to_winners(self, eligible_players, pot_amount, payoffs):
        if not eligible_players:
            return
        
        best_hand_value = eligible_players[0].hand_value
        winners = [p for p in eligible_players if p.hand_value == best_hand_value]
        
        win_amount = pot_amount // len(winners)
        for winner in winners:
            player_idx = next(i for i, p in enumerate(self.game_state['table'].seats.players) if p.uuid == winner.uuid)
            payoffs[player_idx] += win_amount

    def convert_action_type(self, action):
        if action == ActionType.FOLD:
            return 'fold'
        elif action == ActionType.CALL:
            return 'call'
        elif action in [ActionType.RAISE, ActionType.BET, ActionType.ALL_IN]:
            return 'raise'
        else:
            raise ValueError(f"Unknown action type: {action}")

    def get_call_amount(self, game_state):
        current_player = game_state['table'].seats.players[game_state['next_player']]
        max_bet = max(player.stack for player in game_state['table'].seats.players)
        return min(max_bet - current_player.stack, current_player.stack)