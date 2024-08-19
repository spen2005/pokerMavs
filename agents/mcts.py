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
from environment.actions import ActionType, BettingStage, PREFLOP_ACTIONS, POSTFLOP_ACTIONS, get_action_space

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

    def act(self, game_state, hand_strengths, public_strength, epsilon=0.1):
        '''
        print(f"Current street: {game_state['street']}")
        print(f"Current player: {game_state['next_player']}")
        print(f"Current bet: {self.get_current_bet(game_state)}")
        print(f"player's bet in this street: {self.get_player_bet(game_state, game_state['table'].seats.players[game_state['next_player']])}")
        print(f"Total pot: {self.get_total_pot(game_state)}")
'''

        betting_stage = BettingStage.PREFLOP if game_state['street'] == Const.Street.PREFLOP else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)

        # 獲取當前玩家
        current_player = game_state['table'].seats.players[game_state['next_player']]
        
        # 獲取當前玩家的手牌強度
        current_player_index = game_state['next_player']
        current_player_hand_strengths = hand_strengths[current_player_index]
        
        # 獲取策略網絡的輸出
        policy_input = self.prepare_policy_input(game_state, current_player_hand_strengths, public_strength)
        action_probs = self.policy_network(policy_input).detach().numpy().squeeze()

        # 獲取有效動作
        valid_actions = self.get_valid_actions(game_state, action_space)

        # 過濾無效動作的概率
        valid_probs = []
        valid_actions_expanded = []
        for action, amount in valid_actions:
            action_index = self.get_action_index(action_space, action, amount, game_state['small_blind_amount']*2, self.get_total_pot(game_state))
            valid_probs.append(action_probs[action_index])
            valid_actions_expanded.append((action, amount))

        # 重新歸一化概率
        valid_probs = np.array(valid_probs)
        valid_probs += epsilon # epsilon greedy
        valid_probs /= valid_probs.sum()

        # 選擇動作
        chosen_action_index = np.random.choice(len(valid_probs), p=valid_probs)
        chosen_action, chosen_amount = valid_actions_expanded[chosen_action_index]

        # print(f"chosen action: {chosen_action}, amount: {chosen_amount}")
        return {'action': self.convert_action_type(chosen_action), 'amount': chosen_amount}
        
    def mcts_strategy(self, game_state, num_samples=1000):
        # print player
        print(f"player: {game_state['table'].seats.players[game_state['next_player']].name}")
        print("computing mcts strategy...")
        '''
        print("Game state structure:")
        
        for key, value in game_state.items():
            print(f"{key}: {type(value)}")
        
        if 'table' in game_state:
            print("Table structure:")
            table = game_state['table']
            for key, value in table.__dict__.items():
                print(f"{key}: {type(value)}")
            
            print("Player structure:")
            for player in table.seats.players:
                print(f"Player {player.uuid}:")
                for key, value in player.__dict__.items():
                    print(f"  {key}: {type(value)}")
                print(f"  round_action_histories: {player.round_action_histories}")
        '''
        if self.is_game_end(game_state):
            return self.calculate_payoff(game_state), None, None

        street_map = {Const.Street.PREFLOP: BettingStage.PREFLOP, 
                      Const.Street.FLOP: BettingStage.POSTFLOP, 
                      Const.Street.TURN: BettingStage.POSTFLOP, 
                      Const.Street.RIVER: BettingStage.POSTFLOP}
        betting_stage = street_map.get(game_state['street'], BettingStage.PREFLOP)
        action_space = get_action_space(betting_stage)
        num_players = len(game_state['table'].seats.players)

        expected_values = np.zeros((7, num_players))
        action_count = np.zeros(7)

        # calculate hand strengths for each player
        hand_strengths = self.calculate_hand_strengths(game_state)
        # calculate public strength
        public_strength = self.calculate_public_strength(game_state)
        # get the current stack of each player
        current_stacks = [player.stack for player in game_state['table'].seats.players]
        # get the pot size
        pot = self.get_total_pot(game_state)

        for i in range(num_samples):
            # print(f"=== sample {i+1} ===")
            simulated_state = game_state.copy()
            street = simulated_state['street']
            # sample to the next street and evaluate the payoff using the value function
            # print(f"now street: {simulated_state['street']}")
            first_action = self.act(simulated_state, hand_strengths, public_strength, 0.1)
            simulated_state, _ = self.emulator.apply_action(simulated_state, first_action['action'], first_action['amount'])
            
            while street == simulated_state['street']:
                # print now street
                # print(f"now street: {simulated_state['street']}")
                action = self.act(simulated_state, hand_strengths, public_strength)
                simulated_state, _ = self.emulator.apply_action(simulated_state, action['action'], action['amount'])
                
            # 如果回合結束但遊戲沒有結束，進入下一個街道
            if not self.is_game_end(simulated_state):
                # if game not ended, payoffs = the final stack - current stack + value function
                #print("game continued")
                # print now street
                #print(f"now street: {simulated_state['street']}")
                simulated_state = self.move_to_next_street(simulated_state)
            
                value_input = self.prepare_value_input(simulated_state, hand_strengths)
                payoffs = self.value_function.evaluate(*value_input)
            else:
                # if game ended, payoffs = the final stack - current stack
                # print("game ended")
                # get the final stack of each player
                final_stacks = [player.stack for player in simulated_state['table'].seats.players]
                payoffs = [final_stacks[i] - current_stacks[i] for i in range(len(current_stacks))]

            action_index = self.get_action_index(action_space, first_action['action'], first_action['amount'], game_state['small_blind_amount']*2, pot)
            expected_values[action_index] += np.array(payoffs)
            action_count[action_index] += 1
        
        for index in range(7):
            if action_count[index] > 0:
                expected_values[index] /= action_count[index]

        current_player = game_state['next_player']
        current_player_hand_strength = hand_strengths[current_player]
        
        value_input = self.prepare_value_input(game_state, hand_strengths)
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

        #  print avg_expected_values, old_policy, new_policy
        np.set_printoptions(precision=6, suppress=True)

        # print(f"expected_values:\n {expected_values}")
        print(f"current_expected_value:\n{current_player_expected_values}")
        print(f"old_policy:\n {old_policy}")
        print(f"avg_expected_values:\n {avg_expected_value}")
        # print regrets 取到小數點第二位
        print(f"regrets:\n {regrets}")
        print(f"new_policy:\n {new_policy}")

        # based on new policy, choose an action
        # first get valid actions, similar to act()
        valid_actions = self.get_valid_actions(game_state, action_space)

        # print valid_actions
        # print(f"valid_actions: {valid_actions}")

        valid_probs = []
        valid_actions_expanded = []
        for action, amount in valid_actions:
            action_index = self.get_action_index(action_space, action, amount, game_state['small_blind_amount']*2, pot)
            valid_probs.append(new_policy[action_index])
            valid_actions_expanded.append((action, amount))
        
        valid_probs = np.array(valid_probs)
        
        valid_probs += 0.1
        valid_probs[-1] = 0 # decrease probability of all_in
        valid_probs[0] = 0 # decrease probability of fold
        valid_probs /= valid_probs.sum()

        print(f"valid_probs: {valid_probs}")

        action_index = np.random.choice(len(valid_probs), p=valid_probs)
        action, amount = valid_actions_expanded[action_index]

        # action, amount = self.get_action_from_type(action_index, game_state)
        print(f"action: {action}, amount: {amount}")

        return avg_expected_values, policy_input, value_input, new_policy, self.convert_action_type(action), amount
    
    def get_action_index(self, action_space, action_type, amount, blind_amount, pot):
        # print(f"Debug: amount = {amount}, action_type = {action_type}")
        # if it is <enum 'ActionType'> convert it to string
        
        # 使用字符串比较来确保兼容性
        action_type_str = str(action_type).split('.')[-1] if hasattr(action_type, 'name') else str(action_type)

        # print(action_type_str)

        # if fold, return 0
        if action_type_str == "fold" or action_type_str == "FOLD":
            return 0
        # if call, return 1 (regardless of the amount)
        elif action_type_str == "call" or action_type_str == "CALL":
            return 1
        # if raise 2.5bb or bet 0.33 pot, return 2
        elif (action_type_str == "raise" or action_type_str == "RAISE") and amount == 2.5 * blind_amount:
            return 2
        elif (action_type_str == "bet" or action_type_str == "BET") and abs(amount - 0.33 * pot) < 10:
            return 2
        # if raise 5bb or bet 0.66 pot, return 3
        elif (action_type_str == "raise" or action_type_str == "RAISE") and amount == 5 * blind_amount:
            return 3
        elif (action_type_str == "bet" or action_type_str == "BET") and abs(amount - 0.66 * pot) < 10:
            return 3
        # if raise 10bb or bet 1 pot, return 4
        elif (action_type_str == "raise" or action_type_str == "RAISE") and amount == 10 * blind_amount:
            return 4
        elif (action_type_str == "bet" or action_type_str == "BET") and abs(amount - pot) < 10:
            return 4
        # if raise 25bb or bet 2 pot, return 5
        elif (action_type_str == "raise" or action_type_str == "RAISE") and amount == 25 * blind_amount:
            return 5
        elif (action_type_str == "bet" or action_type_str == "BET") and  abs(amount - 2 * pot) < 10:
            return 5
        # if raise more than 25bb, return 6
        elif (action_type_str == "raise" or action_type_str == "RAISE") and amount > 25 * blind_amount:
            return 6
        # if all_in, return 6
        elif (action_type_str == "all_in" or action_type_str == "ALL_IN"):
            return 6
        else:
            raise ValueError(f"Invalid action: {action_type}, {amount}")
    
    def get_action_from_type(self, action_index, game_state):
        # if index is 0, return fold
        if action_index == 0:
            return {'action': ActionType.FOLD, 'amount': 0}
        # if index is 1, return call
        elif action_index == 1:
            return {'action': ActionType.CALL, 'amount': 0}
        # if index is 2 and game_state['street'] == Const.Street.PREFLOP, return raise 2.5bb
        elif action_index == 2 and game_state['street'] == Const.Street.PREFLOP:
            return {'action': ActionType.RAISE, 'amount': 2.5 * game_state['small_blind_amount'] * 2}
        # if index is 2 and game_state['street'] == Const.Street.POSTFLOP, return bet 0.33 pot
        elif action_index == 2 and game_state['street'] == Const.Street.POSTFLOP:
            return {'action': ActionType.BET, 'amount': 0.33 * game_state['pot']}
        # if index is 3 and game_state['street'] == Const.Street.PREFLOP, return raise 5bb
        elif action_index == 3 and game_state['street'] == Const.Street.PREFLOP:
            return {'action': ActionType.RAISE, 'amount': 5 * game_state['small_blind_amount'] * 2}
        # if index is 3 and game_state['street'] == Const.Street.POSTFLOP, return bet 0.66 pot
        elif action_index == 3 and game_state['street'] == Const.Street.POSTFLOP:
            return {'action': ActionType.BET, 'amount': 0.66 * game_state['pot']}
        # if index is 4 and game_state['street'] == Const.Street.PREFLOP, return raise 10bb
        elif action_index == 4 and game_state['street'] == Const.Street.PREFLOP:
            return {'action': ActionType.RAISE, 'amount': 10 * game_state['small_blind_amount'] * 2}
        # if index is 4 and game_state['street'] == Const.Street.POSTFLOP, return bet 1 pot
        elif action_index == 4 and game_state['street'] == Const.Street.POSTFLOP:
            return {'action': ActionType.BET, 'amount': 1 * game_state['pot']}
        # if index is 5 and game_state['street'] == Const.Street.PREFLOP, return raise 25bb
        elif action_index == 5 and game_state['street'] == Const.Street.PREFLOP:
            return {'action': ActionType.RAISE, 'amount': 25 * game_state['small_blind_amount'] * 2}
        # if index is 5 and game_state['street'] == Const.Street.POSTFLOP, return bet 2 pot
        elif action_index == 5 and game_state['street'] == Const.Street.POSTFLOP:
            return {'action': ActionType.BET, 'amount': 2 * game_state['pot']}
        elif action_index == 6:
            return {'action': ActionType.ALL_IN, 'amount': game_state['table'].seats.players[game_state['next_player']].stack}
        else:
            raise ValueError(f"Invalid action index: {action_index}")

    
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
        
        # print("Debug - pye_game_state keys:", pye_game_state.keys())
        
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
            # print("Debug - 'table' not in pye_game_state")
            game_state['community_card'] = []
            game_state['seats'] = []

        if 'pot' in pye_game_state:
            if isinstance(pye_game_state['pot'], dict) and 'main' in pye_game_state['pot']:
                game_state['pot'] = pye_game_state['pot']['main'].get('amount', 0)
            else:
                game_state['pot'] = sum(player.stack for player in pye_game_state['table'].seats.players)
        else:
            # print("Debug - 'pot' not in pye_game_state")
            game_state['pot'] = 0

        if 'deck' in pye_game_state:
            game_state['deck'] = [str(card) for card in pye_game_state['deck'].deck]
        else:
            # print("Debug - 'deck' not in pye_game_state")
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
        # ("preparing policy input...")
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
    
    def prepare_value_input(self, game_state, hand_strengths):
        # print("preparing value input...")
        pot = self.get_total_pot(game_state)  # Use the existing method to get the total pot
        round = game_state['round_count']
        # 將手牌強度轉換為適合 ValueFunction 的格式
        hand_strengths_array = np.array([hs.flatten() for hs in hand_strengths])
        hand_strengths_tensor = torch.tensor(hand_strengths_array, dtype=torch.float32).unsqueeze(0)
        pot_tensor = torch.tensor([[pot]], dtype=torch.float32)
        round_tensor = torch.tensor([[round]], dtype=torch.float32)
        
        return hand_strengths_tensor, pot_tensor, round_tensor

    def calculate_hand_strengths(self, game_state):
        # print("calculating hand strengths...")
        hand_strengths = []
        for player in game_state['table'].seats.players:
            if hasattr(player, 'mcts_hole_card'):
                hole_cards = player.mcts_hole_card
                community_cards = game_state.get('mcts_community_card', [])
                known_cards = hole_cards + community_cards
                strength_matrix = self.phe.monte_carlo_simulation(known_cards)
            else:
                strength_matrix = np.zeros((13, 9))  # 假設一個空的強度矩陣
            hand_strengths.append(strength_matrix)
        
        # print(f"Hand strengths calculated for {len(hand_strengths)} players")
        return hand_strengths

    def calculate_public_strength(self, game_state):
        # print("calculating public strength...")
        community_cards = game_state.get('mcts_community_card', [])
        return self.phe.monte_carlo_simulation(community_cards)

    def convert_cards_to_phe_format(self, cards):
        rank_map = {'A': '14', 'K': '13', 'Q': '12', 'J': '11', 'T': '10'}
        suit_map = {16: 'S', 8: 'H', 4: 'D', 2: 'C'}  # 使用二的次方表示花色
        
        converted_cards = []
        for card in cards:
            # print(f"Original card: {card}")  # 調試輸出
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
        
        # print("All converted cards:", converted_cards)  # 調試輸出
        return converted_cards

    def convert_game_state_cards(self, game_state):
        new_game_state = game_state.copy()
        
        # 為 MCTS 邏輯創建轉換後的卡牌版本
        if 'table' in new_game_state:
            community_card = new_game_state['table'].get_community_card()
            new_game_state['mcts_community_card'] = self.convert_cards_to_phe_format(community_card)
        else:
            new_game_state['mcts_community_card'] = []
        
        # 為 MCTS 邏輯創建轉換後的玩家手牌版本
        if 'table' in new_game_state and hasattr(new_game_state['table'], 'seats'):
            for player in new_game_state['table'].seats.players:
                if player.hole_card:
                    player.mcts_hole_card = self.convert_cards_to_phe_format(player.hole_card)
        
        return new_game_state

    def is_round_end(self, game_state):
        if game_state['street'] not in [Const.Street.PREFLOP, Const.Street.FLOP, Const.Street.TURN, Const.Street.RIVER]:
            return True

        active_players = [player for player in game_state['table'].seats.players if player.is_active()]
        if len(active_players) <= 1:
            return True

        # 檢查是否所有玩家的下注金額相等，且所有玩家都已行動
        bet_amounts = [self.get_player_bet(game_state, player) for player in active_players]
        all_equal = len(set(bet_amounts)) == 1
        
        # 檢查所有玩家是否都已經行動
        all_acted = True
        for player in active_players:
            if not player.round_action_histories:
                all_acted = False
                break
            current_street_actions = player.round_action_histories[-1]
            if not current_street_actions or current_street_actions[-1]['action'] in ['SMALLBLIND', 'BIGBLIND']:
                all_acted = False
                break

        return all_equal and all_acted

    def is_game_end(self, game_state):
        return game_state['street'] == Const.Street.SHOWDOWN or game_state['street'] == Const.Street.FINISHED

    def calculate_payoff(self, game_state):
        payoffs = [0] * self.num_players
        
        # Check if the game is finished
        if game_state['street'] == Const.Street.FINISHED:
            # Use PyPokerEngine to determine the winner and assign the pot
            emulator = Emulator()
            emulator.set_game_rule(self.num_players, game_state['round_count'], game_state['small_blind_amount'], 0)
            final_state, events = emulator.run_until_game_finish(game_state)
            
            # Extract payoffs from the final state
            for player in final_state['table'].seats.players:
                initial_stack = next(p.initial_stack for p in game_state['table'].seats.players if p.uuid == player.uuid)
                payoffs[game_state['table'].seats.players.index(player)] = player.stack - initial_stack
        
        # Check if the game is at showdown stage
        elif game_state['street'] == Const.Street.SHOWDOWN:
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
        
        else:
            # If the game is not at showdown, find the remaining active player
            active_players = [player for player in game_state['table'].seats.players if player.is_active()]
            if len(active_players) == 1:
                # If only one player is active, they win the entire pot
                winner = active_players[0]
                winner_index = game_state['table'].seats.players.index(winner)
                payoffs[winner_index] = self.get_total_pot(game_state)
            else:
                # Use PyPokerEngine to determine the winner and assign the pot
                emulator = Emulator()
                emulator.set_game_rule(self.num_players, game_state['round_count'], game_state['small_blind_amount'], 0)
                final_state, events = emulator.run_until_game_finish(game_state)
                
                # Extract payoffs from the final state
                for player in final_state['table'].seats.players:
                    initial_stack = next(p.initial_stack for p in game_state['table'].seats.players if p.uuid == player.uuid)
                    payoffs[game_state['table'].seats.players.index(player)] = player.stack - initial_stack

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

    def get_current_bet(self, game_state):
        return max(self.get_player_bet(game_state, player) for player in game_state['table'].seats.players)

    def get_player_bet(self, game_state, player):
        return 1000 - player.stack

    def get_min_raise(self, game_state):
        current_bet = self.get_current_bet(game_state)
        print(f"current_bet: {current_bet}")
        last_raise = game_state['small_blind_amount']  # 如果無法獲取上一次的 raise 金額，我們使用小盲作為最小 raise
        return current_bet + max(last_raise, game_state['small_blind_amount'])

    def move_to_next_street(self, game_state):
        street_order = [Const.Street.PREFLOP, Const.Street.FLOP, Const.Street.TURN, Const.Street.RIVER, Const.Street.SHOWDOWN, Const.Street.FINISHED]
        #print(f"game_state['street']: {game_state['street']}")
        current_street_index = street_order.index(game_state['street'])
        if current_street_index < len(street_order) - 1:
            game_state['street'] = street_order[current_street_index + 1]
        else:
            game_state['street'] = Const.Street.SHOWDOWN
        return game_state
    
    def get_player_bet(self, game_state, player):
        return player.paid_sum()

    def get_total_pot(self, game_state):
        return 6000 - (sum(player.stack for player in game_state['table'].seats.players))

    def get_valid_actions(self, game_state, action_space):
        player = game_state['table'].seats.players[game_state['next_player']]
        current_bet = self.get_current_bet(game_state)
        player_bet = self.get_player_bet(game_state, player)
        to_call = current_bet - player_bet
        remaining_stack = player.stack
        # print(f"to_call: {to_call}")

        # 計算玩家在這條street之前的總投注
        previous_streets_bet = player.paid_sum() - player_bet
        '''
        print(f"Player {player.name} previous streets bet: {previous_streets_bet}")
        print(f"Player {player.name} current street bet: {player_bet}")
        print(f"Player {player.name} remaining stack: {remaining_stack}")
        print(f"Current bet to call: {to_call}")
        '''
        valid_actions = []

        for action, values in action_space.items():
            if action == ActionType.FOLD:
                valid_actions.append((action, 0))
            elif action == ActionType.CALL:
                if remaining_stack >= to_call:
                    valid_actions.append((action, to_call+player_bet))
            elif action == ActionType.RAISE:
                for raise_multiplier in values:
                    raise_amount = raise_multiplier * game_state['small_blind_amount'] * 2
                    if raise_amount > current_bet and raise_amount <= remaining_stack + player_bet:
                        valid_actions.append((action, raise_amount))
            elif action == ActionType.BET:
                for bet_multiplier in values:
                    bet_amount = int(bet_multiplier * self.get_total_pot(game_state))
                    if bet_amount > current_bet and bet_amount <= remaining_stack + player_bet:
                        valid_actions.append((action, bet_amount))
            elif action == ActionType.ALL_IN:
                if remaining_stack > 0:
                    valid_actions.append((action, player.paid_sum()+remaining_stack))

        #print(f"Valid actions for player {player.name}: {valid_actions}")
        return valid_actions

    def get_action_amount(self, action, game_state):
        #print("getting action amount...")
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