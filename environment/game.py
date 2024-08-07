from poker_engine import PokerGame, Action

class GameState:
    def __init__(self, public_cards, player_hands, current_player, pot):
        self.public_cards = public_cards
        self.player_hands = player_hands
        self.current_player = current_player
        self.pot = pot
        self.current_bet = 0  # 当前的最大下注
        self.folded_players = []  # 弃牌的玩家

class Action:
    FOLD = 0
    CALL = 1
    RAISE = 2

class PokerGame:
    def __init__(self):
        self.state = GameState([], [], 0, 0)
        self.min_raise = 0  # 初始化最低加注值

    def deal(self):
        # 发牌逻辑
        pass

    def step(self, action, raise_amount=0):
        current_player = self.state.current_player

        if action == Action.FOLD:
            # 记录当前玩家弃牌
            self.state.folded_players.append(current_player)
        elif action == Action.CALL:
            # 当前玩家跟注到当前最大下注
            call_amount = self.state.current_bet
            self.state.pot += call_amount
        elif action == Action.RAISE:
            # 确保加注金额不超过1/3底池
            max_raise = self.state.pot / 3
            if raise_amount > max_raise:
                raise_amount = max_raise
            if raise_amount > self.state.current_bet:
                # 更新底池和当前最大下注
                self.state.pot += raise_amount
                self.state.current_bet = raise_amount
                self.min_raise = raise_amount - self.state.current_bet

        # 更新当前玩家
        self.state.current_player = (current_player + 1) % 6

    def get_state(self):
        # 返回当前游戏状态
        return self.state
