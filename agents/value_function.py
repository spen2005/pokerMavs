import torch
import torch.nn as nn
import torch.nn.functional as F
from .PHE import PokerHandEvaluator

class ValueFunction(nn.Module):
    def __init__(self, num_players, num_hand_categories):
        super(ValueFunction, self).__init__()
        self.num_players = num_players
        self.num_hand_categories = num_hand_categories
        
        # 输入层：每个玩家的手牌强度矩阵 + 底池大小 + 輪數
        input_dim = num_players * num_hand_categories + 1 + 1
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_players)  # 输出每个玩家的期望收益
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, hand_strengths, pot, round):
        # Ensure all inputs are on the same device
        hand_strengths = hand_strengths.to(self.device)
        pot = pot.to(self.device)
        round = round.to(self.device)
        
        batch_size = hand_strengths.size(0)
        x = hand_strengths.view(batch_size, -1)  # 展平手牌强度矩阵
        x = torch.cat([x, pot, round], dim=1)  # 连接底池大小和轮数
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        expected_payoffs = self.fc3(x)
        
        return expected_payoffs

    def evaluate(self, hand_strengths, pot, round):
        with torch.no_grad():
            hand_strengths = hand_strengths.to(self.device)
            pot = pot.to(self.device)
            round = round.to(self.device)
            expected_payoffs = self.forward(hand_strengths, pot, round)
        return expected_payoffs.cpu().squeeze().tolist()

# 使用示例
# evaluator = PokerHandEvaluator()
# value_function = ValueFunction(num_players=6, num_hand_categories=10)
# hand_strengths = [evaluator.evaluate(player_cards) for player_cards in all_player_cards]
# expected_payoffs = value_function.evaluate(hand_strengths, pot_size)