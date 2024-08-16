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
    
    def forward(self, hand_strengths, pot):
        # hand_strengths: [batch_size, num_players, num_hand_categories]
        # pot: [batch_size, 1]
        # round: [batch_size, 1]
        
        batch_size = hand_strengths.size(0)
        x = hand_strengths.view(batch_size, -1)  # 展平手牌强度矩阵
        x = torch.cat([x, pot], dim=1)  # 连接底池大小
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        expected_payoffs = self.fc3(x)
        
        return expected_payoffs

    def evaluate(self, hand_strengths, pot):
        with torch.no_grad():
            hand_strengths = torch.tensor(hand_strengths, dtype=torch.float32)
            pot = torch.tensor([[pot]], dtype=torch.float32)
            expected_payoffs = self.forward(hand_strengths, pot)
        return expected_payoffs.squeeze().tolist()

# 使用示例
# evaluator = PokerHandEvaluator()
# value_function = ValueFunction(num_players=6, num_hand_categories=10)
# hand_strengths = [evaluator.evaluate(player_cards) for player_cards in all_player_cards]
# expected_payoffs = value_function.evaluate(hand_strengths, pot_size)