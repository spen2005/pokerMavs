import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, num_players=6, hand_strength_dim=117, public_strength_dim=5):
        super(PolicyNetwork, self).__init__()
        
        self.num_players = num_players
        
        # 计算输入维度
        input_dim = hand_strength_dim + public_strength_dim + 5 + 2 * num_players
        
        # 输入层
        self.input_fc = nn.Linear(input_dim, 128)
        
        # 隐藏层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # 输出层
        self.output_fc = nn.Linear(32, action_dim)
    
    def forward(self, game_state):
        round_num, hand_strength, public_strength, my_position, active_players, \
        player_status, player_bets, min_bet, max_bet = game_state
        
        # 将所有输入连接成一个向量
        x = torch.cat([
            round_num.unsqueeze(-1),
            hand_strength.flatten(),
            public_strength,
            my_position.unsqueeze(-1),
            active_players.unsqueeze(-1),
            player_status,
            player_bets,
            min_bet.unsqueeze(-1),
            max_bet.unsqueeze(-1)
        ], dim=-1)
        
        # 通过网络层
        x = F.relu(self.input_fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.output_fc(x), dim=-1)
        
        return action_probs
