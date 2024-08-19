import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_players=6):
        super(PolicyNetwork, self).__init__()
        
        self.num_players = num_players
        
        # Calculate input dimension
        hand_strength_dim = 13 * 9  # 13x9 matrix flattened
        public_strength_dim = 13 * 9  # 13x9 matrix flattened
        other_inputs = 5 + 2 * num_players
        input_dim = hand_strength_dim + public_strength_dim + other_inputs
        
        # Set action dimension to 7
        action_dim = 7
        
        # Input layer
        self.input_fc = nn.Linear(input_dim, 256)
        
        # Hidden layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer
        self.output_fc = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.input_fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.output_fc(x), dim=-1)
        return action_probs