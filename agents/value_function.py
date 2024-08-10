import torch
import torch.nn as nn
import torch.nn.functional as F
from .card_embedding import CardEmbedding

class ValueFunction(nn.Module):
    def __init__(self, card_types, state_dim, embed_dim):
        super(ValueFunction, self).__init__()
        self.card_embedding = CardEmbedding(card_types, embed_dim)
        self.state_fc = nn.Linear(state_dim, 128)
        self.card_fc = nn.Linear(embed_dim * card_types, 128)
        self.fc1 = nn.Linear(128 + 128, 128)
        self.fc2 = nn.Linear(128, 1)  # Output a single value
    
    def forward(self, cards, state):
        card_embs = self.card_embedding(cards)
        card_embs = card_embs.sum(dim=1)  # Sum across cards
        
        state_embs = F.relu(self.state_fc(state))
        card_embs = F.relu(self.card_fc(card_embs))
        
        x = torch.cat([state_embs, card_embs], dim=-1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

    def evaluate(self, cards, state):
        with torch.no_grad():
            cards = torch.tensor(cards, dtype=torch.long)
            state = torch.tensor(state, dtype=torch.float32)
            value = self.forward(cards, state)
        return value.item()
