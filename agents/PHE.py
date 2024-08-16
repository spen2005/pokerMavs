import torch
import torch.nn as nn

class CardEmbedding(nn.Module):
    def __init__(self, card_types, embed_dim):
        super(CardEmbedding, self).__init__()
        self.rank_embed = nn.Embedding(13, embed_dim)  # 13 ranks
        self.suit_embed = nn.Embedding(4, embed_dim)   # 4 suits
        self.card_embed = nn.Embedding(52, embed_dim)  # 52 cards total
    
    def forward(self, cards):
        # cards: (batch_size, num_cards)
        ranks = cards // 4
        suits = cards % 4
        
        card_embs = self.card_embed(cards)
        rank_embs = self.rank_embed(ranks)
        suit_embs = self.suit_embed(suits)
        
        embeddings = card_embs + rank_embs + suit_embs
        return embeddings
