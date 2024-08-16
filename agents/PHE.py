import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

'''
def is_cuda_available():
    try:
        import cupy
        import cupy.cuda.runtime as cupy_runtime
        # Check if a GPU is available and accessible
        device_count = cupy_runtime.getDeviceCount()
        return device_count > 0
    except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
        return False

# Determine if CUDA is available
if is_cuda_available():
    import cupy as cp
    use_cuda = True
else:
    import numpy as cp
    use_cuda = False
'''
use_cuda = False

class PokerHandEvaluator:
    def __init__(self):
        self.hand_types = [
            "Straight flush", "Four of a kind", "Full house", "Flush", 
            "Straight", "Three of a kind", "Two pair", "Pair", "High card"
        ]

    def suit_count(self, community_cards):
        suits = [card[-1] for card in community_cards]
        return Counter(suits)

    def rank_count(self, community_cards):
        ranks = [int(card[:-1]) for card in community_cards]
        return Counter(ranks)

    def type_of_hand_kicker(self, cards):
        # Convert card format if necessary
        cards = [self.convert_card_format(card) for card in cards]
        
        # the input would be 7 cards, texas poker has 5 cards, so we need to find the best 5 cards
        # 1. determine whether it is straight flush
        for i in range(14, 4, -1):
            needed_cards = [f'{14 if rank == 1 else rank}{cards[0][-1]}' for rank in range(i-4, i+1)]
            if all(card in cards for card in needed_cards):
                matrix = np.zeros((13, 9))
                matrix[14-i, 0] = 1
                return matrix
        # 2. determine whether it is four of a kind
        # if there is a rank with 4 cards, then it is four of a kind
        rank_counts = Counter([int(card[:-1]) for card in cards])
        for i in range(14, 1, -1):
            if rank_counts.get(i, 0) == 4:
                matrix = np.zeros((13, 9))
                matrix[14-i, 1] = 1
                return matrix
        # 3. determine whether it is full house
        # if there is a rank with 3 cards and another rank with 2 cards, then it is full house
        for i in range(14, 1, -1):
            if rank_counts.get(i, 0) == 3:
                for j in range(14, 1, -1):
                    if rank_counts.get(j, 0) == 2:
                        matrix = np.zeros((13, 9))
                        matrix[14-i, 2] = 1
                        return matrix
        # 4. determine whether it is flush
        # if there is a suit with 5 cards, then it is flush
        suit_counts = self.suit_count(cards)
        for suit in ['H', 'D', 'S', 'C']:
            if suit_counts.get(suit, 0) >= 5:
                # get the max rank of the cards of the determined suit
                suited_cards = [int(card[:-1]) for card in cards if card[-1] == suit]
                suited_cards.sort(reverse=True)
                max_rank = suited_cards[0]
                matrix = np.zeros((13, 9))
                matrix[14-max_rank, 3] = 1
                return matrix
            
        # 5. determine whether it is straight
        # if there is a straight, then it is straight
        ranks = [int(card[:-1]) for card in cards]
        ranks = list(set(ranks))  # Remove duplicates
        if 14 in ranks:
            ranks.append(1)
        for i in range(14, 4, -1):
            needed_cards = list(range(i-4, i+1))
            if all(rank in ranks for rank in needed_cards):
                matrix = np.zeros((13, 9))
                matrix[14-i, 4] = 1
                return matrix
        # 6. determine whether it is three of a kind
        # if there is a rank with 3 cards, then it is three of a kind(no need to consider full house because it will be returned in the full house)
        for i in range(14, 1, -1):
            if rank_counts.get(i, 0) == 3:
                matrix = np.zeros((13, 9))
                matrix[14-i, 5] = 1
                return matrix
        # 7. determine whether it is two pair
        # if there are two ranks with 2 cards, then it is two pair
        pairs = [rank for rank in range(14, 1, -1) if rank_counts.get(rank, 0) == 2]
        if len(pairs) == 2:
            matrix = np.zeros((13, 9))
            matrix[14-pairs[0], 6] = 1
            return matrix
        # 8. determine whether it is pair
        # if there is a rank with 2 cards, then it is pair
        for i in range(14, 1, -1):
            if rank_counts.get(i, 0) == 2:
                matrix = np.zeros((13, 9))
                matrix[14-i, 7] = 1
                return matrix
        # 9. determine whether it is high card
        # if there is no pair, then it is high card, then find the one with the max rank
        max_rank = max(ranks)
        matrix = np.zeros((13, 9))
        matrix[14-max_rank, 8] = 1
        return matrix

    def convert_card_format(self, card):
        rank = card[:-1]
        suit = card[-1]
        return f"{rank.zfill(2)}{suit}"  # Ensure rank is always two digits for consistency

    def monte_carlo_simulation(self, known_cards, num_samples=100000):
        all_cards = [f"{rank}{suit}" for rank in range(2, 15) for suit in ['H', 'D', 'S', 'C']]
        remaining_cards = list(set(all_cards) - set(known_cards))

        total_prob_matrix = cp.zeros((13, 9)) if use_cuda else np.zeros((13, 9))
        # if known_cards have 3 cards
        if len(known_cards) == 0:
            for _ in range(num_samples):
                sample_cards = np.random.choice(remaining_cards, 7, replace=False)
                prob_matrix = self.type_of_hand_kicker(sample_cards.tolist())
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 1:
            for _ in range(num_samples):
                sample_cards = np.random.choice(remaining_cards, 6, replace=False)
                full_hand = known_cards + sample_cards.tolist()
                prob_matrix = self.type_of_hand_kicker(full_hand)
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 2:
            for _ in range(num_samples):
                sample_cards = np.random.choice(remaining_cards, 5, replace=False)
                full_hand = known_cards + sample_cards.tolist()
                prob_matrix = self.type_of_hand_kicker(full_hand)
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 3:
            for _ in range(num_samples):
                sample_cards = np.random.choice(remaining_cards, 4, replace=False)
                full_hand = known_cards + sample_cards.tolist()
                prob_matrix = self.type_of_hand_kicker(full_hand)
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 4:
            for _ in range(num_samples):
                sample_cards = np.random.choice(remaining_cards, 3, replace=False)
                full_hand = known_cards + sample_cards.tolist()
                prob_matrix = self.type_of_hand_kicker(full_hand)
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 5:
            for _ in range(num_samples):
                sample_cards = np.random.choice(remaining_cards, 2, replace=False)
                full_hand = known_cards + sample_cards.tolist()
                prob_matrix = self.type_of_hand_kicker(full_hand)
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 6:
            for _ in range(num_samples):
                sample_card = np.random.choice(remaining_cards, 1, replace=False)
                full_hand = known_cards + sample_card.tolist()
                prob_matrix = self.type_of_hand_kicker(full_hand)
                total_prob_matrix += prob_matrix
        elif len(known_cards) == 7:
            for _ in range(num_samples):
                prob_matrix = self.type_of_hand_kicker(known_cards)
                total_prob_matrix += prob_matrix
        else:
            print("Please input 0 1 2 3 4 5 6 7 cards")
        avg_prob_matrix = total_prob_matrix / num_samples
        return avg_prob_matrix
    def brutal_search(self, known_cards):
        total_prob_matrix = cp.zeros((13, 9)) if use_cuda else np.zeros((13, 9))

        # if known_cards have 0 cards
        if len(known_cards) == 0:
            print("Complexity too high to use brute force, using Monte Carlo simulation instead")
            return self.monte_carlo_simulation(known_cards)
        elif len(known_cards) == 1:
            print("Complexity too high to use brute force, using Monte Carlo simulation instead")
            return self.monte_carlo_simulation(known_cards)
        elif len(known_cards) == 2:
            all_cards = [f"{rank}{suit}" for rank in range(2, 15) for suit in ['H', 'D', 'S', 'C']]
            remaining_cards = list(set(all_cards) - set(known_cards))
            len_remaining_cards = len(remaining_cards)
            for i in range(len_remaining_cards):
                for j in range(i+1, len_remaining_cards):
                    for k in range(j+1, len_remaining_cards):
                        for l in range(k+1, len_remaining_cards):
                            for m in range(l+1, len_remaining_cards):
                                # evaluate the probability using type_of_hand_kicker
                                prob_matrix = self.type_of_hand_kicker(known_cards + [remaining_cards[i], remaining_cards[j], remaining_cards[k], remaining_cards[l], remaining_cards[m]])
                                total_prob_matrix += prob_matrix
                print(i)
            combinations = len_remaining_cards * (len_remaining_cards - 1) * (len_remaining_cards - 2) * (len_remaining_cards - 3) * (len_remaining_cards - 4)/(5*4*3*2)
            avg_prob_matrix = total_prob_matrix/combinations
        # if known_cards have 3 cards
        elif len(known_cards) == 3:
            all_cards = [f"{rank}{suit}" for rank in range(2, 15) for suit in ['H', 'D', 'S', 'C']]
            remaining_cards = list(set(all_cards) - set(known_cards))
            len_remaining_cards = len(remaining_cards)
            for i in range(len_remaining_cards):
                for j in range(i+1, len_remaining_cards):
                    for k in range(j+1, len_remaining_cards):
                        for l in range(k+1, len_remaining_cards):
                            # evaluate the probability using type_of_hand_kicker
                            prob_matrix = self.type_of_hand_kicker(known_cards + [remaining_cards[i], remaining_cards[j], remaining_cards[k], remaining_cards[l]])
                            total_prob_matrix += prob_matrix
                print(i)
            combinations = len_remaining_cards * (len_remaining_cards - 1) * (len_remaining_cards - 2) * (len_remaining_cards - 3)/(4*3*2)
            avg_prob_matrix = total_prob_matrix/combinations
        elif len(known_cards) == 4:
            all_cards = [f"{rank}{suit}" for rank in range(2, 15) for suit in ['H', 'D', 'S', 'C']]
            remaining_cards = list(set(all_cards) - set(known_cards))
            len_remaining_cards = len(remaining_cards)
            for i in range(len_remaining_cards):
                for j in range(i+1, len_remaining_cards):
                    for k in range(j+1, len_remaining_cards):
                        # evaluate the probability using type_of_hand_kicker
                        prob_matrix = self.type_of_hand_kicker(known_cards + [remaining_cards[i], remaining_cards[j], remaining_cards[k]])
                        total_prob_matrix += prob_matrix
                print(i)
            combinations = len_remaining_cards * (len_remaining_cards - 1) * (len_remaining_cards - 2)/(3*2)
            avg_prob_matrix = total_prob_matrix/combinations
        elif len(known_cards) == 5:
            all_cards = [f"{rank}{suit}" for rank in range(2, 15) for suit in ['H', 'D', 'S', 'C']]
            remaining_cards = list(set(all_cards) - set(known_cards))
            len_remaining_cards = len(remaining_cards)
            for i in range(len_remaining_cards):
                for j in range(i+1, len_remaining_cards):
                    # evaluate the probability using type_of_hand_kicker
                    prob_matrix = self.type_of_hand_kicker(known_cards + [remaining_cards[i], remaining_cards[j]])
                    total_prob_matrix += prob_matrix
                print(i)
            combinations = len_remaining_cards * (len_remaining_cards - 1)/(2)
            avg_prob_matrix = total_prob_matrix/combinations
        elif len(known_cards) == 6:
            all_cards = [f"{rank}{suit}" for rank in range(2, 15) for suit in ['H', 'D', 'S', 'C']]
            remaining_cards = list(set(all_cards) - set(known_cards))
            len_remaining_cards = len(remaining_cards)
            for i in range(len_remaining_cards):
                # evaluate the probability using type_of_hand_kicker
                prob_matrix = self.type_of_hand_kicker(known_cards + [remaining_cards[i]])
                total_prob_matrix += prob_matrix
                print(i)
            avg_prob_matrix = total_prob_matrix/len_remaining_cards
        elif len(known_cards) == 7:
            avg_prob_matrix = self.type_of_hand_kicker(known_cards)
        else:
            print("Please input 0 1 2 3 4 5 6 7 cards")
        return avg_prob_matrix

    def plot_heatmap_with_sum(self, probabilities, input_cards):
        # Calculate sum of probabilities for each hand type
        summed_probabilities = np.sum(probabilities, axis=0)
        
        # Add the summed probabilities as a new row
        extended_probabilities = np.vstack((probabilities, summed_probabilities))
        
        plt.figure(figsize=(12, 10))
        heatmap = plt.imshow(extended_probabilities, cmap='viridis', aspect='auto')
        
        # Adding color bar
        plt.colorbar(label='Probability')
        
        # Adding text annotations to the heatmap
        for i in range(extended_probabilities.shape[0]):
            for j in range(extended_probabilities.shape[1]):
                plt.text(j, i, f'{extended_probabilities[i, j]:.3f}', 
                         ha='center', va='center', 
                         color='white' if extended_probabilities[i, j] < 0.5 else 'black')
        
        # Setting the labels and titles
        plt.xlabel('Hand Type')
        plt.ylabel('Kicker')
        plt.xticks(ticks=np.arange(len(self.hand_types)), labels=self.hand_types, rotation=45, ha='right')
        yticks = ['A']+['K']+['J']+['Q']+list(range(10, 1, -1)) + ['Sum']
        plt.yticks(ticks=np.arange(14), labels=yticks)
        
        # Update the title with input cards
        # input_cards = ['14H', '12S', '4H'], if it is 14, change to A, if it is 13, change to K, if it is 12, change to Q, if it is 11, change to J
        input_cards = [card.replace('14', 'A').replace('13', 'K').replace('12', 'Q').replace('11', 'J') for card in input_cards]
        # convert SHDC to ♠♥♦♣
        input_cards = [card.replace('H', '♠').replace('D', '♥').replace('S', '♦').replace('C', '♣') for card in input_cards]
        # Should be A ♦Q ♣4, the order should change to suit, rank
        input_cards = [card[1]+card[0] for card in input_cards]
        
        plt.title(f'Poker Hand Type Probabilities with Sum\nInput Cards: {", ".join(input_cards)}')
        
        plt.tight_layout()
        plt.show()

# Example usage:
# evaluator = PokerHandEvaluator()
# community_cards = ['8C', '2S', '5D', '11H', '5S'] # Example community cards
# probabilities = evaluator.evaluate(community_cards)
# evaluator.plot_heatmap_with_sum(probabilities)

if __name__ == "__main__":
    evaluator = PokerHandEvaluator()
    known_cards = ['8C', '2S', '5D', '11H', '5S']  # Example known cards

    start_time = time.time()

    #prob_matrix = evaluator.monte_carlo_simulation(known_cards, num_samples=250000)
    prob_matrix = evaluator.brutal_search(known_cards)
    #prob_matrix = evaluator.type_of_hand_kicker(known_cards)
    end_time = time.time()
    print(f"Monte Carlo simulation executed in {end_time - start_time:.2f} seconds")

    evaluator.plot_heatmap_with_sum(prob_matrix, known_cards)