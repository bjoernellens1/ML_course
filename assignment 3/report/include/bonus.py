import random

class CardDeck:
    def __init__(self):
        # Initialize the deck of cards
        self.cards = ['R', 'R', 'B', 'B']
        # Shuffle the deck of cards
        random.shuffle(self.cards)
        # Initialize the prior probability
        self.prior_red = 0.5
        self.likelihood = 0.5


    def draw_card(self, index):
        """
        Draw a card from the deck
        :param index: the index of the card to be drawn
        :return: the card and its color
        """
        card = self.cards.pop(index)
        color = 'Red' if card == 'R' else 'Black'
        return (card, color)

    def get_card_color(self, index):
        """
        Get the color of a card
        :return: the color of the card
        """
        card = self.cards[index]
        color = 'Red' if card == 'R' else 'Black'
        return color

    def get_remainder(self):
        """
        Get the remainder deck
        :return: the remainder deck
        """
        return self.cards

    """
    Update the prior probability based on Bayes' rule
    function name: update_probabilities
    """
    def update_probabilities(self, card_color):
        """
        Update the prior probability based on Bayes' rule
        :param card_color: the color of the card drawn
        """
        if card_color == 'Red':
            self.prior_red = self.get_posterior('Red')
        elif card_color == 'Black':
            self.prior_red = self.get_posterior('Black')

    """
    Calculate the posterior probability based on the prior probability, likelihood, and marginal likelihood
    function name: get_posterior
    """
    def get_posterior(self, card_color):
        """
        Calculate the posterior probability based on the prior probability, likelihood, and marginal likelihood
        :param card_color: the color of the card drawn
        :return: the posterior probability of drawing a red card or a black card
        """
        likelihood = self.get_likelihood(card_color)
        marginal_likelihood = self.get_marginal_likelihood(card_color)
        posterior = (self.prior_red * likelihood) / marginal_likelihood
        return posterior

    def get_likelihood(self, card_color):
        """
        Calculate the likelihood of drawing a certain card (red or black) given the color of the card already drawn
        :param card_color: the color of the card drawn
        :return: the likelihood of drawing a certain card (red or black) given the color of the card already drawn
        """
        if card_color == 'Red':
            num_red_cards = self.cards.count('R')
            num_cards = len(self.cards)
            return num_red_cards / num_cards
        elif card_color == 'Black':
            num_black_cards = self.cards.count('B')
            num_cards = len(self.cards)
            return num_black_cards / num_cards

    def get_marginal_likelihood(self, card_color):
        """
        Calculate the marginal likelihood of drawing a certain card (red or black) regardless of the color of the card already drawn
        :param card_color: the color of the card drawn
        :return: the marginal likelihood of drawing a certain card (red or black) regardless of the color of the card already drawn
        """
        if card_color == 'Red':
            num_red_cards = self.cards.count('R')
            num_black_cards = self.cards.count('B')
            return (self.prior_red * num_red_cards / len(self.cards)) + ((1 - self.prior_red) * num_black_cards / len(self.cards))
        elif card_color == 'Black':
            num_red_cards = self.cards.count('R')
            num_black_cards = self.cards.count('B')
            return ((1 - self.prior_red) * num_red_cards / len(self.cards)) + (self.prior_red * num_black_cards / len(self.cards))


    def play_game(self):
        while self.cards:
            for i, card in enumerate(self.cards):
                print(f'{i}: Unknown')
             # only one card left
            if len(self.cards) == 1:
                # code missing here
                self.cards.pop()
            else:
                index = int(input(f"Remainder deck: {self.get_remainder()}\nEnter the index of the card you want to draw:"))
                card, color = self.draw_card(index-1)
                print(f'You drew a {color} card with value {card}')

                # Calculate the likelihood and marginal likelihood based on the color of the drawn card
                self.get_likelihood(color)
                self.get_marginal_likelihood(color)

            # Update the prior probability based on Bayes' rule
            self.update_probabilities(color)
            posterior_red = self.get_posterior('Red')  # you must calculate this value
            posterior_black = self.get_posterior('Black') # you must calculate this value
            # Print the posterior probabilities
            print(f"After drawing a {card} card:")
            print(f"The probability that the other card is red: {posterior_red:.2f}")
            print(f"The probability that the other card is black: {posterior_black:.2f}")
        print("The deck is empty.")