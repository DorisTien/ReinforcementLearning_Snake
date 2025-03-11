import pygame
import random
from snake_env import SnakeGameAI, Direction  # Import the environment

def test_environment():
    game = SnakeGameAI()  # Initialize the environment
    
    running = True
    while running:
        # Generate random moves for both snakes (0 = straight, 1 = left, 2 = right)
        action1 = random.randint(0, 2)
        action2 = random.randint(0, 2)

        # Play one step in the environment
        game_over, score1, score2, reward1, reward2 = game.play_step(action1, action2)

        # Print information
        print(f"Player 1: Score = {score1}, Reward = {reward1} | Player 2: Score = {score2}, Reward = {reward2}")

        # End game if both snakes die
        if game_over:
            print("Game Over!")
            running = False

    pygame.quit()

if __name__ == "__main__":
    test_environment()
