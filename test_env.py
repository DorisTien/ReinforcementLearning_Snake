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

        # Get state for both snakes (fix function call)
        state1 = game.get_state(1)
        state2 = game.get_state(2)

        # Placeholder for game termination logic
        game_over = False  # Keep running until a max step limit or logical termination
        reward1 = random.randint(-10, 10)  # Placeholder reward
        reward2 = random.randint(-10, 10)  # Placeholder reward
        score1 = random.randint(0, 10)  # Placeholder score
        score2 = random.randint(0, 10)  # Placeholder score

        # Print information
        print(f"Player 1: Score = {score1}, Reward = {reward1} | Player 2: Score = {score2}, Reward = {reward2}")

        # End game if the termination condition is met
        if game_over:
            print("Game Over!")
            if step_count >= 1000:
                print("Max steps reached, ending test.")
                running = False  # Run for at least 1000 steps before stopping
            print("Max steps reached, ending test.")
            running = False

    pygame.quit()

if __name__ == "__main__":
    test_environment()
