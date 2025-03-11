import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)   # Snake 1
GREEN1 = (0, 255, 0)  # Snake 2
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 30  # Adjust speed for AI training
FOOD_COUNT = 7  # Number of food items on the board at the same time

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('2-Player Snake AI with Multiple Foods')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game state for a new episode."""
        # Snake 1
        self.direction1 = Direction.RIGHT
        self.head1 = Point(self.w / 4, self.h / 2)
        self.snake1 = [self.head1,
                      Point(self.head1.x - BLOCK_SIZE, self.head1.y),
                      Point(self.head1.x - (2 * BLOCK_SIZE), self.head1.y)]
        self.score1 = 0

        # Snake 2
        self.direction2 = Direction.LEFT
        self.head2 = Point(3 * self.w / 4, self.h / 2)
        self.snake2 = [self.head2,
                      Point(self.head2.x + BLOCK_SIZE, self.head2.y),
                      Point(self.head2.x + (2 * BLOCK_SIZE), self.head2.y)]
        self.score2 = 0

        # Place multiple foods
        self.foods = []
        self._place_foods()

    def _place_foods(self):
        """Places multiple food items randomly on the board."""
        self.foods.clear()
        while len(self.foods) < FOOD_COUNT:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food = Point(x, y)

            # Ensure food does not overlap with snakes or existing foods
            if food not in self.snake1 and food not in self.snake2 and food not in self.foods:
                self.foods.append(food)

    def get_state(self, snake_num):
        """Returns the state representation for AI training."""
        if snake_num == 1:
            head = self.snake1[0]
            direction = self.direction1
        else:
            head = self.snake2[0]
            direction = self.direction2

        # Points around the snake
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Current movement direction
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        # Food position relative to snake head
        food_left = any(food.x < head.x for food in self.foods)
        food_right = any(food.x > head.x for food in self.foods)
        food_up = any(food.y < head.y for food in self.foods)
        food_down = any(food.y > head.y for food in self.foods)

        state = [
            # Danger in the current direction
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)
    
    def play_step(self, action1=None, action2=None):
        """Plays one step in the game with AI-controlled moves."""
        # AI-Controlled Moves
        if action1 is not None:
            self.direction1 = self._move_ai(self.direction1, action1)
        if action2 is not None:
            self.direction2 = self._move_ai(self.direction2, action2)

        # Move both snakes
        self._move(self.direction1, 1)
        self.snake1.insert(0, self.head1)

        self._move(self.direction2, 2)
        self.snake2.insert(0, self.head2)
        
        # Check for collisions
        game_over = False
        if self._is_collision(self.head1, self.snake1) or self._is_collision(self.head2, self.snake2) or self.head1 == self.head2:
            game_over = True
            return game_over, self.score1, self.score2, -10, -10  # Penalty for dying

        # Check food consumption
        reward1 = 0
        reward2 = 0

        if self.head1 in self.foods:
            self.score1 += 1
            reward1 = 10
            self.foods.remove(self.head1)
            self._place_foods()

        if self.head2 in self.foods:
            self.score2 += 1
            reward2 = 10
            self.foods.remove(self.head2)
            self._place_foods()

        # Remove tail
        self.snake1.pop()
        self.snake2.pop()

        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return game_over, self.score1, self.score2, reward1, reward2
    
    def _is_collision(self, head, snake):
        """Checks if a snake collides with a wall or itself."""
        if head.x >= self.w or head.x < 0 or head.y >= self.h or head.y < 0:
            return True
        if head in snake[1:]:
            return True
        return False

    def _move_ai(self, direction, action):
        """Maps AI action (0=straight, 1=left, 2=right) to new direction."""
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(direction)

        if action == 1:  # Turn Left
            new_dir = directions[(idx - 1) % 4]
        elif action == 2:  # Turn Right
            new_dir = directions[(idx + 1) % 4]
        else:  # Continue Straight
            new_dir = direction

        return new_dir

    def _move(self, direction, snake_num):
        """Moves a snake in the given direction."""
        if snake_num == 1:
            x, y = self.head1.x, self.head1.y
        else:
            x, y = self.head2.x, self.head2.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        if snake_num == 1:
            self.head1 = Point(x, y)
        else:
            self.head2 = Point(x, y)

    def _update_ui(self):
        """Updates the game display."""
        self.display.fill(BLACK)
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.snake1:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.snake2:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
