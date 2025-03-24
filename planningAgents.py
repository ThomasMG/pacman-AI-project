from layout import Layout
from pacman import Directions
from game import Agent
import random
import game
import util
from pacman import GameState
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import math
import matplotlib.pyplot as plt

class DQNPacman(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNPacman, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x))
        x2 = F.leaky_relu(self.fc2(x1))
        return self.fc3(x2)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def plot_training_stats(eps_rewards, max_rewards, min_rewards, episode_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(eps_rewards, color='b', label='Episode Reward')
    ax1.plot(max_rewards, color='g', label='Max Reward')
    ax1.plot(min_rewards, color='r', label='Min Reward')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Rewards per Episode")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(episode_losses, color='purple')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss per Episode")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, **kwargs):
        self.layout = layout
        self.deep_copy_layout = layout.deepCopy()
        self.best_model = None

        self.n_actions = 5 # (n,s,e,w, stop)
        self.init_total_food = self.layout.totalFood
        self.init_food_positions = self.layout.food.asList()
        
        self.food_idx = {food: self.init_food_positions.index(food) for food in self.init_food_positions}
        self.state_size = self.get_state_size()

        self.action_indices = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.action_size = len(self.actions)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # hyperparameters
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = 128
        self.GAMMA = 0.95
        self.EPS_START = 0.90
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.1
        self.LR = 1e-3

        self.policy_net = DQNPacman(self.state_size, self.n_actions).to(self.device)
        self.target_net = DQNPacman(self.state_size, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(30000)

        self.steps_done = 0
        self.is_training = True
        self.offline_planning()
        # self.load_model()

    def reset_environment(self):
        state = GameState()
        numGhostAgents = self.deep_copy_layout.getNumGhosts()
        state.initialize(self.deep_copy_layout, numGhostAgents)
        return state

    def check_terminal(self, state: GameState):
        if state.isWin() or state.isLose():
            return True
        # if state.getNumFood() == 0:
        #     return True
        return False

    def get_state_size(self):    
        base_features = 2 + 2*self.layout.getNumGhosts() + 1 + 1 + self.layout.getNumGhosts() # pacman pos + ghost pos + total food left + pac state + ghost state
        extra_features = 1 + 2*self.init_total_food + 4  # current score + food direction (dx, dy) + walls
        return base_features+extra_features

    def get_state_features(self, state: GameState):
        features_vector = []
        
        ## PACMAN POS
        pacman_pos = state.getPacmanPosition()
        features_vector.extend(pacman_pos)

        ## GHOSTS POS
        ghost_positions = state.getGhostPositions()
        for ghost_pos in ghost_positions:
            features_vector.extend(ghost_pos)
        
        ## FOOD LEFT
        food_grid = state.getFood()
        food_positions = food_grid.asList()
        features_vector.append(len(food_positions)/self.init_total_food) # norm

        dir_map = {'North':0, 'South':1, 'East':2, 'West':3, 'Stop':4}

        ## PAC DIR
        pac_state = state.getPacmanState()
        features_vector.append(dir_map[pac_state.configuration.getDirection()])

        ## GHOSTS DIR
        for g in range(1, state.getNumAgents()):
            g_state = state.getGhostState(g)
            features_vector.append(dir_map[g_state.configuration.getDirection()])

        ## CURRENT SCORE (norm)
        norm_score = state.getScore() / 500.0
        features_vector.append(norm_score)
        
        ## FOOD POS
        food_pos_vec = np.zeros(self.init_total_food*2)-1
        if food_positions:
            for food_c in food_positions:
                current_idx = self.food_idx[food_c]
                food_pos_vec[current_idx*2] = food_c[0]
                food_pos_vec[current_idx*2+1] = food_c[1]
        features_vector.extend(food_pos_vec)

        ## WALLS INFO
        is_wall = np.zeros(4) # wall = 0, non-wall=1

        # n
        new_pos = (pacman_pos[0], pacman_pos[1] + 1)
        if new_pos[1] >= self.layout.height or self.layout.walls[new_pos[0]][new_pos[1]]:
            is_wall[0] = 1
        else:
            is_wall[0] = 0

        # s
        new_pos = (pacman_pos[0], pacman_pos[1] - 1)
        if new_pos[1] < 0 or self.layout.walls[new_pos[0]][new_pos[1]]:
            is_wall[1] = 1
        else:
            is_wall[1] = 0

        # east
        new_pos = (pacman_pos[0] + 1, pacman_pos[1])
        if new_pos[0] >= self.layout.width or self.layout.walls[new_pos[0]][new_pos[1]]:
            is_wall[2] = 1
        else:
            is_wall[2] = 0

        # wst
        new_pos = (pacman_pos[0] - 1, pacman_pos[1])
        if new_pos[0] < 0 or self.layout.walls[new_pos[0]][new_pos[1]]:
            is_wall[3] = 1
        else:
            is_wall[3] = 0

        features_vector.extend(is_wall)
        return torch.tensor(features_vector, dtype=torch.float32)
                
    def select_action(self, state, legal_actions):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        # print(eps_threshold)
        if random.random() < eps_threshold and self.is_training:
            return random.choice(legal_actions)
        
        state_tensor = self.get_state_features(state).unsqueeze(0).to(self.device) 
        q_values = self.policy_net(state_tensor)
        
        # filter Q-values for legal actions only
        legal_q = []
        for action in legal_actions:
            idx = self.action_indices[action]
            legal_q.append((action, q_values[0, idx].item()))

        if not self.is_training:
            print(legal_q)
        
        # choose the action with the highest Q-value
        return max(legal_q, key=lambda x: x[1])[0]
            
    def optim_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        states, actions, next_states, rewards = batch.state, batch.action, batch.next_state, batch.reward
    
        state_batch = torch.cat(states).to(device=self.device)
        action_indices = torch.tensor([self.action_indices[a] for a in actions], device=self.device)
        reward_batch = torch.tensor(rewards, device=self.device)
        next_state_batch = torch.cat(next_states).to(device=self.device)
    
        # current Q values
        current_q_values = self.policy_net(state_batch)
        # print(current_q_values)
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1))
        # print(current_q_values)
        
        # next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute target Q values
        target_q_values = reward_batch + (self.GAMMA * next_q_values)
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # print('loss: ', loss.item())
        self.optimizer.step()
        return loss.item()

    def save_model(self, model, path="dqn_pacman.pth"):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="dqn_pacman.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.is_training = False  

    def offline_planning(self):
        self.is_training = True
        num_episodes = 100
        max_episode_rewards = [] 
        min_episode_rewards = [] 
        eps_rewards = []
        episode_losses = []  
        best_score = -math.inf

        hard_update_frequency = 50

        for episode in range(num_episodes):
            print("Episode:", episode)
            state = self.reset_environment()
            done = False
            max_reward = -math.inf
            min_reward = math.inf
            losses = []
            previous_score = state.getScore()
            eps_avg_rew = []

            while not done:
                legal_actions = state.getLegalPacmanActions()
                action = self.select_action(state, legal_actions)
                next_state = state.generateSuccessor(0, action)
                done = self.check_terminal(next_state)         

                # Simulate ghost moves
                if not done:
                    next_copy = next_state.deepCopy()
                    for g in range(1, next_state.getNumAgents()):
                        dist = util.Counter()
                        for a in next_copy.getLegalActions(g):
                            dist[a] = 1.0
                        dist.normalize()
                        if len(dist) == 0:
                            next_state = next_copy.generateSuccessor(g, Directions.STOP)
                        else:
                            next_state = next_copy.generateSuccessor(g, util.chooseFromDistribution(dist))
                        # next_copy = next_state.deepCopy()
                        done = self.check_terminal(next_state)
                        if done:
                            break

                current_score = next_state.getScore()       
                reward = current_score - previous_score

                previous_food_count = len(state.getFood().asList())
                current_food_count = len(next_state.getFood().asList())

                if current_food_count < previous_food_count:
                    food_eaten = previous_food_count - current_food_count
                    reward += food_eaten * 30  # more per food
                                
                # penalize score based on ghost distance
                ghost_positions = state.getGhostPositions()
                pacman_pos = state.getPacmanPosition()

                min_dist = -1
                for ghost_pos in ghost_positions:
                    distance = util.manhattanDistance(pacman_pos, ghost_pos)
                    if distance == 0:
                        distance = 0.5
                    min_dist = min(min_dist, distance)
                penalty = 10 / min_dist 
                reward -= penalty

                previous_score = current_score
                eps_avg_rew.append(reward)

                max_reward = max(reward, max_reward)
                min_reward = min(reward, min_reward)
                done = self.check_terminal(next_state)

                self.memory.push(
                    torch.tensor(self.get_state_features(state)).unsqueeze(0),
                    action,
                    torch.tensor(self.get_state_features(next_state)).unsqueeze(0),
                    reward
                )
                
                state = next_state
                            
                if len(self.memory) >= self.BATCH_SIZE:
                    loss = self.optim_model()
                    if loss is not None:
                        losses.append(loss)

                if current_score > best_score:
                    best_score = current_score
                self.best_model = self.policy_net

                # self.optim_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

            max_episode_rewards.append(max_reward)
            min_episode_rewards.append(min_reward)
            eps_rewards.append(np.mean(eps_avg_rew))
            avg_loss = sum(losses)/len(losses) if losses else 0
            episode_losses.append(avg_loss)
            print(f"max: {max_reward}, min: {min_reward}, avg: {np.mean(eps_avg_rew)}")

            if (episode + 1) % hard_update_frequency == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print("hard update")

        print('best score: ', best_score)
        self.save_model(self.best_model)
        self.is_training = False

        plot_training_stats(eps_rewards, max_episode_rewards, min_episode_rewards, episode_losses)
            
    def getAction(self, state: GameState):
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return Directions.STOP
        return self.select_action(state, legal_actions)