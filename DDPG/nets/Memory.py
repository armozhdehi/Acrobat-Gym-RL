import numpy as np
from pydantic import BaseModel, Field

class MemoryConfig(BaseModel):
    """
    Configuration for Memory.

    Attributes:
        size (int): Size of the replay buffer.
        input_dim (tuple): Dimension of the input state.
        action_dim (int): Dimension of the action space.
    """
    size: int = Field(..., gt=0)
    input_dim: tuple = Field(...)
    action_dim: int = Field(..., gt=0)

class Memory:
    """
    Replay buffer for DDPG agent.

    Attributes:
        size (int): Size of the replay buffer.
        index (int): Current index in the buffer.
        state_store (np.ndarray): Array to store states.
        resulted_state_store (np.ndarray): Array to store next states.
        action_store (np.ndarray): Array to store actions.
        reward_store (np.ndarray): Array to store rewards.
        done_store (np.ndarray): Array to store done flags.
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the replay buffer.

        Args:
            config (MemoryConfig): Configuration object for the replay buffer.
        """
        self.size = config.size
        self.index = 0
        self.state_store = np.zeros((self.size, *config.input_dim))
        self.resulted_state_store = np.zeros((self.size, *config.input_dim))
        self.action_store = np.zeros((self.size, config.action_dim))
        self.reward_store = np.zeros(self.size)
        self.done_store = np.zeros(self.size, dtype=bool)
    
    def store(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Done flag.
        """
        idx = self.index % self.size
        self.state_store[idx] = state        
        self.action_store[idx] = action        
        self.reward_store[idx] = reward       
        self.resulted_state_store[idx] = next_state        
        self.done_store[idx] = done
        
        self.index += 1
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Size of the batch to sample.

        Returns:
            Tuple: Batch of states, actions, rewards, next states, and done flags.
        """
        max_idx = min(self.size, self.index)
        batch_indices = np.random.choice(max_idx, batch_size)
        
        states = self.state_store[batch_indices]
        actions = self.action_store[batch_indices]
        rewards = self.reward_store[batch_indices]
        next_states = self.resulted_state_store[batch_indices]
        dones = self.done_store[batch_indices]
        
        return states, actions, rewards, next_states, dones

# Example usage:
config = MemoryConfig(
    size=1000000,
    input_dim=(8,),
    action_dim=2
)
memory = Memory(config)