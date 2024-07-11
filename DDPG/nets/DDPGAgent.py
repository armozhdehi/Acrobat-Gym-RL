import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from ActorNetwork import ActorNetwork, ActorNetworkConfig
from CriticNetwork import CriticNetwork, CriticNetworkConfig
from OU_Noise import OU_Noise
from Memory import Memory

class DDPGAgentConfig(BaseModel):
    """
    Configuration for DDPG Agent.

    Attributes:
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic network.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update parameter.
        input_dim (int): Dimension of the input layer.
        fc1_units (int): Number of units in the first fully connected layer.
        fc2_units (int): Number of units in the second fully connected layer.
        action_dim (int): Dimension of the action space.
        memory_size (int): Size of the replay buffer.
        batch_size (int): Batch size for training.
        actor_model_file (str): File path for saving and loading the actor model weights.
        critic_model_file (str): File path for saving and loading the critic model weights.
        target_actor_model_file (str): File path for saving and loading the target actor model weights.
        target_critic_model_file (str): File path for saving and loading the target critic model weights.
    """
    actor_lr: float = Field(..., gt=0)
    critic_lr: float = Field(..., gt=0)
    gamma: float = Field(..., gt=0)
    tau: float = Field(..., gt=0)
    input_dim: int = Field(..., gt=0)
    fc1_units: int = Field(..., gt=0)
    fc2_units: int = Field(..., gt=0)
    action_dim: int = Field(..., gt=0)
    memory_size: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    actor_model_file: str
    critic_model_file: str
    target_actor_model_file: str
    target_critic_model_file: str

class DDPGAgent:
    def __init__(self, config: DDPGAgentConfig):
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.tau = config.tau
        
        self.memory = Memory(config.memory_size, config.input_dim, config.action_dim)
        self.noise = OU_Noise(mu = np.zeros(config.action_dim))
        
        actor_config = ActorNetworkConfig(
            learning_rate=config.actor_lr,
            input_dim=config.input_dim,
            fc1_units=config.fc1_units,
            fc2_units=config.fc2_units,
            action_dim=config.action_dim,
            model_file=config.actor_model_file
        )
        
        critic_config = CriticNetworkConfig(
            learning_rate=config.critic_lr,
            input_dim=config.input_dim,
            fc1_units=config.fc1_units,
            fc2_units=config.fc2_units,
            action_dim=config.action_dim,
            model_file=config.critic_model_file
        )
        
        self.online_actor = ActorNetwork(actor_config)
        self.online_critic = CriticNetwork(critic_config)
        
        target_actor_config = actor_config.copy(update={'model_file': config.target_actor_model_file})
        target_critic_config = critic_config.copy(update={'model_file': config.target_critic_model_file})
        
        self.target_actor = ActorNetwork(target_actor_config)
        self.target_critic = CriticNetwork(target_critic_config)
        
        self.update_targets(tau = 1)
        
    def act(self, state):
        state = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        mu = self.online_actor(state)
        mu += tf.convert_to_tensor(self.noise(), dtype=tf.float32)
        return mu.numpy()[0]
    
    def save_models(self):
        self.online_actor.save_model()
        self.online_critic.save_model()
        self.target_actor.save_model()
        self.target_critic.save_model()
        
    def load_models(self):
        self.online_actor.load_model()
        self.online_critic.load_model()
        self.target_actor.load_model()
        self.target_critic.load_model()
        
    def learn(self):
        if self.memory.index < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        target_actions_ = self.target_actor(states_)
        target_Q_values_ = self.target_critic(states_, target_actions_)
        online_Q_values = self.online_critic(states, actions)
        
        target_Q_values_[dones] = 0.0
        target_Q_values_ = tf.squeeze(target_Q_values_)
        
        target = rewards + self.gamma * target_Q_values_
        
        with tf.GradientTape() as tape:
            critic_loss = tf.keras.losses.MSE(target, tf.squeeze(online_Q_values))
        critic_grads = tape.gradient(critic_loss, self.online_critic.trainable_variables)
        self.online_critic.optimizer.apply_gradients(zip(critic_grads, self.online_critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_pred = self.online_actor(states)
            actor_loss = -tf.reduce_mean(self.online_critic(states, actions_pred))
        actor_grads = tape.gradient(actor_loss, self.online_actor.trainable_variables)
        self.online_actor.optimizer.apply_gradients(zip(actor_grads, self.online_actor.trainable_variables))
        
        self.update_targets()
        
    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for target_param, online_param in zip(self.target_actor.trainable_variables, self.online_actor.trainable_variables):
            target_param.assign(tau * online_param + (1 - tau) * target_param)
            
        for target_param, online_param in zip(self.target_critic.trainable_variables, self.online_critic.trainable_variables):
            target_param.assign(tau * online_param + (1 - tau) * target_param)

# Example usage:
config = DDPGAgentConfig(
    actor_lr=0.001,
    critic_lr=0.002,
    gamma=0.99,
    tau=0.005,
    input_dim=8,
    fc1_units=400,
    fc2_units=300,
    action_dim=2,
    memory_size=1000000,
    batch_size=64,
    actor_model_file='online_actor.h5',
    critic_model_file='online_critic.h5',
    target_actor_model_file='target_actor.h5',
    target_critic_model_file='target_critic.h5'
)
agent = DDPGAgent(config)