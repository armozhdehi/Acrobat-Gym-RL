import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from pydantic import BaseModel, Field
from ActorNetwork import ActorNetwork, ActorNetworkConfig
from CriticNetwork import CriticNetwork, CriticNetworkConfig
from Memory import Memory

class OUNoiseConfig(BaseModel):
    """
    Configuration for OU Noise.

    Attributes:
        mu (np.ndarray): Mean of the noise process.
        sigma (float): Volatility parameter (default 0.5).
        theta (float): Speed of mean reversion (default 0.2).
        dt (float): Time step (default 0.01).
        x0 (Optional[np.ndarray]): Initial value of the noise process (default None).
    """
    mu: np.ndarray
    sigma: float = Field(default=0.5, gt=0)
    theta: float = Field(default=0.2, gt=0)
    dt: float = Field(default=1e-2, gt=0)
    x0: Optional[np.ndarray] = None

class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.

    Attributes:
        mu (np.ndarray): Mean of the noise process.
        sigma (float): Volatility parameter.
        theta (float): Speed of mean reversion.
        dt (float): Time step.
        x0 (Optional[np.ndarray]): Initial value of the noise process.
        x_prev (np.ndarray): Previous value of the noise process.
    """

    def __init__(self, config: OUNoiseConfig):
        """
        Initialize the OU Noise process.

        Args:
            config (OUNoiseConfig): Configuration object for the OU Noise process.
        """
        self.mu = config.mu
        self.sigma = config.sigma
        self.theta = config.theta
        self.dt = config.dt
        self.x0 = config.x0
        
        self.reset()

    def __call__(self):
        """
        Generate the next value of the noise process.

        Returns:
            np.ndarray: Next value of the noise process.
        """
        diff = self.mu - self.x_prev
        rnd = np.random.normal(size=self.mu.shape)
        x = self.x_prev + self.theta * self.dt * diff + self.sigma * np.sqrt(self.dt) * rnd
        self.x_prev = x
        return x
    
    def reset(self):
        """
        Reset the noise process to the initial value.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

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
        
        noise_config = OUNoiseConfig(mu = np.zeros(config.action_dim))
        self.noise = OUNoise(noise_config)
        
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
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        target_actions = self.target_actor(next_states)
        target_Q_values = self.target_critic(next_states, target_actions)
        online_Q_values = self.online_critic(states, actions)
        
        target_Q_values = tf.where(dones, tf.zeros_like(target_Q_values), target_Q_values)
        target_Q_values = tf.squeeze(target_Q_values)
        
        target = rewards + self.gamma * target_Q_values
        
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

