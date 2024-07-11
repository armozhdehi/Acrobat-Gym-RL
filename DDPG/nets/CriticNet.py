import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from pydantic import BaseModel, Field, validator

class CriticNetConfig(BaseModel):
    """
    Configuration for Critic Network.

    
    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        input_dim (int): Dimension of the input layer.
        fc1_units (int): Number of units in the first fully connected layer.
        fc2_units (int): Number of units in the second fully connected layer.
        action_dim (int): Dimension of the action space.
        model_file (str): File path for saving and loading the model weights.
    """
    learning_rate: float = Field(..., gt=0)
    input_dim: int = Field(..., gt=0)
    fc1_units: int = Field(..., gt=0)
    fc2_units: int = Field(..., gt=0)
    action_dim: int = Field(..., gt=0)
    model_file: str

    @validator('model_file')
    def validate_model_file(cls, v):
        if not v.endswith('.h5'):
            raise ValueError('Model file must end with .h5')
        return v

class CriticNet(Model):
    """
    Critic Network for DDPG.
    """

    def __init__(self, config: CriticNetConfig):
        """
        Initialize the Critic Network.

        Args:
            config (CriticNetConfig): Configuration object for the Critic Network.
        """
        super(CriticNet, self).__init__()
        self.fc1 = layers.Dense(config.fc1_units, input_shape=(config.input_dim,), kernel_initializer=self.init_weights(config.input_dim))
        self.ln1 = layers.LayerNormalization()
        self.fc2 = layers.Dense(config.fc2_units, kernel_initializer=self.init_weights(config.fc1_units))
        self.ln2 = layers.LayerNormalization()
        self.action_layer = layers.Dense(config.fc2_units, kernel_initializer=self.init_weights(config.action_dim))
        self.Q_value = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        
        self.optimizer = optimizers.Adam(learning_rate=config.learning_rate, decay=0.01)
        self.model_file = config.model_file
    
    def call(self, state, action):
        """
        Forward pass through the network.

        Args:
            state (tf.Tensor): Input state tensor.
            action (tf.Tensor): Input action tensor.

        Returns:
            tf.Tensor: Q-value predictions.
        """
        s = self.fc1(state)
        s = self.ln1(s)
        s = tf.nn.relu(s)
        s = self.fc2(s)
        s = self.ln2(s)
        a = self.action_layer(action)
        s_a = tf.nn.relu(tf.add(s, a))
        Q_s_a = self.Q_value(s_a)
        
        return Q_s_a
    
    def init_weights(self, size):
        """
        Initialize weights for the layers.

        Args:
            size (int): Size of the input for the layer.

        Returns:
            tf.initializers.Initializer: Uniform initializer for the weights.
        """
        return tf.random_uniform_initializer(minval=-1./np.sqrt(size), maxval=1./np.sqrt(size))
    
    def save_model(self):
        """
        Save the model weights to the specified file.
        """
        print(f'Saving {self.model_file}...')
        self.save_weights(self.model_file)

    def load_model(self):
        """
        Load the model weights from the specified file.
        """
        print(f'Loading {self.model_file}...')
        self.load_weights(self.model_file)

# Example usage:
config = CriticNetConfig(
    learning_rate=0.001,
    input_dim=8,
    fc1_units=400,
    fc2_units=300,
    action_dim=2,
    model_file='critic.h5'
)
critic = CriticNet(config)
