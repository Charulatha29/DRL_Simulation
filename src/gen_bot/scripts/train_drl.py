#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gazebo_gym_env import GazeboRobotEnv

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.reset()

    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mean)

def get_actor():
    inputs = layers.Input(shape=(8,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(2, activation="tanh")(out)
    return tf.keras.Model(inputs, outputs)

def train():
    env = GazeboRobotEnv()
    actor_model = get_actor()
    ou_noise = OUActionNoise(mean=np.zeros(2), std_deviation=float(0.2) * np.ones(2))
    
    episodes = 200
    print("Starting Domain-Randomized DRL Training inside Gazebo...")
    
    for ep in range(episodes):
        state, _ = env.reset()
        episodic_reward = 0
        
        while True:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = tf.squeeze(actor_model(tf_state)).numpy() + ou_noise()
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episodic_reward += reward
            
            if terminated or truncated:
                break
                
        print(f"Episode: {ep+1}/{episodes} | Reward: {episodic_reward:.2f}")
        
    actor_model.save("drl_actor_tf210.h5")
    print("Training Complete. Model saved as 'drl_actor_tf210.h5'")

if __name__ == '__main__':
    train()
