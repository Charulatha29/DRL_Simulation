#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

class GazeboRobotEnv(gym.Env):
    def __init__(self):
        super(GazeboRobotEnv, self).__init__()
        
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('gazebo_gym_env')
        
        # Subscriptions & Publishers
        self.node.create_subscription(Odometry, '/odom', self._bot_odom_cb, 10)
        self.node.create_subscription(Odometry, '/obstacle/odom', self._obs_odom_cb, 10)
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.obs_cmd_pub = self.node.create_publisher(Twist, '/obstacle/cmd_vel', 10)
        
        # Service client to reset Gazebo world on episode termination
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')

        # Environment Parameters
        self.max_v, self.max_w = 0.35, 1.8
        self.bot_radius, self.obs_radius = 0.15, 0.10
        
        # Action space: [v_linear_norm, w_angular_norm]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # State space: [dist_wp, angle_wp, dist_obs, angle_obs, v_curr, w_curr]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, 0.0, -np.pi, 0.0, -self.max_w], dtype=np.float32),
            high=np.array([3.0, np.pi, 3.0, np.pi, self.max_v, self.max_w], dtype=np.float32)
        )
        
        self.bot_pos = np.array([0.25, 0.25], dtype=np.float32)
        self.bot_theta = 0.0
        self.obs_pos = np.array([1.20, 0.20], dtype=np.float32)
        self.active_waypoint = np.array([2.15, 0.95], dtype=np.float32)
        self.v_curr, self.w_curr = 0.0, 0.0
        self.obs_dir = 1.0
        self.steps = 0

    def _bot_odom_cb(self, msg):
        self.bot_pos[0] = msg.pose.pose.position.x
        self.bot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.bot_theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        self.v_curr = msg.twist.twist.linear.x
        self.w_curr = msg.twist.twist.angular.z

    def _obs_odom_cb(self, msg):
        self.obs_pos[0] = msg.pose.pose.position.x
        self.obs_pos[1] = msg.pose.pose.position.y

    def _get_obs(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        g_vec = self.active_waypoint - self.bot_pos
        dist_goal = float(np.linalg.norm(g_vec))
        angle_goal = float((math.atan2(g_vec[1], g_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)
        
        o_vec = self.obs_pos - self.bot_pos
        dist_obs = float(np.linalg.norm(o_vec))
        angle_obs = float((math.atan2(o_vec[1], o_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)
        
        return np.array([dist_goal, angle_goal, dist_obs, angle_obs, self.v_curr, self.w_curr], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        target_v = float(((action[0] + 1.0) / 2.0) * self.max_v)
        target_w = float(action[1] * self.max_w)
        
        cmd = Twist()
        cmd.linear.x = target_v
        cmd.angular.z = target_w
        self.cmd_pub.publish(cmd)
        
        # Dynamic Obstacle Patrol
        if self.obs_pos[1] > 1.00: self.obs_dir = -1.0
        elif self.obs_pos[1] < 0.20: self.obs_dir = 1.0
        obs_cmd = Twist()
        obs_cmd.linear.y = 0.15 * self.obs_dir
        self.obs_cmd_pub.publish(obs_cmd)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        obs = self._get_obs()
        dist_goal, _, dist_obs, _, _, _ = obs
        
        reward = -0.1
        terminated = False
        truncated = self.steps >= 300
        
        if dist_goal < 0.15:
            reward += 100.0
            terminated = True
        elif dist_obs < (self.bot_radius + self.obs_radius):
            reward -= 100.0
            terminated = True
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.cmd_pub.publish(Twist())
        
        # Call ROS 2 Gazebo Reset Service
        if self.reset_sim_client.wait_for_service(timeout_sec=1.0):
            req = Empty.Request()
            self.reset_sim_client.call_async(req)
            
        return self._get_obs(), {}