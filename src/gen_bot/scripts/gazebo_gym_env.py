#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
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
        self.marker_pub = self.node.create_publisher(Marker, '/visual_goal_marker', 10)
        
        # Service client to reset Gazebo world on episode termination
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')

        # Environment Parameters
        self.max_v, self.max_w = 0.35, 1.8
        self.bot_radius, self.obs_radius = 0.15, 0.10
        
        # Strict 2.4m x 1.2m Maze Boundaries (Robot Center Limits)
        self.x_min, self.x_max = 0.15, 2.25
        self.y_min, self.y_max = 0.15, 1.05
        
        # Action space: [v_linear_norm, w_angular_norm]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Updated 8D State space: [d_wp, θ_wp, d_obs, θ_obs, v_rel_x, v_rel_y, v_curr, w_curr]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, 0.0, -np.pi, -1.5, -1.5, 0.0, -self.max_w], dtype=np.float32),
            high=np.array([3.0, np.pi, 3.0, np.pi, 1.5, 1.5, self.max_v, self.max_w], dtype=np.float32)
        )
        
        self.bot_pos = np.array([0.25, 0.25], dtype=np.float32)
        self.bot_theta = 0.0
        self.obs_pos = np.array([1.20, 0.20], dtype=np.float32)
        self.active_waypoint = np.array([2.15, 0.95], dtype=np.float32)
        self.v_curr, self.w_curr = 0.0, 0.0
        
        # Relative Velocity Tracking
        self.prev_d_obs = None
        self.prev_theta_obs = None
        self.prev_dist_wp = None
        self.dt = 0.1
        self.steps = 0
        
        # Domain Randomization State
        self.obs_dir = 1.0
        self.obs_vx = 0.0
        self.obs_vy = 0.20

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

    def _publish_goal_marker(self, goal_x, goal_y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "active_goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_x)
        marker.pose.position.y = float(goal_y)
        marker.pose.position.z = 0.1
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.2
        marker.color.a = 0.9
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def is_out_of_bounds(self, pos):
        return not (self.x_min <= pos[0] <= self.x_max and self.y_min <= pos[1] <= self.y_max)

    def _get_obs(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        g_vec = self.active_waypoint - self.bot_pos
        dist_goal = float(np.linalg.norm(g_vec))
        angle_goal = float((math.atan2(g_vec[1], g_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)
        
        o_vec = self.obs_pos - self.bot_pos
        dist_obs = float(np.linalg.norm(o_vec))
        angle_obs = float((math.atan2(o_vec[1], o_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)
        
        # Calculate Relative Obstacle Velocities
        if self.prev_d_obs is not None:
            v_rel_x = float((dist_obs * math.cos(angle_obs) - self.prev_d_obs * math.cos(self.prev_theta_obs)) / self.dt)
            v_rel_y = float((dist_obs * math.sin(angle_obs) - self.prev_d_obs * math.sin(self.prev_theta_obs)) / self.dt)
        else:
            v_rel_x, v_rel_y = 0.0, 0.0
            
        self.prev_d_obs = dist_obs
        self.prev_theta_obs = angle_obs
        
        self._publish_goal_marker(self.active_waypoint[0], self.active_waypoint[1])
        
        return np.array([dist_goal, angle_goal, dist_obs, angle_obs, v_rel_x, v_rel_y, self.v_curr, self.w_curr], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        target_v = float(((action[0] + 1.0) / 2.0) * self.max_v)
        target_w = float(action[1] * self.max_w)
        
        cmd = Twist()
        cmd.linear.x = target_v
        cmd.angular.z = target_w
        self.cmd_pub.publish(cmd)
        
        # Dynamic Obstacle Patrol Logic (Bounce between y=0.25 and y=0.95)
        if self.obs_pos[1] > 0.95:
            self.obs_dir = -1.0
        elif self.obs_pos[1] < 0.25:
            self.obs_dir = 1.0
            
        obs_cmd = Twist()
        obs_cmd.linear.x = 0.0
        obs_cmd.linear.y = float(0.20 * self.obs_dir)
        self.obs_cmd_pub.publish(obs_cmd)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        obs = self._get_obs()
        dist_goal, _, dist_obs, _, _, _, _, _ = obs
        
        terminated = False
        truncated = self.steps >= 300
        
        # Reward Computation
        reward_progress = (self.prev_dist_wp - dist_goal) * 20.0 if self.prev_dist_wp is not None else 0.0
        self.prev_dist_wp = dist_goal
        
        reward_clearance = -5.0 if dist_obs < 0.40 else 0.0
        reward_smoothness = -0.1 * abs(target_w)
        reward = reward_progress + reward_clearance + reward_smoothness - 0.1
        
        if dist_goal < 0.15:
            reward += 100.0
            terminated = True
        elif dist_obs < (self.bot_radius + self.obs_radius):
            reward -= 150.0
            terminated = True
        elif self.is_out_of_bounds(self.bot_pos):
            reward -= 150.0
            terminated = True
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_d_obs = None
        self.prev_theta_obs = None
        self.prev_dist_wp = None
        self.cmd_pub.publish(Twist())
        
        # Randomize Waypoint inside Interior 2.4m x 1.2m bounds
        self.active_waypoint[0] = np.random.uniform(0.3, 2.1)
        self.active_waypoint[1] = np.random.uniform(0.3, 0.9)
        
        if self.reset_sim_client.wait_for_service(timeout_sec=1.0):
            req = Empty.Request()
            self.reset_sim_client.call_async(req)
            
        return self._get_obs(), {}
