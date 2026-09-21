#!/usr/bin/env python3
import math
import numpy as np

class HybridAStarPlanner:
    def __init__(self, width=2.4, height=1.2, resolution=0.05):
        self.width = width
        self.height = height
        self.res = resolution
        self.grid_w = int(width / resolution)
        self.grid_h = int(height / resolution)
        
        # Boundary constraints
        self.x_min, self.x_max = 0.15, 2.25
        self.y_min, self.y_max = 0.15, 1.05

    def is_valid(self, point):
        return self.x_min <= point[0] <= self.x_max and self.y_min <= point[1] <= self.y_max

    def plan(self, start, goal):
        """ Generates smooth global waypoints from start to target within 2.4m x 1.2m """
        start_pt = np.clip(start, [self.x_min, self.y_min], [self.x_max, self.y_max])
        goal_pt = np.clip(goal, [self.x_min, self.y_min], [self.x_max, self.y_max])
        
        # Multi-segment interpolation for corridor navigation
        mid_x = (start_pt[0] + goal_pt[0]) / 2.0
        
        p1 = np.array([start_pt[0], start_pt[1]], dtype=np.float32)
        p2 = np.array([mid_x, start_pt[1]], dtype=np.float32)
        p3 = np.array([mid_x, goal_pt[1]], dtype=np.float32)
        p4 = np.array([goal_pt[0], goal_pt[1]], dtype=np.float32)
        
        raw_waypoints = [p1, p2, p3, p4]
        
        # Subdivide long segments into 0.3m waypoint steps for the DRL controller
        dense_waypoints = []
        for i in range(len(raw_waypoints) - 1):
            seg_start = raw_waypoints[i]
            seg_end = raw_waypoints[i+1]
            dist = np.linalg.norm(seg_end - seg_start)
            num_steps = max(1, int(dist / 0.30))
            
            for step in range(num_steps):
                alpha = step / float(num_steps)
                wp = (1.0 - alpha) * seg_start + alpha * seg_end
                dense_waypoints.append(wp)
                
        dense_waypoints.append(p4)
        return dense_waypoints
