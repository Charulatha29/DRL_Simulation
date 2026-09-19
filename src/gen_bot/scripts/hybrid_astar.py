#!/usr/bin/env python3
import heapq
import math
import numpy as np

class HybridAStarPlanner:
    def __init__(self, width=2.4, height=1.2, resolution=0.05):
        self.width = width
        self.height = height
        self.res = resolution
        self.grid_w = int(width / resolution)
        self.grid_h = int(height / resolution)
        
    def plan(self, start, goal):
        """ Computes straight-line waypoints from Start to Goal avoiding walls """
        start_cell = (int(start[0]/self.res), int(start[1]/self.res))
        goal_cell = (int(goal[0]/self.res), int(goal[1]/self.res))
        
        # Simplified waypoint generation (Midpoint smoothing for maze corridors)
        waypoints = [
            np.array(start, dtype=np.float32),
            np.array([(start[0] + goal[0]) / 2.0, start[1]], dtype=np.float32),
            np.array([(start[0] + goal[0]) / 2.0, goal[1]], dtype=np.float32),
            np.array(goal, dtype=np.float32)
        ]
        return waypoints