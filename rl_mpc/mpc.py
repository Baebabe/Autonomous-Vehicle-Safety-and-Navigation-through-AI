import numpy as np
import casadi as ca
import glob
import os
import sys
sys.path.append('C:\\Users\\msi\\miniconda3\\envs\\sdcarAB\\Carla-0.10.0-Win64-Shipping\\PythonAPI')
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import math
import pygame
from queue import PriorityQueue
from typing import List, Tuple, Set


class MPCController:
    def __init__(self):
        # MPC Parameters
        self.dt = 0.1  # Time step [s]
        self.N = 10    # Prediction horizon
        self.L = 2.9  # Wheelbase [m]
        
        # State constraints
        self.max_speed = 15.0  # [km/h]
        self.max_steer = 0.7   # [rad]
        
        # Weights for MPC cost function
        self.Q_x = 5.0      # Position x - INCREASED for better path following
        self.Q_y = 5.0      # Position y - INCREASED for better path following
        self.Q_yaw = 2.0    # Yaw angle - INCREASED for better path following
        self.Q_v = 0.5      # Velocity - INCREASED for smoother velocity control
        self.R_throttle = 0.5  # REDUCED to allow more aggressive control when needed
        self.Q_cte = 8.0    # NEW: Cross-track error weight
        self.Q_etheta = 3.0 # NEW: Heading error weight
        self.R_steer = 1.0  # NEW: Explicit steering weight
        self.R_steer_rate = 2.5 # NEW: Increased penalty for steering changes
    
        # Path tracking
        self.waypoints = []
        self.visited_waypoints = set()
        self.current_waypoint_index = 0
        self.waypoint_distance_threshold = 3.0  # Distance to consider waypoint reached
        
        # A* parameters
        self.waypoint_distance = 3.0
        self.max_search_dist = 200.0
        
        # Visualization
        self.debug_lifetime = 0.1
        self.path_visualization_done = False
        
        # Store reference to vehicle and world
        self._parent = None
        self.world = None
        self.map = None
        
        # Initialize MPC solver
        self._setup_mpc()
        
    def _setup_mpc(self):
        """Setup MPC optimization problem with improved path centering"""
        # State variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y, yaw, v)
        n_states = states.size1()

        # Control inputs
        throttle = ca.SX.sym('throttle')
        steering = ca.SX.sym('steering')
        controls = ca.vertcat(throttle, steering)
        n_controls = controls.size1()

        # Vehicle dynamics - IMPROVED with more accurate model
        rhs = ca.vertcat(
            v * ca.cos(yaw),
            v * ca.sin(yaw),
            v * ca.tan(steering) / self.L,
            throttle * 10.0 - 0.1 * v
        )

        # Create function for dynamics
        f = ca.Function('f', [states, controls], [rhs])

        # Decision variables
        X = ca.SX.sym('X', n_states, self.N + 1)
        U = ca.SX.sym('U', n_controls, self.N)

        # NEW: Added two extra parameters per waypoint for path tangent vector
        # Format: [initial_state, ref_x1, ref_y1, ref_yaw1, ref_v1, ref_dx1, ref_dy1, ...]
        P = ca.SX.sym('P', n_states + (n_states + 2) * self.N)

        # Initialize objective and constraints
        obj = 0
        g = []

        # Initial state constraint
        g.append(X[:, 0] - P[0:n_states])

        # Previous control for rate constraints (initialized at 0)
        prev_steering = U[1, 0]

        # Dynamics constraints and objective function
        for k in range(self.N):
            # State and control at current step
            st = X[:, k]
            con = U[:, k]

            # RK4 integration
            k1 = f(st, con)
            k2 = f(st + self.dt/2 * k1, con)
            k3 = f(st + self.dt/2 * k2, con)
            k4 = f(st + self.dt * k3, con)
            st_next = st + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            # Add dynamics constraint
            g.append(X[:, k + 1] - st_next)

            # Reference state at current step
            ref_idx = n_states + k * (n_states + 2)
            ref_x = P[ref_idx]
            ref_y = P[ref_idx + 1]
            ref_yaw = P[ref_idx + 2]
            ref_v = P[ref_idx + 3]

            # Path tangent vector (for better cross-track error calculation)
            path_dx = P[ref_idx + 4]
            path_dy = P[ref_idx + 5]

            # Normalize the tangent vector
            path_norm = ca.sqrt(path_dx**2 + path_dy**2) + 1e-6  # Avoid division by zero
            path_dx = path_dx / path_norm
            path_dy = path_dy / path_norm

            # Calculate cross-track error (perpendicular distance to the path)
            # This uses the tangent vector to determine distance perpendicular to path
            dx = X[0, k] - ref_x
            dy = X[1, k] - ref_y

            # Cross-track error - perpendicular distance to the path
            # Using a dot product with the normal vector to the path
            cte = ca.fabs(-path_dy * dx + path_dx * dy)

            # Calculate yaw error correctly considering angle wrapping
            yaw_diff = X[2, k] - ref_yaw
            yaw_cost = ca.fmod(yaw_diff + ca.pi, 2*ca.pi) - ca.pi

            # Calculate heading error relative to the path tangent
            # Path tangent angle
            path_angle = ca.atan2(path_dy, path_dx)
            # Heading error - difference between vehicle yaw and path tangent
            etheta = ca.fmod(X[2, k] - path_angle + ca.pi, 2*ca.pi) - ca.pi

            # Position error costs
            obj += self.Q_x * dx**2 + self.Q_y * dy**2

            # Cross-track error cost - heavily penalize deviation from the path center
            obj += self.Q_cte * cte**2

            # Yaw error costs
            obj += self.Q_yaw * yaw_cost**2

            # Heading error cost - align vehicle orientation with the path
            obj += self.Q_etheta * etheta**2

            # Velocity error cost
            obj += self.Q_v * (X[3, k] - ref_v)**2

            # Control costs
            obj += self.R_throttle * U[0, k]**2  # Throttle cost
            obj += self.R_steer * U[1, k]**2     # Steering cost

            # Steering rate cost (change in steering)
            if k > 0:
                obj += self.R_steer_rate * (U[1, k] - U[1, k-1])**2

            # Update previous control
            prev_steering = U[1, k]

        # Terminal cost - higher weight for final state
        terminal_idx = n_states + (self.N-1) * (n_states + 2)
        obj += 10.0 * self.Q_x * (X[0, self.N] - P[terminal_idx])**2
        obj += 10.0 * self.Q_y * (X[1, self.N] - P[terminal_idx + 1])**2

        # Create solver with better options
        opts = {
            'ipopt': {
                'print_level': 0,
                'acceptable_tol': 1e-7,
                'acceptable_obj_change_tol': 1e-5,
                'max_iter': 150,
                'warm_start_init_point': 'yes'
            },
            'print_time': 0
        }

        nlp = {
            'x': ca.vertcat(ca.vec(X), ca.vec(U)),
            'f': obj,
            'g': ca.vertcat(*g),
            'p': P
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Save dimensions and symbolic variables
        self.n_states = n_states
        self.n_controls = n_controls
        self.X = X
        self.U = U
        self.P = P
        
    # def set_path(self, world, start_location, end_location):
    #     """Generate shortest path using A* algorithm"""
    #     try:
    #         self.world = world
    #         self.map = world.get_map()
            
    #         # Convert locations to waypoints
    #         start_waypoint = self.map.get_waypoint(start_location)
    #         end_waypoint = self.map.get_waypoint(end_location)
            
    #         print(f"Planning path using A* from {start_waypoint.transform.location} to {end_waypoint.transform.location}")
            
    #         # Find path using A*
    #         path, distance = self._find_path_astar(start_waypoint, end_waypoint)
            
    #         if not path:
    #             print("No path found!")
    #             return False
            
    #         self.waypoints = path
    #         self.current_waypoint_index = 0
    #         self.visited_waypoints.clear()
    #         self.path_visualization_done = False
            
    #         print(f"Path found with {len(path)} waypoints and distance {distance:.1f} meters")
            
    #         # Visualize the path
    #         self._visualize_complete_path(world, path, distance)
            
    #         return True
            
    #     except Exception as e:
    #         print(f"Path planning failed: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         return False
        
    def _heuristic(self, waypoint, goal_waypoint):
        """A* heuristic: straight-line distance to goal"""
        return waypoint.transform.location.distance(goal_waypoint.transform.location)
    
    # def _find_path_astar(self, start_wp, end_wp):
    #     """A* algorithm implementation for finding shortest path"""
    #     # Priority queue for A* (f_score, counter, waypoint)
    #     counter = 0  # Unique identifier for comparing waypoints
    #     open_set = PriorityQueue()
    #     start_f_score = self._heuristic(start_wp, end_wp)
    #     open_set.put((start_f_score, counter, start_wp))
        
    #     # For reconstructing path
    #     came_from = {}
        
    #     # g_score: cost from start to current node
    #     g_score = {}
    #     g_score[start_wp] = 0
        
    #     # f_score: estimated total cost from start to goal through current node
    #     f_score = {}
    #     f_score[start_wp] = start_f_score
        
    #     # Keep track of explored paths for visualization
    #     explored_paths = []
        
    #     # Set for tracking nodes in open_set
    #     open_set_hash = {start_wp}
        
    #     while not open_set.empty():
    #         current = open_set.get()[2]
    #         open_set_hash.remove(current)
            
    #         # Check if we reached the destination
    #         if current.transform.location.distance(end_wp.transform.location) < self.waypoint_distance:
    #             # Reconstruct path
    #             path = []
    #             temp_current = current
    #             while temp_current in came_from:
    #                 path.append(temp_current)
    #                 temp_current = came_from[temp_current]
    #             path.append(start_wp)
    #             path.reverse()
                
    #             # Calculate total distance
    #             total_distance = g_score[current]
                
    #             # Visualize explored paths
    #             self._visualize_exploration(explored_paths)
                
    #             return path, total_distance
            
    #         # Get next waypoints
    #         next_wps = current.next(self.waypoint_distance)
            
    #         for next_wp in next_wps:
    #             # Calculate tentative g_score
    #             tentative_g_score = g_score[current] + current.transform.location.distance(
    #                 next_wp.transform.location)
                
    #             if next_wp not in g_score or tentative_g_score < g_score[next_wp]:
    #                 # This is a better path, record it
    #                 came_from[next_wp] = current
    #                 g_score[next_wp] = tentative_g_score
    #                 f_score[next_wp] = tentative_g_score + self._heuristic(next_wp, end_wp)
                    
    #                 if next_wp not in open_set_hash:
    #                     counter += 1  # Increment counter for unique comparison
    #                     open_set.put((f_score[next_wp], counter, next_wp))
    #                     open_set_hash.add(next_wp)
                        
    #                     # Store for visualization
    #                     explored_paths.append((current, next_wp))
                        
    #                     # Real-time visualization of exploration
    #                     self.world.debug.draw_line(
    #                         current.transform.location + carla.Location(z=0.5),
    #                         next_wp.transform.location + carla.Location(z=0.5),
    #                         thickness=0.1,
    #                         color=carla.Color(64, 64, 255),  # Light blue for exploration
    #                         life_time=0.1
    #                     )
        
    #     return None, float('inf')

    def set_path(self, world, start_location, end_location, timeout=60.0):
        """Generate shortest path using A* algorithm with GlobalRoutePlanner fallback"""
        try:
            self.world = world
            self.map = world.get_map()

            # Convert locations to waypoints
            start_waypoint = self.map.get_waypoint(start_location)
            end_waypoint = self.map.get_waypoint(end_location)

            print(f"Planning path using A* from {start_waypoint.transform.location} to {end_waypoint.transform.location}")

            # Find path using A* with timeout
            import time
            start_time = time.time()
            path, distance = self._find_path_astar(start_waypoint, end_waypoint)

            # Check if path planning timed out or failed
            planning_time = time.time() - start_time
            if not path:
                print(f"A* failed after {planning_time:.2f} seconds!.")

                # Import GlobalRoutePlanner and GlobalRoutePlannerDAO
                try:
                    
                    
                    from carla.agents.navigation.global_route_planner import GlobalRoutePlanner
                    from carla.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
                    # Setup DAO and GlobalRoutePlanner
                    dao = GlobalRoutePlannerDAO(world.get_map(), 2.0)  # 2.0 meter resolution
                    grp = GlobalRoutePlanner(dao)
                    grp.setup()

                    # Get route as list of waypoints - using the same start/end waypoints as A*
                    route = grp.trace_route(
                        start_waypoint.transform.location, 
                        end_waypoint.transform.location
                    )

                    # Extract waypoints from the route
                    # In some CARLA versions, the route contains (waypoint, RoadOption) tuples
                    if isinstance(route[0], tuple):
                        path = [item[0] for item in route]  
                    else:
                        path = route

                    if not path:
                        print("GlobalRoutePlanner failed to find a path!")
                        return False

                    # Calculate route distance
                    distance = 0
                    for i in range(len(path) - 1):
                        distance += path[i].transform.location.distance(path[i+1].transform.location)

                    print(f"GlobalRoutePlanner found a path with {len(path)} waypoints and distance {distance:.1f} meters")

                except Exception as grp_error:
                    print(f"GlobalRoutePlanner failed: {str(grp_error)}")
                    print("Falling back to basic CARLA navigation")

                    # Last resort: Try to use basic CARLA waypoint following
                    current_wp = start_waypoint
                    path = [current_wp]
                    total_distance = 0
                    max_iterations = 20000  # Limit to prevent infinite loops
                    iterations = 0

                    # Simple approach: Follow road to destination
                    while (current_wp.transform.location.distance(end_waypoint.transform.location) > 10.0 and 
                           iterations < max_iterations):
                        next_wps = current_wp.next(2.0)  # Get next waypoints
                        if not next_wps:
                            break

                        # Choose waypoint closest to destination
                        next_wp = min(next_wps, key=lambda wp: wp.transform.location.distance(end_waypoint.transform.location))
                        path.append(next_wp)

                        # Update distance
                        total_distance += current_wp.transform.location.distance(next_wp.transform.location)
                        current_wp = next_wp
                        iterations += 1

                    if iterations >= max_iterations:
                        print("Basic navigation reached iteration limit!")
                        return False

                    if not path or len(path) < 2:
                        print("Basic navigation failed to find a path!")
                        return False

                    print(f"Basic navigation found a path with {len(path)} waypoints and distance {total_distance:.1f} meters")
                    distance = total_distance

            self.waypoints = path
            self.current_waypoint_index = 0
            self.visited_waypoints.clear()
            self.path_visualization_done = False

            # Visualize the path
            self._visualize_complete_path(world, path, distance)

            return True

        except Exception as e:
            print(f"Path planning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _find_path_astar(self, start_wp, end_wp, timeout=60.0):
        """A* algorithm implementation for finding shortest path with timeout"""
        # Priority queue for A* (f_score, counter, waypoint)
        counter = 0  # Unique identifier for comparing waypoints
        open_set = PriorityQueue()
        start_f_score = self._heuristic(start_wp, end_wp)
        open_set.put((start_f_score, counter, start_wp))

        # For reconstructing path
        came_from = {}

        # g_score: cost from start to current node
        g_score = {}
        g_score[start_wp] = 0

        # f_score: estimated total cost from start to goal through current node
        f_score = {}
        f_score[start_wp] = start_f_score

        # Keep track of explored paths for visualization
        explored_paths = []

        # Set for tracking nodes in open_set
        open_set_hash = {start_wp}

        # Start time for timeout tracking
        import time
        start_time = time.time()

        while not open_set.empty():
            # Check if timeout reached
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"A* search timed out after {elapsed_time:.2f} seconds")
                return None, float('inf')

            current = open_set.get()[2]
            open_set_hash.remove(current)

            # Check if we reached the destination
            if current.transform.location.distance(end_wp.transform.location) < self.waypoint_distance:
                # Reconstruct path
                path = []
                temp_current = current
                while temp_current in came_from:
                    path.append(temp_current)
                    temp_current = came_from[temp_current]
                path.append(start_wp)
                path.reverse()

                # Calculate total distance
                total_distance = g_score[current]

                # Visualize explored paths
                self._visualize_exploration(explored_paths)

                return path, total_distance

            # Get next waypoints
            next_wps = current.next(self.waypoint_distance)

            for next_wp in next_wps:
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + current.transform.location.distance(
                    next_wp.transform.location)

                if next_wp not in g_score or tentative_g_score < g_score[next_wp]:
                    # This is a better path, record it
                    came_from[next_wp] = current
                    g_score[next_wp] = tentative_g_score
                    f_score[next_wp] = tentative_g_score + self._heuristic(next_wp, end_wp)

                    if next_wp not in open_set_hash:
                        counter += 1  # Increment counter for unique comparison
                        open_set.put((f_score[next_wp], counter, next_wp))
                        open_set_hash.add(next_wp)

                        # Store for visualization
                        explored_paths.append((current, next_wp))

                        # Real-time visualization of exploration
                        self.world.debug.draw_line(
                            current.transform.location + carla.Location(z=0.5),
                            next_wp.transform.location + carla.Location(z=0.5),
                            thickness=0.1,
                            color=carla.Color(64, 64, 255),  # Light blue for exploration
                            life_time=0.1
                        )

        return None, float('inf')

    def get_control(self, vehicle, world=None):
        """Calculate control commands using MPC"""
        try:
            if not self.waypoints:
                print("No waypoints available!")
                return carla.VehicleControl(throttle=0, steer=0, brake=1.0)
            
            # Get current vehicle state
            current_state = self._get_vehicle_state(vehicle)
            
            # Update current waypoint index - FIXED WAYPOINT FOLLOWING LOGIC
            self._update_current_waypoint_index(vehicle)
            
            # Generate reference trajectory
            ref_traj = self._generate_reference_trajectory(current_state)
            
            # Solve MPC problem
            u_optimal = self._solve_mpc(current_state, ref_traj)
            
            # Check if optimization failed
            if u_optimal is None:
                print("MPC optimization failed!")
                return carla.VehicleControl(throttle=0, steer=0, brake=1.0)
            
            # Extract throttle and steering from u_optimal
            throttle = u_optimal[0]
            steer = u_optimal[1]  # Direct steering from MPC
            
            # Create and return control command
            control = carla.VehicleControl(
                throttle=max(0, min(1, throttle)),  # Clip throttle between 0 and 1
                steer=float(steer),                 # Use MPC steering directly
                brake=max(0, -throttle),            # Apply brake if throttle is negative
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )
            
            # Update visualization if world is provided
            if world is not None:
                self._visualize(world, vehicle)
            
            return control
            
        except Exception as e:
            print(f"Error in get_control: {str(e)}")
            import traceback
            traceback.print_exc()
            return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

    def _solve_mpc(self, current_state, ref_traj):
        """Solve MPC optimization problem with improved path centering"""
        try:
            # Dimensions
            nx = self.n_states * (self.N + 1)
            nu = self.n_controls * self.N

            # Initialize solution vectors with improved initial guess
            x_init = np.zeros(nx + nu)

            # Set initial state and propagate with simple dynamics
            x_init[:self.n_states] = current_state

            # Simple propagation for better initialization
            for i in range(1, self.N + 1):
                prev_idx = (i-1) * self.n_states
                curr_idx = i * self.n_states

                # Propagate state with constant velocity and heading
                dt = self.dt
                x_init[curr_idx] = x_init[prev_idx] + x_init[prev_idx + 3] * np.cos(x_init[prev_idx + 2]) * dt
                x_init[curr_idx + 1] = x_init[prev_idx + 1] + x_init[prev_idx + 3] * np.sin(x_init[prev_idx + 2]) * dt
                x_init[curr_idx + 2] = x_init[prev_idx + 2]
                x_init[curr_idx + 3] = x_init[prev_idx + 3]

            # Set control initialization
            for i in range(self.N):
                # Set small initial throttle and steering
                x_init[nx + i*self.n_controls] = 0.1  # Small throttle
                x_init[nx + i*self.n_controls + 1] = 0.0  # Zero steering

            # Parameters vector with extended reference information (including path tangent)
            # Format: [initial_state, ref_x1, ref_y1, ref_yaw1, ref_v1, ref_dx1, ref_dy1, ...]
            p = np.zeros(self.n_states + (self.n_states + 2) * self.N)
            p[:self.n_states] = current_state

            for i in range(self.N):
                if i < len(ref_traj):
                    idx = self.n_states + i * (self.n_states + 2)
                    p[idx:idx + self.n_states + 2] = ref_traj[i]  # Now includes dx, dy for path tangent
                else:
                    idx = self.n_states + i * (self.n_states + 2)
                    p[idx:idx + self.n_states + 2] = ref_traj[-1]

            # Variable bounds
            lbx = -np.inf * np.ones(nx + nu)
            ubx = np.inf * np.ones(nx + nu)

            # State bounds
            for i in range(self.N + 1):
                lbx[i * self.n_states + 3] = 0  # Minimum velocity
                ubx[i * self.n_states + 3] = self.max_speed / 3.6  # Maximum velocity

            # Control bounds
            for i in range(self.N):
                lbx[nx + i*self.n_controls] = -1.0    # Minimum throttle
                ubx[nx + i*self.n_controls] = 1.0     # Maximum throttle
                lbx[nx + i*self.n_controls + 1] = -self.max_steer  # Minimum steering
                ubx[nx + i*self.n_controls + 1] = self.max_steer   # Maximum steering

            # Constraints bounds
            ng = self.n_states * (self.N + 1)
            lbg = np.zeros(ng)
            ubg = np.zeros(ng)

            # Solve NLP
            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=p
            )

            # Check solution status
            if self.solver.stats()['success']:
                # Extract control inputs (first control only)
                u_opt = sol['x'][nx:nx + self.n_controls]
                return np.array(u_opt).flatten()  # Return array with throttle and steering
            else:
                print(f"Optimization failed: {self.solver.stats()['return_status']}")
                return None

        except Exception as e:
            print(f"Error in MPC optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    
    def _get_vehicle_state(self, vehicle):
        """Get current vehicle state [x, y, yaw, v]"""
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        v = math.sqrt(velocity.x**2 + velocity.y**2)  # m/s
        
        return np.array([x, y, yaw, v])
    
    def _generate_reference_trajectory(self, current_state):
        """Generate reference trajectory from waypoints with path tangent information"""
        ref_traj = []

        # Start from current waypoint index
        wp_idx = self.current_waypoint_index

        # Calculate lookahead distance based on current speed
        current_speed = current_state[3]  # m/s
        lookahead_steps = max(1, min(5, int(current_speed * 1.5)))  # Dynamic lookahead

        # Generate reference points for prediction horizon
        for i in range(self.N):
            # Calculate target waypoint index with lookahead
            target_idx = wp_idx + i // lookahead_steps

            if target_idx < len(self.waypoints):
                wp = self.waypoints[target_idx]
                x = wp.transform.location.x
                y = wp.transform.location.y
                yaw = math.radians(wp.transform.rotation.yaw)

                # Calculate path tangent vector (direction of the path)
                # Look ahead for tangent calculation
                tangent_idx = min(target_idx + 1, len(self.waypoints) - 1)
                if tangent_idx > target_idx:
                    next_wp = self.waypoints[tangent_idx]
                    dx = next_wp.transform.location.x - x
                    dy = next_wp.transform.location.y - y
                else:
                    # At the last waypoint, use the approach direction
                    prev_idx = max(0, target_idx - 1)
                    prev_wp = self.waypoints[prev_idx]
                    dx = x - prev_wp.transform.location.x
                    dy = y - prev_wp.transform.location.y

                # Normalize tangent vector
                mag = math.sqrt(dx**2 + dy**2)
                if mag > 0.001:  # Avoid division by zero
                    dx /= mag
                    dy /= mag
                else:
                    # Fallback: use waypoint's yaw to compute tangent
                    dx = math.cos(yaw)
                    dy = math.sin(yaw)

                # Adjust target speed based on path curvature
                if target_idx + 1 < len(self.waypoints):
                    next_wp = self.waypoints[target_idx + 1]

                    # Calculate path curvature
                    dx1 = x - current_state[0]
                    dy1 = y - current_state[1]
                    dx2 = next_wp.transform.location.x - x
                    dy2 = next_wp.transform.location.y - y

                    # Compute angle between vectors
                    angle1 = math.atan2(dy1, dx1)
                    angle2 = math.atan2(dy2, dx2)
                    angle_diff = abs(angle1 - angle2)
                    while angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff

                    # Adjust speed based on curvature - more reduction in speed for sharp turns
                    curvature_factor = 1.0 - 0.6 * (angle_diff / math.pi)  # More aggressive speed reduction
                    target_speed = (self.max_speed / 3.6) * max(0.25, curvature_factor)  # Min 25% speed
                else:
                    target_speed = self.max_speed / 3.6

                # For last waypoint, set target speed to zero
                if target_idx == len(self.waypoints) - 1:
                    target_speed = 0.0
            else:
                # If we run out of waypoints, use the last one
                wp = self.waypoints[-1]
                x = wp.transform.location.x
                y = wp.transform.location.y
                yaw = math.radians(wp.transform.rotation.yaw)

                # Use last segment for tangent
                if len(self.waypoints) > 1:
                    prev_wp = self.waypoints[-2]
                    dx = x - prev_wp.transform.location.x
                    dy = y - prev_wp.transform.location.y
                    mag = math.sqrt(dx**2 + dy**2)
                    if mag > 0.001:
                        dx /= mag
                        dy /= mag
                    else:
                        dx = math.cos(yaw)
                        dy = math.sin(yaw)
                else:
                    dx = math.cos(yaw)
                    dy = math.sin(yaw)

                target_speed = 0.0  # Target zero velocity at end of path

            # Store reference state with path tangent
            ref_traj.append([x, y, yaw, target_speed, dx, dy])

        return np.array(ref_traj)
    
    def _update_current_waypoint_index(self, vehicle):
        """Update current waypoint index based on vehicle position"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return
        
        # Get vehicle location
        vehicle_loc = vehicle.get_location()
        
        # Check if current waypoint is reached
        current_wp = self.waypoints[self.current_waypoint_index]
        distance = vehicle_loc.distance(current_wp.transform.location)
        
        # Debug information
        print(f"Distance to waypoint {self.current_waypoint_index}: {distance:.2f}m")
        
        # Waypoint reached if within threshold
        if distance < self.waypoint_distance_threshold:
            self.visited_waypoints.add(self.current_waypoint_index)
            self.current_waypoint_index += 1
            print(f"Waypoint {self.current_waypoint_index-1} reached, moving to next waypoint")
            
            # Safety check
            if self.current_waypoint_index >= len(self.waypoints):
                print("Reached final waypoint!")
                self.current_waypoint_index = len(self.waypoints) - 1
        
        # Lookahead logic - skip waypoints if we're closer to a future waypoint
        elif self.current_waypoint_index + 1 < len(self.waypoints):
            # Check distances to next few waypoints
            for i in range(1, min(5, len(self.waypoints) - self.current_waypoint_index)):
                next_wp_idx = self.current_waypoint_index + i
                next_wp = self.waypoints[next_wp_idx]
                next_distance = vehicle_loc.distance(next_wp.transform.location)
                
                # If we're closer to a future waypoint and significantly off the current one
                if next_distance < distance * 0.7:  # 30% closer to future waypoint
                    # Skip to that waypoint
                    print(f"Skipping to waypoint {next_wp_idx} as it's closer")
                    for j in range(self.current_waypoint_index, next_wp_idx):
                        self.visited_waypoints.add(j)
                    self.current_waypoint_index = next_wp_idx
                    break
    
    def _get_speed(self, vehicle):
        """Get current speed in km/h"""
        vel = vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2)
    
    def _visualize(self, world, vehicle):
        """Visualize real-time progress along the A* path and MPC prediction"""
        try:
            if not self.waypoints:
                return

            # Vehicle current pose
            v_loc = vehicle.get_location()
            v_rot = vehicle.get_transform().rotation

            # First time visualization of complete path with A* results
            if not self.path_visualization_done:
                self._visualize_complete_path(world, self.waypoints, 
                                            self._calculate_total_distance(self.waypoints))
                self.path_visualization_done = True

            # Draw current progress on path
            for i in range(len(self.waypoints) - 1):
                start = self.waypoints[i].transform.location
                end = self.waypoints[i + 1].transform.location

                # Color based on whether waypoint has been visited
                if i < self.current_waypoint_index:
                    color = carla.Color(0, 255, 0)  # Green for visited
                else:
                    color = carla.Color(255, 0, 0)  # Red for upcoming

                world.debug.draw_line(
                    start + carla.Location(z=0.5),
                    end + carla.Location(z=0.5),
                    thickness=0.1,
                    color=color,
                    life_time=self.debug_lifetime
                )

            # Draw current target waypoint
            if self.current_waypoint_index < len(self.waypoints):
                target = self.waypoints[self.current_waypoint_index]
                world.debug.draw_point(
                    target.transform.location + carla.Location(z=0.5),
                    size=0.2,
                    color=carla.Color(0, 255, 0),  # Green
                    life_time=0.1
                )

                # Also draw a path from vehicle to current target
                vehicle_loc = vehicle.get_location()
                world.debug.draw_arrow(
                    vehicle_loc + carla.Location(z=0.5),
                    target.transform.location + carla.Location(z=0.5),
                    thickness=0.1,
                    arrow_size=0.3,
                    color=carla.Color(255, 255, 0),  # Yellow
                    life_time=self.debug_lifetime
                )

            # Visualize next few waypoints
            for i in range(1, min(5, len(self.waypoints) - self.current_waypoint_index)):
                next_wp_idx = self.current_waypoint_index + i
                next_wp = self.waypoints[next_wp_idx]

                # Draw a line from previous to next waypoint
                prev_wp = self.waypoints[next_wp_idx - 1]
                world.debug.draw_line(
                    prev_wp.transform.location + carla.Location(z=0.5),
                    next_wp.transform.location + carla.Location(z=0.5),
                    thickness=0.1,
                    color=carla.Color(0, 255, 0),  # Green
                    life_time=0.1
                )

                # Draw waypoint with different size based on index
                size = 0.1 + 0.02 * i
                world.debug.draw_point(
                    next_wp.transform.location + carla.Location(z=0.5),
                    size=size,
                    color=carla.Color(0, 255, 0),  # Green
                    life_time=0.1
                )

            # Draw MPC predicted trajectory if available
            if hasattr(self, 'predicted_trajectory') and self.predicted_trajectory is not None:
                for i in range(len(self.predicted_trajectory) - 1):
                    world.debug.draw_line(
                        carla.Location(x=self.predicted_trajectory[i][0], 
                                      y=self.predicted_trajectory[i][1], 
                                      z=vehicle.get_location().z + 0.5),
                        carla.Location(x=self.predicted_trajectory[i+1][0], 
                                      y=self.predicted_trajectory[i+1][1], 
                                      z=vehicle.get_location().z + 0.5),
                        thickness=0.05,
                        color=carla.Color(0, 255, 255),  # Cyan
                        life_time=self.debug_lifetime
                    )

            # Show the center of the lane at the vehicle's position
            current_lane_wp = self.map.get_waypoint(v_loc)
            world.debug.draw_point(
                current_lane_wp.transform.location + carla.Location(z=0.5),
                size=0.2,
                color=carla.Color(255, 255, 0),  # Yellow
                life_time=0.1
            )

            # Calculate and visualize cross-track error
            lane_vec = current_lane_wp.transform.get_forward_vector()

            # Manually normalize the vector
            lane_vec_length = (lane_vec.x**2 + lane_vec.y**2)**0.5
            if lane_vec_length > 0:
                normalized_x = lane_vec.x / lane_vec_length
                normalized_y = lane_vec.y / lane_vec_length
            else:
                normalized_x, normalized_y = 0, 0

            # Create unit vector
            lane_vec = carla.Vector3D(normalized_x, normalized_y, 0)

            # Calculate perpendicular vector (right side of lane)
            perp_vec = carla.Vector3D(-lane_vec.y, lane_vec.x, 0)

            # Calculate vehicle offset from lane center
            vehicle_vec = carla.Vector3D(
                v_loc.x - current_lane_wp.transform.location.x,
                v_loc.y - current_lane_wp.transform.location.y,
                0
            )

            # Cross track error (positive = right of center, negative = left of center)
            cte = vehicle_vec.x * perp_vec.x + vehicle_vec.y * perp_vec.y

            # Visualize the cross-track error
            perp_point = carla.Location(
                x=current_lane_wp.transform.location.x + perp_vec.x * cte,
                y=current_lane_wp.transform.location.y + perp_vec.y * cte,
                z=current_lane_wp.transform.location.z
            )

            world.debug.draw_line(
                current_lane_wp.transform.location + carla.Location(z=0.5),
                perp_point + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(255, 0, 0),  # Red
                life_time=0.1
            )

            # Display CTE value
            world.debug.draw_string(
                perp_point + carla.Location(z=1.0),
                f"CTE: {cte:.2f}m",
                draw_shadow=True,
                color=carla.Color(255, 255, 255),
                life_time=0.1
            )

            # Draw progress percentage and current metrics
            progress = (len(self.visited_waypoints) / len(self.waypoints)) * 100 if self.waypoints else 0
            current_loc = vehicle.get_location()
            distance_to_target = current_loc.distance(
                self.waypoints[self.current_waypoint_index].transform.location
            ) if self.current_waypoint_index < len(self.waypoints) else 0

            # Draw progress information
            info_text = [
                f"Progress: {progress:.1f}%",
                f"Distance to next waypoint: {distance_to_target:.1f}m",
                f"Waypoints remaining: {len(self.waypoints) - self.current_waypoint_index}",
                f"Speed: {self._get_speed(vehicle):.1f} km/h"
            ]

            for i, text in enumerate(info_text):
                world.debug.draw_string(
                    current_loc + carla.Location(z=2.0 + i * 0.5),  # Stack text vertically
                    text,
                    color=carla.Color(255, 255, 255),
                    life_time=self.debug_lifetime
                )
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            pass

    def _visualize_complete_path(self, world, path, total_distance):
        """Visualize the complete planned path with metrics"""
        if not path:
            return
            
        # Draw complete path
        for i in range(len(path) - 1):
            start = path[i].transform.location
            end = path[i + 1].transform.location
            
            # Draw path line
            world.debug.draw_line(
                start + carla.Location(z=0.5),
                end + carla.Location(z=0.5),
                thickness=0.3,
                color=carla.Color(0, 255, 0),  # Green
                life_time=0.0
            )
            
            # Draw waypoint markers
            world.debug.draw_point(
                start + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(255, 255, 0),  # Yellow
                life_time=0.0
            )
        
        # Draw start and end points
        world.debug.draw_point(
            path[0].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(0, 255, 0),  # Green for start
            life_time=0.0
        )
        
        world.debug.draw_point(
            path[-1].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(255, 0, 0),  # Red for end
            life_time=0.0
        )

        # Draw distance markers and path info
        for i in range(len(path) - 1):
            current_distance = path[0].transform.location.distance(path[i].transform.location)
            if int(current_distance) % 10 == 0:  # Every 10 meters
                world.debug.draw_string(
                    path[i].transform.location + carla.Location(z=2.0),
                    f"{int(current_distance)}m",
                    color=carla.Color(255, 255, 255),
                    life_time=0.0
                )

        # Draw path information
        info_location = path[0].transform.location + carla.Location(z=3.0)
        world.debug.draw_string(
            info_location,
            f"Path Length: {total_distance:.1f}m",
            color=carla.Color(0, 255, 0),
            life_time=0.0
        )

    def _calculate_total_distance(self, path):
        """Calculate total path distance"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += path[i].transform.location.distance(
                path[i + 1].transform.location
            )
        return total_distance

    def _visualize_exploration(self, explored_paths):
        """Visualize all explored paths"""
        if self.world is None:
            return
            
        for start_wp, end_wp in explored_paths:
            self.world.debug.draw_line(
                start_wp.transform.location + carla.Location(z=0.5),
                end_wp.transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(173, 216, 230),  # Light blue
                life_time=0.0
            )