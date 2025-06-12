import carla
import math
import numpy as np
from queue import PriorityQueue
import time

class NavigationController:
    def __init__(self):
        # Control parameters
        self.max_steer = 0.7
        self.target_speed = 30.0  # km/h

        # Adjusted controller gains for smoother control
        self.k_p_lateral = 0.9    # Reduced from 1.5
        self.k_p_heading = 0.8    # Reduced from 1.0
        self.k_p_speed = 1.0      

        # Path tracking
        self.waypoints = []
        self.visited_waypoints = set()
        self.current_waypoint_index = 0
        self.waypoint_distance_threshold = 2.0

        # A* parameters
        self.waypoint_distance = 2.0
        self.max_search_dist = 200.0

        # Store reference to vehicle for speed calculations
        self._parent = None
        self.last_control = None
        
        # Visualization
        self.debug_lifetime = 0.1
        self.path_visualization_done = False
        
        # Obstacle detection
        self.obstacles = []
        
        # World reference for visualization
        self.world = None
        self.map = None
        self.last_valid_waypoint_index = 0
        self.last_recovery_time = 0
        self.recovery_cooldown = 1.0 

    def _heuristic(self, waypoint, goal_waypoint):
        """A* heuristic: straight-line distance to goal"""
        return waypoint.transform.location.distance(goal_waypoint.transform.location)

    def set_path(self, world, start_location, end_location):
        """Generate shortest path using A* algorithm"""
        try:
            self.world = world
            self.map = world.get_map()
            
            # Convert locations to waypoints
            start_waypoint = self.map.get_waypoint(start_location)
            end_waypoint = self.map.get_waypoint(end_location)
            
            print(f"Planning path using A* from {start_waypoint.transform.location} to {end_waypoint.transform.location}")
            
            # Find path using A*
            path, distance = self._find_path_astar(start_waypoint, end_waypoint)
            
            if not path:
                print("No path found!")
                return False
            
            self.waypoints = path
            self.current_waypoint_index = 0
            self.visited_waypoints.clear()
            
            print(f"Path found with {len(path)} waypoints and distance {distance:.1f} meters")
            
            # Visualize the path and exploration
            self._visualize_complete_path(world, path, distance)
            
            return True
            
        except Exception as e:
            print(f"Path planning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _find_path_astar(self, start_wp, end_wp):
        """A* algorithm implementation for finding shortest path with strict lane adherence"""
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

        while not open_set.empty():
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

            # NEW: Get next waypoints but only consider waypoints in the same lane
            next_wps = current.next(self.waypoint_distance)
            # Filter for waypoints in the same lane
            same_lane_wps = [wp for wp in next_wps if wp.lane_id == current.lane_id]

            # If no same-lane waypoints are available, then use the original ones but with a penalty
            if not same_lane_wps and next_wps:
                next_wps_to_use = next_wps
                lane_change_penalty = 50.0  # High penalty for changing lanes
            else:
                next_wps_to_use = same_lane_wps if same_lane_wps else next_wps
                lane_change_penalty = 0.0

            for next_wp in next_wps_to_use:
                # Calculate tentative g_score with lane penalty if applicable
                base_distance = current.transform.location.distance(next_wp.transform.location)
                # Add lane change penalty if lane IDs don't match
                lane_penalty = lane_change_penalty if next_wp.lane_id != current.lane_id else 0.0
                tentative_g_score = g_score[current] + base_distance + lane_penalty

                if next_wp not in g_score or tentative_g_score < g_score[next_wp]:
                    # This is a better path, record it
                    came_from[next_wp] = current
                    g_score[next_wp] = tentative_g_score
                    f_score[next_wp] = tentative_g_score + self._heuristic(next_wp, end_wp)

                    if next_wp not in open_set_hash:
                        counter += 1  # Increment counter for unique comparison
                        open_set.put((f_score[next_wp], counter, next_wp))
                        open_set_hash.add(next_wp)

                        # Store for visualization with color based on lane change
                        explored_paths.append((current, next_wp))

                        # Real-time visualization of exploration
                        color = carla.Color(64, 64, 255)  # Light blue for exploration
                        if next_wp.lane_id != current.lane_id:
                            color = carla.Color(255, 64, 64)  # Red for lane changes

                        self.world.debug.draw_line(
                            current.transform.location + carla.Location(z=0.5),
                            next_wp.transform.location + carla.Location(z=0.5),
                            thickness=0.1,
                            color=color,
                            life_time=0.1
                        )

        return None, float('inf')

    def get_control(self, vehicle, world=None):
        """Calculate control commands to follow path with enhanced curvature-based speed control"""
        try:
            if not self.waypoints:
                print("No waypoints available!")
                return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

            self._parent = vehicle
            # Get current vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            current_speed = self._get_speed(vehicle)

            # Update the world reference if provided
            if world is not None and self.world is None:
                self.world = world

            # First, update path memory if we're moving steadily
            if current_speed > 2.0 and self.current_waypoint_index > self.last_valid_waypoint_index:
                # Only update when we're actually making progress
                self.last_valid_waypoint_index = self.current_waypoint_index

            # ENHANCED: More aggressive off-path detection
            if self.current_waypoint_index < len(self.waypoints):
                target_wp = self.waypoints[self.current_waypoint_index]
                distance_to_path = vehicle_loc.distance(target_wp.transform.location)

                # If we're further from path or haven't recovered recently
                current_time = time.time()
                if (distance_to_path > 2.0 and  # Reduced from 3.0
                    (not hasattr(self, 'last_recovery_time') or 
                     current_time - self.last_recovery_time > self.recovery_cooldown)):
                    print(f"Off path: {distance_to_path:.2f}m - triggering recovery")
                    return self.force_path_recovery(vehicle)

            # Update current waypoint index based on distance
            self._update_waypoint_progress(vehicle_loc)

            # Get target waypoint
            target_wp = self.waypoints[self.current_waypoint_index]

            # Calculate distance to current waypoint
            distance_to_waypoint = vehicle_loc.distance(target_wp.transform.location)

            # Calculate steering
            steer = self._calculate_steering(vehicle_transform, target_wp.transform)

            # ENHANCED: Calculate path curvature for next few waypoints
            look_ahead = 3  # Look at next 3 waypoints
            upcoming_curvature = self._calculate_upcoming_curvature(look_ahead)

            # ENHANCED: Speed control based on path curvature and distance to waypoint
            # Significantly reduce speed in turns
            base_target_speed = self.target_speed
            curvature_factor = max(0.3, 1.0 - (upcoming_curvature * 2.0))  # More aggressive speed reduction

            # Combine factors for final target speed
            target_speed = base_target_speed * curvature_factor

            # Further reduce speed if we're far from the target waypoint
            if distance_to_waypoint > 3.0:  # Reduced from 5.0
                target_speed *= 0.6  # More aggressive speed reduction (from 0.7)

            error = target_speed - current_speed

            # Calculate throttle and brake
            if error > 0:
                throttle = min(abs(error) * self.k_p_speed, 0.7)  # Reduced max throttle for better control
                brake = 0.0
            else:
                throttle = 0.0
                # More aggressive braking for better speed control in turns
                brake = min(abs(error) * self.k_p_speed * 1.2, 0.8)  # Increased brake factor

            # Ensure minimum throttle when starting from stop
            if current_speed < 0.1 and not brake:
                throttle = max(throttle, 0.3)  # Minimum throttle to overcome inertia

            # Gradual steering changes for stability
            if self.last_control:
                max_steer_change = 0.12  # Slightly increased (from 0.1)
                steer = np.clip(
                    steer,
                    self.last_control.steer - max_steer_change,
                    self.last_control.steer + max_steer_change
                )

            # Debug output
            print(f"\nPath following status:")
            print(f"Current waypoint index: {self.current_waypoint_index}/{len(self.waypoints)-1}")
            print(f"Last valid waypoint index: {self.last_valid_waypoint_index}")
            print(f"Distance to waypoint: {distance_to_waypoint:.2f}m")
            print(f"Current speed: {current_speed:.2f}km/h")
            print(f"Target speed: {target_speed:.2f}km/h (curvature factor: {curvature_factor:.2f})")
            print(f"Upcoming curvature: {upcoming_curvature:.3f}")
            print(f"Controls - Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}")

            # Visualize the path if world is available
            if self.world is not None:
                self._visualize(self.world, vehicle)

            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )

            self.last_control = control
            return control

        except Exception as e:
            print(f"Error in get_control: {str(e)}")
            import traceback
            traceback.print_exc()
            return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

    def _calculate_upcoming_curvature(self, look_ahead=3):
        """Calculate the maximum curvature in the upcoming path segment"""
        try:
            if self.current_waypoint_index >= len(self.waypoints) - 2:
                return 0.0

            max_curvature = 0.0
            for i in range(min(look_ahead, len(self.waypoints) - self.current_waypoint_index - 2)):
                idx = self.current_waypoint_index + i

                # Get three consecutive waypoints
                p1 = self.waypoints[idx].transform.location
                p2 = self.waypoints[idx + 1].transform.location
                p3 = self.waypoints[idx + 2].transform.location

                # Calculate vectors
                v1 = np.array([p2.x - p1.x, p2.y - p1.y])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])

                # Calculate angle between vectors
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)

                if norms < 1e-6:
                    continue

                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # Normalize curvature to [0, 1] range and consider it absolute value
                segment_curvature = abs(angle / np.pi)
                max_curvature = max(max_curvature, segment_curvature)

            return max_curvature

        except Exception as e:
            print(f"Error calculating upcoming curvature: {str(e)}")
            return 0.0

    def _calculate_steering(self, vehicle_transform, waypoint_transform):
        """Calculate steering angle with strict lane adherence and enhanced path following"""
        try:
            # Current vehicle state
            veh_loc = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            current_speed = self._get_speed(self._parent)

            # NEW: Get the map if not already available
            if self.map is None and self._parent is not None:
                self.map = self._parent.get_world().get_map()

            # NEW: Get current vehicle waypoint to check lane position
            if self.map is not None:
                current_waypoint = self.map.get_waypoint(veh_loc)
                target_waypoint = self.waypoints[self.current_waypoint_index]

                # NEW: Stronger lane centering for strict path adherence
                lane_centering_correction = 0.0
                if current_waypoint:
                    # Get lane direction vector
                    lane_dir = current_waypoint.transform.get_forward_vector()

                    # Get vector from vehicle to lane center
                    lane_center = current_waypoint.transform.location
                    to_center = carla.Vector3D(
                        lane_center.x - veh_loc.x,
                        lane_center.y - veh_loc.y,
                        0
                    )

                    # Normalize vector
                    to_center_norm = math.sqrt(to_center.x**2 + to_center.y**2)
                    if to_center_norm > 0.1:  # Avoid division by zero
                        to_center.x /= to_center_norm
                        to_center.y /= to_center_norm

                    # Cross product to determine direction (positive = need to turn right)
                    cross_z = lane_dir.x * to_center.y - lane_dir.y * to_center.x

                    # Calculate distance to lane center for proportional control
                    dist_to_center = current_waypoint.transform.location.distance(veh_loc)

                    # ENHANCED: Stronger correction when further from center
                    lane_centering_correction = -cross_z * dist_to_center * 0.8  # Increased from 0.5

                    # Limit correction but allow stronger corrections
                    lane_centering_correction = np.clip(lane_centering_correction, -0.7, 0.7)  # Increased from -0.5, 0.5

                    # Debug visualization
                    if self.world:
                        self.world.debug.draw_line(
                            veh_loc + carla.Location(z=0.5),
                            lane_center + carla.Location(z=0.5),
                            thickness=0.1,
                            color=carla.Color(0, 255, 255),  # Cyan
                            life_time=0.1
                        )

            # ENHANCED: Shorter lookahead for much stricter path following
            # Dynamically adjust lookahead based on speed but keep it very short
            base_lookahead = max(1.0, min(current_speed * 0.1, 3.0))  # Further reduced lookahead

            # Calculate path curvature with enhanced influence
            curvature = self._estimate_path_curvature()

            # ENHANCED: Reduce lookahead in turns even more aggressively
            lookahead_distance = base_lookahead * (1.0 - 0.4 * abs(curvature))  # More reduction in turns

            # Use more preview points with higher weighting on close points
            preview_points = self._get_preview_points(lookahead_distance)

            # Calculate weighted steering based on preview points
            total_steering = 0
            total_weight = 0

            for i, target_loc in enumerate(preview_points):
                # Much higher weight on closest point for lane adherence
                weight = 1.0 / (i * 0.5 + 1.0)  # Even more emphasis on immediate path

                # Convert target to vehicle's coordinate system
                dx = target_loc.x - veh_loc.x
                dy = target_loc.y - veh_loc.y

                # Transform target into vehicle's coordinate system
                cos_yaw = math.cos(vehicle_yaw)
                sin_yaw = math.sin(vehicle_yaw)

                target_x = dx * cos_yaw + dy * sin_yaw
                target_y = -dx * sin_yaw + dy * cos_yaw

                # Calculate angle and immediate path curvature
                angle = math.atan2(target_y, target_x)
                if abs(target_x) > 0.01:
                    point_curvature = 2.0 * target_y / (target_x * target_x + target_y * target_y)
                else:
                    point_curvature = 0.0

                # ENHANCED: Improved factor balance for tighter tracking
                point_steering = (
                    0.2 * point_curvature +  # Further reduced curvature influence
                    1.2 * self.k_p_lateral * (target_y / lookahead_distance) +  # Significantly increased cross-track
                    0.8 * self.k_p_heading * angle  # Stronger heading correction
                )

                total_steering += point_steering * weight
                total_weight += weight

            # Calculate final steering
            if total_weight > 0:
                steering = total_steering / total_weight
            else:
                steering = 0.0

            # Add lane centering correction
            steering += lane_centering_correction

            # NEW: Don't reduce steering authority at higher speeds as much
            speed_factor = min(current_speed / 40.0, 1.0)  # Higher threshold for speed-based reduction
            max_steer_change = 0.15 * (1.0 - 0.2 * speed_factor)  # Allow more rapid steering changes

            # Apply steering limits but allow more aggressive changes when needed
            if self.last_control:
                steering = np.clip(
                    steering,
                    self.last_control.steer - max_steer_change,
                    self.last_control.steer + max_steer_change
                )

            # ENHANCED: Maintain higher steering authority in turns
            max_steer = self.max_steer * (1.0 + 0.5 * abs(curvature))  # Increased authority in turns
            steering = np.clip(steering, -max_steer, max_steer)

            return steering

        except Exception as e:
            print(f"Error in steering calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _estimate_path_curvature(self):
        """Estimate the curvature of the upcoming path section"""
        try:
            if self.current_waypoint_index >= len(self.waypoints) - 2:
                return 0.0

            # Get three consecutive waypoints
            p1 = self.waypoints[self.current_waypoint_index].transform.location
            p2 = self.waypoints[self.current_waypoint_index + 1].transform.location
            p3 = self.waypoints[self.current_waypoint_index + 2].transform.location

            # Calculate vectors
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])

            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norms < 1e-6:
                return 0.0

            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Normalize curvature to [-1, 1] range
            curvature = angle / np.pi

            return curvature

        except Exception as e:
            print(f"Error calculating curvature: {str(e)}")
            return 0.0

    def _get_preview_points(self, base_lookahead):
        """Get preview points with tighter spacing for better control"""
        preview_points = []
        current_idx = self.current_waypoint_index

        # ENHANCED: Get more points with even closer spacing
        # Multipliers less than 1.0 will look at points closer than the base lookahead
        distances = [base_lookahead * mult for mult in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]]  # More points, even closer

        # Track the lane ID of the current waypoint for adherence
        current_lane_id = self.waypoints[current_idx].lane_id if current_idx < len(self.waypoints) else None

        for dist in distances:
            # Find waypoint at approximately this distance
            accumulated_dist = 0
            idx = current_idx

            while idx < len(self.waypoints) - 1:
                wp1 = self.waypoints[idx].transform.location
                wp2 = self.waypoints[idx + 1].transform.location
                segment_dist = wp1.distance(wp2)

                if accumulated_dist + segment_dist >= dist:
                    # Interpolate point at exact distance
                    remaining = dist - accumulated_dist
                    fraction = remaining / segment_dist
                    x = wp1.x + fraction * (wp2.x - wp1.x)
                    y = wp1.y + fraction * (wp2.y - wp1.y)
                    z = wp1.z + fraction * (wp2.z - wp1.z)
                    preview_points.append(carla.Location(x, y, z))
                    break
                
                accumulated_dist += segment_dist
                idx += 1

            if idx >= len(self.waypoints) - 1 and len(self.waypoints) > 0:
                preview_points.append(self.waypoints[-1].transform.location)

        return preview_points

    def _update_waypoint_progress(self, vehicle_location):
        """Update progress along waypoints with more strict adherence to path order"""
        if self.current_waypoint_index >= len(self.waypoints):
            return

        current_wp = self.waypoints[self.current_waypoint_index]
        distance = vehicle_location.distance(current_wp.transform.location)

        # Only update waypoint if we're close enough
        if distance < self.waypoint_distance_threshold:
            # ENHANCED: Look ahead to see if the next waypoint is closer
            if self.current_waypoint_index < len(self.waypoints) - 1:
                next_wp = self.waypoints[self.current_waypoint_index + 1]
                next_distance = vehicle_location.distance(next_wp.transform.location)

                # Only advance if we're getting closer to the next waypoint
                if next_distance < distance + 0.5:  # Added small tolerance
                    self.visited_waypoints.add(self.current_waypoint_index)
                    self.current_waypoint_index = min(self.current_waypoint_index + 1, 
                                                   len(self.waypoints) - 1)


    def _reset_control_state(self):
        """Reset control state after aggressive braking"""
        self.last_control = None
        # Reset any accumulated steering
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)

    def force_path_recovery(self, vehicle):
        """Force vehicle back to path with enhanced recovery behavior"""
        if not self.waypoints:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        try:
            # Get current vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            vehicle_forward = vehicle_transform.get_forward_vector()
            current_speed = self._get_speed(vehicle)

            # Get current map info if not already available
            if self.map is None and vehicle.get_world() is not None:
                self.map = vehicle.get_world().get_map()

            # Get current vehicle's lane
            current_lane_waypoint = self.map.get_waypoint(vehicle_loc) if self.map else None
            current_lane_id = current_lane_waypoint.lane_id if current_lane_waypoint else None

            # Store original waypoint index for debugging
            original_index = self.current_waypoint_index

            # Safety check to avoid index errors
            if self.current_waypoint_index >= len(self.waypoints):
                self.current_waypoint_index = len(self.waypoints) - 1

            # ENHANCED: Expanded search window for recovery
            start_idx = max(0, self.current_waypoint_index - 10)  # Look further back
            end_idx = min(len(self.waypoints), self.current_waypoint_index + 20)

            # Find closest waypoint with multiple factors
            min_score = float('inf')
            best_idx = self.current_waypoint_index

            for i in range(start_idx, end_idx):
                wp = self.waypoints[i]
                wp_loc = wp.transform.location

                # ENHANCED: Multiple factors for waypoint selection
                # 1. Distance to waypoint
                distance = vehicle_loc.distance(wp_loc)

                # 2. Direction alignment (prefer waypoints we're facing)
                wp_forward = wp.transform.get_forward_vector()
                dot_product = wp_forward.x * vehicle_forward.x + wp_forward.y * vehicle_forward.y
                direction_factor = 1.0 - max(0, dot_product)  # 0 if perfect alignment, 1 if opposite

                # 3. Progress factor (prefer waypoints that are ahead in the path)
                progress_penalty = max(0, self.current_waypoint_index - i) * 0.5  # Penalty for going backwards

                # Combine factors - weighting distance most heavily
                combined_score = distance + (direction_factor * 3.0) + progress_penalty

                if combined_score < min_score:
                    min_score = combined_score
                    best_idx = i

            # Update waypoint index
            self.current_waypoint_index = best_idx

            # Get target waypoint
            target_wp = self.waypoints[self.current_waypoint_index]
            target_loc = target_wp.transform.location

            # ENHANCED: Calculate stronger corrective steering
            # Vector from vehicle to target
            to_target = carla.Vector3D(
                target_loc.x - vehicle_loc.x,
                target_loc.y - vehicle_loc.y,
                0
            )

            # Normalize vector
            distance_to_target = math.sqrt(to_target.x**2 + to_target.y**2)
            if distance_to_target > 0.1:
                to_target.x /= distance_to_target
                to_target.y /= distance_to_target

            # Calculate direct steering to target
            vehicle_right = carla.Vector3D(
                math.cos(math.radians(vehicle_transform.rotation.yaw + 90)),
                math.sin(math.radians(vehicle_transform.rotation.yaw + 90)),
                0
            )

            # Cross product for direct steering
            direct_steer = to_target.x * vehicle_right.x + to_target.y * vehicle_right.y

            # Calculate heading correction
            heading_correction = self._calculate_steering(vehicle_transform, target_wp.transform)

            # ENHANCED: Combine direct and path-based steering with stronger weight to direct
            steer = 0.7 * direct_steer + 0.3 * heading_correction

            # ENHANCED: More aggressive recovery with hard speed limits
            if distance_to_target > 5.0:  # Very far from path
                max_recovery_speed = 5.0  # Very slow for major corrections
            else:
                max_recovery_speed = 10.0  # Slightly faster for minor corrections

            # Speed control
            if current_speed > max_recovery_speed:
                throttle = 0.0
                brake = 0.8  # Strong braking to get to safe speed
            elif current_speed < 2.0:  # Too slow
                throttle = 0.4
                brake = 0.0
            else:
                # Maintain controlled speed during recovery
                throttle = 0.3
                brake = 0.0

            # Limit steering for stability
            steer = np.clip(steer, -0.8, 0.8)

            # Create recovery control
            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False
            )

            # Store last control for reference
            self.last_control = control

            # Store the last recovery time for rate limiting
            self.last_recovery_time = time.time()

            # Debug visualization
            try:
                debug = vehicle.get_world().debug
                debug.draw_line(
                    vehicle_loc,
                    target_loc,
                    thickness=0.5,
                    color=carla.Color(255, 0, 255),  # Magenta for recovery
                    life_time=0.5
                )

                # Print debug info
                print(f"RECOVERY MODE: Target WP idx: {best_idx}, " +
                      f"Original idx: {original_index}, " +
                      f"Distance: {distance_to_target:.2f}m, " +
                      f"Speed: {current_speed:.2f}km/h, " +
                      f"Controls - Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}")
            except:
                pass
            
            return control
        except Exception as e:
            print(f"Error in path recovery: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._reset_control_state()
    
    def _get_speed(self, vehicle):
        """Get current speed in km/h"""
        vel = vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2)
    
    # New Visualization Functions
    def _visualize(self, world, vehicle):
        """Visualize real-time progress along the A* path"""
        if not self.waypoints:
            return
            
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
                target.transform.location + carla.Location(z=1.0),
                size=0.1,
                color=carla.Color(0, 255, 255),  # Cyan
                life_time=self.debug_lifetime
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
            f"Waypoints remaining: {len(self.waypoints) - self.current_waypoint_index}"
        ]
        
        for i, text in enumerate(info_text):
            world.debug.draw_string(
                current_loc + carla.Location(z=2.0 + i * 0.5),  # Stack text vertically
                text,
                color=carla.Color(255, 255, 255),
                life_time=self.debug_lifetime
            )

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

    def update_obstacles(self, obstacles):
        """Update list of detected obstacles"""
        self.obstacles = obstacles