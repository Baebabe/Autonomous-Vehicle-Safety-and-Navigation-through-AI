import os
import sys
import time
import numpy as np
import pygame
import carla
import random
import math

class SafetyController:
    def __init__(self, parent_actor, world, controller):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True

        # MODIFIED: Adjusted detection parameters for more realistic stopping
        self.detection_distance = 70.0  # Keep detection distance high (70m)
        self.lane_width = 3.5
        self.last_detected = False

        self.recovery_mode = False
        self.recovery_start_time = None
        self.max_recovery_time = 5.0  # INCREASED from 3s to 5s to give more time to recover
        self.min_speed_threshold = 0.5  # m/s
        self.last_brake_time = None
        self.brake_cooldown = 1.0  # seconds

        # MODIFIED: Braking parameters for more realistic stopping
        self.time_to_collision_threshold = 5.0
        self.min_safe_distance = 8.0  # REDUCED from 15m to 8m
        self.emergency_brake_distance = 7.0  # REDUCED from 7m to 5m

        # MODIFIED: Gradual braking parameters with closer distances
        self.target_stop_distance = 5.0  # Stop 5m from obstacles, not 30m
        self.deceleration_start_distance = 30.0  # REDUCED from 40m to 30m
        self.traffic_light_slowdown_distance = 50.0  # Increased from 25.0 to 50.0
        self.deceleration_profile = [  # Extended for gradual braking from further away
            (50.0, 0.02),  # Very light braking at 50m
            (40.0, 0.03),
            (30.0, 0.05),
            (20.0, 0.1),
            (15.0, 0.2),
            (10.0, 0.5),
            (8.0, 0.8),
            (5.0, 1.0)
        ]

        # MODIFIED: Traffic light parameters
        self.traffic_light_detection_distance = 100.0  # INCREASED from 70m to 100m
        self.traffic_light_detection_angle = 90.0  # INCREASED from 45° to 60° for better detection
        self.yellow_light_threshold = 2.5  # REDUCED from 3s to 2.5s
        self.traffic_light_stop_distance = 3.0  # Keep at 3m
        self.override_timeout = 15.0  # INCREASED from 10s to 15s for more reliable stopping
        self.last_tl_brake_time = None
        self.tl_override_start_time = None
        self.is_tl_override_active = False
        self.green_light_resume_attempts = 0
        self.max_green_light_resume_attempts = 5  # INCREASED from 3 to 5 attempts
        self.last_green_light_time = None
        self.green_light_grace_period = 0.3  # REDUCED from 0.5s to 0.3s for faster response
        self.last_tl_state = None
        self.stuck_detection_timeout = 10.0  # NEW: Detect if we're stuck at a green light
        self.stuck_at_light = False  # NEW: Flag for stuck state

        # Create collision sensor
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        print("Enhanced Safety controller initialized with improved obstacle and traffic light handling")


    def _on_collision(self, event):
        """Collision event handler"""
        print("!!! COLLISION DETECTED !!!")
        self._emergency_stop()

    def _calculate_time_to_collision(self, ego_velocity, other_velocity, distance):
        """Calculate time to collision based on relative velocity"""
        # Get velocity magnitudes
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        other_speed = math.sqrt(other_velocity.x**2 + other_velocity.y**2)
        
        # If both vehicles are stationary or moving at same speed
        if abs(ego_speed - other_speed) < 0.1:
            return float('inf') if distance > self.emergency_brake_distance else 0
            
        # Calculate time to collision
        relative_speed = ego_speed - other_speed
        if relative_speed > 0:  # Only if we're moving faster than the other vehicle
            return distance / relative_speed
        return float('inf')

    def _is_in_same_lane(self, ego_location, ego_forward, other_location):
        """
        More accurate lane detection that handles curves by checking waypoints and road curvature
        """
        try:
            # Get the current waypoint of ego vehicle
            ego_waypoint = self._world.get_map().get_waypoint(ego_location)
            other_waypoint = self._world.get_map().get_waypoint(other_location)

            if not ego_waypoint or not other_waypoint:
                return True, 0  # Conservative approach if waypoints can't be found

            # Check if vehicles are on the same road and in the same lane
            same_road = (ego_waypoint.road_id == other_waypoint.road_id)
            same_lane = (ego_waypoint.lane_id == other_waypoint.lane_id)

            # Vector to other vehicle
            to_other = other_location - ego_location

            # Project onto forward direction for distance calculation
            forward_dist = (to_other.x * ego_forward.x + to_other.y * ego_forward.y)

            if forward_dist <= 0:  # Vehicle is behind
                return False, forward_dist

            # If they're on different roads/lanes, no need for further checks
            if not (same_road and same_lane):
                return False, forward_dist

            # For vehicles on the same lane, check if they're within reasonable distance
            # Get a series of waypoints ahead to account for road curvature
            next_waypoints = ego_waypoint.next(forward_dist)
            if not next_waypoints:
                return True, forward_dist  # Conservative approach if can't get waypoints

            # Find the closest waypoint to the other vehicle
            min_dist = float('inf')
            closest_wp = None

            for wp in next_waypoints:
                dist = wp.transform.location.distance(other_location)
                if dist < min_dist:
                    min_dist = dist
                    closest_wp = wp

            # If no waypoint found within reasonable distance, vehicles aren't in same lane
            if not closest_wp:
                return False, forward_dist

            # Check lateral distance from the predicted path
            # Use a more generous threshold in curves
            road_curvature = abs(ego_waypoint.transform.rotation.yaw - closest_wp.transform.rotation.yaw)
            lateral_threshold = self.lane_width * (0.5 + (road_curvature / 90.0) * 0.3)  # Increase threshold in curves

            is_in_lane = min_dist < lateral_threshold

            if self.debug and forward_dist > 0:
                color = carla.Color(0, 255, 0) if is_in_lane else carla.Color(128, 128, 128)
                self._world.debug.draw_line(
                    ego_location,
                    other_location,
                    thickness=0.1,
                    color=color,
                    life_time=0.1
                )
                # Draw predicted path for debugging
                for wp in next_waypoints:
                    self._world.debug.draw_point(
                        wp.transform.location,
                        size=0.1,
                        color=carla.Color(0, 0, 255),
                        life_time=0.1
                    )

            return is_in_lane, forward_dist

        except Exception as e:
            print(f"Lane check error: {str(e)}")
            return True, 0  # Conservative approach - assume in lane if error
    
    def _get_traffic_light_state(self):
        """
        Improved traffic light detection with better error handling and debugging
        """
        try:
            ego_location = self._parent.get_location()
            ego_transform = self._parent.get_transform()
            ego_forward = ego_transform.get_forward_vector()
            ego_waypoint = self._world.get_map().get_waypoint(ego_location)
            
            # Debug information for diagnosing detection issues
            if self.debug:
                self._world.debug.draw_arrow(
                    ego_location,
                    ego_location + ego_forward * 5.0,
                    thickness=0.2,
                    arrow_size=0.2,
                    color=carla.Color(0, 255, 255),
                    life_time=0.1
                )
                
            # Method 1: Direct API call (most reliable when available)
            traffic_light = self._parent.get_traffic_light()
            if traffic_light:
                light_location = traffic_light.get_location()
                distance = ego_location.distance(light_location)
                
                if self.debug:
                    print(f"Method 1: Direct API detected traffic light at {distance:.1f}m")
                    
                return traffic_light.get_state(), distance, traffic_light, light_location
            
            # Method 2: Current waypoint's traffic light
            if ego_waypoint and hasattr(ego_waypoint, 'get_traffic_light'):
                traffic_light = ego_waypoint.get_traffic_light()
                if traffic_light:
                    light_location = traffic_light.get_location()
                    distance = ego_location.distance(light_location)
                    
                    if self.debug:
                        print(f"Method 2: Waypoint detected traffic light at {distance:.1f}m")
                        
                    return traffic_light.get_state(), distance, traffic_light, light_location
            
            # Method 3: Scan upcoming waypoints with progress logging
            if ego_waypoint:
                next_wp = ego_waypoint
                distance_accumulated = 0
                scan_points = int(self.traffic_light_detection_distance / 2.0)
                
                if self.debug:
                    print(f"Method 3: Scanning {scan_points} waypoints ahead...")
                    
                for i in range(scan_points):
                    next_wps = next_wp.next(2.0)
                    if not next_wps:
                        if self.debug and i < 5:
                            print(f"Method 3: Path ends after {i} waypoints")
                        break
                        
                    next_wp = next_wps[0]
                    distance_accumulated += 2.0
                    
                    # Log waypoint positions for debugging
                    if self.debug and i % 5 == 0:
                        self._world.debug.draw_point(
                            next_wp.transform.location,
                            size=0.1,
                            color=carla.Color(0, 255, 0),
                            life_time=0.1
                        )
                    
                    # Check traffic light at this waypoint
                    if hasattr(next_wp, 'get_traffic_light'):
                        traffic_light = next_wp.get_traffic_light()
                        if traffic_light:
                            light_location = traffic_light.get_location()
                            distance = ego_location.distance(light_location)
                            
                            if self.debug:
                                print(f"Method 3: Found traffic light at waypoint {i}, distance {distance:.1f}m")
                                
                            return traffic_light.get_state(), distance, traffic_light, light_location
                    
                    # Check if at junction
                    if next_wp.is_junction:
                        if self.debug:
                            print(f"Method 3: Junction found at waypoint {i}")
            
            # Method 4: Direct search with improved angle calculation
            traffic_lights = self._world.get_actors().filter('traffic.traffic_light*')
            if self.debug:
                light_count = len(traffic_lights)
                print(f"Method 4: Scanning {light_count} traffic lights in the world")
                
            min_distance = float('inf')
            closest_light = None
            closest_light_loc = None
            
            for light in traffic_lights:
                light_loc = light.get_location()
                distance = ego_location.distance(light_loc)
                
                # Only check lights within detection distance
                if distance < self.traffic_light_detection_distance:
                    # Improved vector calculation
                    to_light = light_loc - ego_location
                    
                    # Normalize to avoid math errors
                    norm = math.sqrt(to_light.x**2 + to_light.y**2 + to_light.z**2)
                    if norm < 0.001:  # Avoid division by zero
                        continue
                        
                    to_light_normalized = carla.Vector3D(
                        to_light.x / norm,
                        to_light.y / norm,
                        to_light.z / norm
                    )
                    
                    # Calculate dot product with forward vector (cosine of angle)
                    forward_dot = ego_forward.x * to_light_normalized.x + ego_forward.y * to_light_normalized.y
                    
                    # Prevent math domain errors
                    clamped_dot = max(-1.0, min(1.0, forward_dot))
                    angle = math.acos(clamped_dot) * 180 / math.pi
                    
                    # Log all detected lights
                    if self.debug and angle < 90:  # Only log lights somewhat in front
                        self._world.debug.draw_line(
                            ego_location,
                            light_loc,
                            thickness=0.1,
                            color=carla.Color(255, 255, 0),
                            life_time=0.1
                        )
                        self._world.debug.draw_string(
                            light_loc + carla.Location(z=1.0),
                            f'Light: {angle:.1f}°, {distance:.1f}m',
                            color=carla.Color(255, 255, 0),
                            life_time=0.1
                        )
                    
                    # Check if light is within our detection angle and closer than any previously found
                    if angle < self.traffic_light_detection_angle and distance < min_distance:
                        # Additional verification for lane association
                        if ego_waypoint:
                            light_waypoint = self._world.get_map().get_waypoint(light_loc)
                            if light_waypoint and light_waypoint.road_id == ego_waypoint.road_id:
                                min_distance = distance
                                closest_light = light
                                closest_light_loc = light_loc
                                
                                if self.debug:
                                    print(f"Method 4: Verified light on same road {light_waypoint.road_id}, angle {angle:.1f}°")
                        else:
                            # If we don't have ego waypoint, be more permissive
                            min_distance = distance
                            closest_light = light
                            closest_light_loc = light_loc
            
            if closest_light:
                if self.debug:
                    print(f"Method 4: Found closest traffic light at {min_distance:.1f}m")
                    
                    # Draw debug visualization
                    color = carla.Color(255, 255, 0)  # Yellow by default
                    if closest_light.get_state() == carla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0)  # Red
                    elif closest_light.get_state() == carla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0)  # Green
                    
                    self._world.debug.draw_line(
                        ego_location,
                        closest_light_loc,
                        thickness=0.3,
                        color=color,
                        life_time=0.1
                    )
                    self._world.debug.draw_point(
                        closest_light_loc,
                        size=0.3,
                        color=color,
                        life_time=0.1
                    )
                    self._world.debug.draw_string(
                        closest_light_loc + carla.Location(z=2.0),
                        f'Traffic Light: {closest_light.get_state()}, {min_distance:.1f}m',
                        color=color,
                        life_time=0.1
                    )
                
                return closest_light.get_state(), min_distance, closest_light, closest_light_loc
            
            # No traffic light found
            if self.debug:
                print("No traffic light detected by any method")
                
            return None, float('inf'), None, None
            
        except Exception as e:
            print(f"ERROR in traffic light detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create a visual warning in the world
            if self.debug:
                self._world.debug.draw_string(
                    ego_location + carla.Location(z=3.0),
                    f'!!! TRAFFIC LIGHT DETECTION ERROR !!!',
                    color=carla.Color(255, 0, 0),
                    life_time=0.5
                )
                
            return None, float('inf'), None, None
    
    def _handle_traffic_light(self):
        """Enhanced traffic light handling with more robust stopping and resuming"""
        light_state, distance, light_actor, light_location = self._get_traffic_light_state()
        current_time = time.time()
        
        # Get current speed
        ego_velocity = self._parent.get_velocity()
        speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        speed_kmh = 3.6 * speed  # Convert to km/h for display
        
        # NEW: Store the last traffic light state for comparison
        previous_state = self.last_tl_state
        self.last_tl_state = light_state
        
        # ENHANCED: If no traffic light is detected, check if we need to release override
        if light_state is None or distance == float('inf'):
            # Reset traffic light override if we're no longer detecting a light
            if self.is_tl_override_active:
                # NEW: Check if we're stuck (stopped but no light)
                if speed < 0.5 and current_time - self.tl_override_start_time > self.stuck_detection_timeout:
                    print("WARNING: Stuck in traffic with no light detected. Forcing resume...")
                    self._force_resume_path()
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                    self.stuck_at_light = False
                    return False
                    
                if (current_time - self.tl_override_start_time > self.override_timeout):
                    print("Releasing traffic light override (timeout)")
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                    self._force_resume_path()  # Force resume
            return False
        
        # Convert to carla.TrafficLightState object if needed
        if isinstance(light_state, str):
            if light_state == "Red":
                light_state = carla.TrafficLightState.Red
            elif light_state == "Yellow":
                light_state = carla.TrafficLightState.Yellow
            elif light_state == "Green":
                light_state = carla.TrafficLightState.Green
        
        # Handle different light states
        if light_state == carla.TrafficLightState.Red:
            # Red light - we should come to a stop
            if distance < self.traffic_light_slowdown_distance:
                # Reset green light counter
                self.green_light_resume_attempts = 0
                self.last_green_light_time = None
                self.stuck_at_light = False
                
                # Calculate stopping distance
                stopping_distance = distance - self.traffic_light_stop_distance
                
                # Activate traffic light override
                if not self.is_tl_override_active:
                    self.is_tl_override_active = True
                    self.tl_override_start_time = time.time()
                    print(f"\n!!! RED LIGHT DETECTED - Distance: {distance:.1f}m, Speed: {speed_kmh:.1f} km/h !!!")
                
                if stopping_distance <= 0:
                    # We've reached or passed the stopping point
                    self._emergency_stop()
                    print(f"RED LIGHT STOP - Distance: {distance:.1f}m")
                else:
                    # Apply gradual braking based on distance
                    brake_intensity = self._calculate_brake_intensity(stopping_distance)
                    
                    # Increase braking intensity for red lights
                    brake_intensity = min(1.0, brake_intensity * 1.5)
                    
                    print(f"RED LIGHT BRAKING - Distance: {distance:.1f}m, Brake: {brake_intensity:.2f}")
                    control = carla.VehicleControl(
                        throttle=0.0,
                        brake=brake_intensity,
                        steer=self._maintain_path_steer(),  # Keep steering while braking
                        hand_brake=False
                    )
                    self._parent.apply_control(control)
                    self.last_tl_brake_time = time.time()
                return True
        
        elif light_state == carla.TrafficLightState.Yellow:
            # ENHANCED Yellow light handling with more reliable stopping
            time_to_light = distance / max(speed, 0.1)  # Avoid division by zero
            
            # MODIFIED: More aggressive yellow light handling
            if time_to_light > self.yellow_light_threshold or distance < 15.0:
                # We can't clear the intersection in time, stop
                if distance < self.traffic_light_slowdown_distance:
                    # Reset green light counter
                    self.green_light_resume_attempts = 0
                    self.last_green_light_time = None
                    self.stuck_at_light = False
                    
                    # Activate traffic light override
                    if not self.is_tl_override_active:
                        self.is_tl_override_active = True
                        self.tl_override_start_time = time.time()
                        print(f"\n!!! YELLOW LIGHT DETECTED (stopping) - Distance: {distance:.1f}m, Time: {time_to_light:.1f}s !!!")
                    
                    # Calculate stopping distance
                    stopping_distance = distance - self.traffic_light_stop_distance
                    
                    if stopping_distance <= 0:
                        # We've reached or passed the stopping point
                        self._emergency_stop()
                        print(f"YELLOW LIGHT STOP - Distance: {distance:.1f}m")
                    else:
                        # Apply gradual braking based on distance
                        brake_intensity = self._calculate_brake_intensity(stopping_distance)
                        # INCREASED braking for yellow lights
                        brake_intensity = min(1.0, brake_intensity * 1.3)
                        
                        print(f"YELLOW LIGHT BRAKING - Distance: {distance:.1f}m, Brake: {brake_intensity:.2f}")
                        control = carla.VehicleControl(
                            throttle=0.0,
                            brake=brake_intensity,
                            steer=self._maintain_path_steer(),  # Keep steering while braking
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                        self.last_tl_brake_time = time.time()
                    return True
            else:
                # We can clear the intersection, proceed with caution
                if self.is_tl_override_active:
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                print(f"YELLOW LIGHT (proceeding) - Distance: {distance:.1f}m, Time: {time_to_light:.1f}s")
                return False
        
        elif light_state == carla.TrafficLightState.Green:
            # ENHANCED: Improved green light handling with stuck detection
            
            # Check if we're coming from a different state to green
            state_change_to_green = (previous_state != carla.TrafficLightState.Green and 
                                    light_state == carla.TrafficLightState.Green)
            
            # Track when we first see the green light
            if state_change_to_green:
                self.last_green_light_time = current_time
                print(f"\n!!! GREEN LIGHT DETECTED - Distance: {distance:.1f}m !!!")
            
            # If we were stopped for a red/yellow light and now it's green
            if self.is_tl_override_active:
                # Wait for a short grace period before trying to resume
                if self.last_green_light_time and (current_time - self.last_green_light_time) >= self.green_light_grace_period:
                    # Increment resume attempt counter
                    self.green_light_resume_attempts += 1
                    
                    # ENHANCED: Check if we're stuck at a green light
                    if speed < 0.5 and self.green_light_resume_attempts > 2:
                        self.stuck_at_light = True
                        
                    print(f"GREEN LIGHT - Resuming operation (attempt {self.green_light_resume_attempts})")
                    
                    # NEW: Apply increasing throttle based on number of attempts
                    throttle_value = min(0.7 + (self.green_light_resume_attempts * 0.05), 1.0)
                    
                    # Force resume with stronger throttle if we're stuck
                    if self.stuck_at_light:
                        control = carla.VehicleControl(
                            throttle=1.0,  # Maximum throttle to get unstuck
                            brake=0.0,
                            steer=self._maintain_path_steer(),
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                        print("STUCK AT GREEN LIGHT - Applying maximum throttle")
                    else:
                        # Normal resume
                        control = carla.VehicleControl(
                            throttle=throttle_value,
                            brake=0.0,
                            steer=self._maintain_path_steer(),
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                    
                    # Only release override after we're moving or max attempts reached
                    if speed > 1.0 or self.green_light_resume_attempts >= self.max_green_light_resume_attempts:
                        self.is_tl_override_active = False
                        self.tl_override_start_time = None
                        self.green_light_resume_attempts = 0
                        self.stuck_at_light = False
                        print(f"GREEN LIGHT - Successfully resumed normal operation")
                    
                    # Always return False on green to allow controller to work
                    return False
            else:
                # Already moving - no need to do anything special
                self.green_light_resume_attempts = 0
                self.stuck_at_light = False
            
            return False
        
        # Reset override for unknown light state
        if self.is_tl_override_active:
            self.is_tl_override_active = False
            self.tl_override_start_time = None
        
        return False
    
    def _force_resume_path(self):
        """Force vehicle to resume movement after stopping for a traffic light"""
        try:
            # Reset brake and recovery state
            self.last_brake_time = None
            self.recovery_mode = False
            self.recovery_start_time = None

            # ENHANCED: Get current velocity to determine needed thrust
            ego_velocity = self._parent.get_velocity()
            current_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # Apply stronger initial acceleration when completely stopped
            throttle_value = 0.9 if current_speed < 0.5 else 0.7

            control = carla.VehicleControl(
                throttle=throttle_value,
                brake=0.0,
                steer=self._maintain_path_steer(),
                hand_brake=False
            )
            self._parent.apply_control(control)

            # NEW: Apply control multiple times to overcome inertia
            for _ in range(3):
                self._parent.apply_control(control)
                time.sleep(0.05)  # Short delay between control applications

            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()

        except Exception as e:
            print(f"Error forcing resume: {str(e)}")
    
    def _maintain_path_steer(self):
        """Get steering value to maintain path while braking"""
        try:
            if not self._controller:
                return 0.0
                
            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    vehicle_transform = self._parent.get_transform()
                    
                    if hasattr(self._controller, '_calculate_steering'):
                        return self._controller._calculate_steering(vehicle_transform, target_wp.transform)
            
            return 0.0
        
        except Exception as e:
            print(f"Error calculating steering: {str(e)}")
            return 0.0
    
    def _calculate_brake_intensity(self, distance):
        """Calculate brake intensity based on distance to obstacle/traffic light"""
        # Find the appropriate braking level from the deceleration profile
        for dist, brake in self.deceleration_profile:
            if distance <= dist:
                return brake
        
        return 0.0  # No braking needed
    
    def _apply_gradual_braking(self, distance, ttc):
        """Apply gradual braking based on distance and time to collision"""
        # ENHANCED: Emergency stop for very close obstacles
        if distance < 3.0 or ttc < 0.5:
            print(f"!!! EMERGENCY STOP !!! Distance: {distance:.1f}m, TTC: {ttc:.1f}s")
            self._emergency_stop()
            return True

        # MODIFIED: More realistic and appropriate braking intensity
        brake_intensity = self._calculate_brake_intensity(distance)

        # If TTC is very small, increase braking
        if ttc < 2.0:
            brake_intensity = max(brake_intensity, 0.8)
        elif ttc < 3.0:
            brake_intensity = max(brake_intensity, 0.5)

        if brake_intensity > 0:
            ego_velocity = self._parent.get_velocity()
            speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)  # km/h

            # NEW: Don't brake too hard at moderate distances
            if distance > 15.0 and brake_intensity > 0.2:
                brake_intensity = 0.2

            print(f"Gradual braking - Distance: {distance:.1f}m, TTC: {ttc:.1f}s, Speed: {speed:.1f} km/h, Brake: {brake_intensity:.2f}")

            # ENHANCED: Apply some throttle at long distances to prevent premature stopping
            throttle = 0.0
            if distance > 20.0 and brake_intensity < 0.3:
                throttle = 0.1

            control = carla.VehicleControl(
                throttle=throttle,
                brake=brake_intensity,
                steer=self._maintain_path_steer(),  # Keep steering while braking
                hand_brake=False
            )
            self._parent.apply_control(control)
            return True

        return False
    
    def check_safety(self):
        """Enhanced safety checking with prioritized traffic light handling"""
        try:
            # First check traffic lights - THIS MUST TAKE PRIORITY
            if self._handle_traffic_light():
                # If we're handling a traffic light, skip other checks
                return
    
            # Only check for other vehicles if we're not dealing with a traffic light
            all_vehicles = self._world.get_actors().filter('vehicle.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_velocity = self._parent.get_velocity()
    
            # NEW: Also check for pedestrians
            all_pedestrians = self._world.get_actors().filter('walker.*')
    
            detected_obstacles = []
            min_distance = float('inf')
            min_ttc = float('inf')  # Time to collision
    
            # Current speed in km/h
            speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
    
            # Check for vehicles
            for vehicle in all_vehicles:
                if vehicle.id == self._parent.id:
                    continue
                
                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)
    
                # Check vehicles within detection range
                if distance < self.detection_distance:
                    is_in_lane, forward_dist = self._is_in_same_lane(ego_location, ego_forward, vehicle_location)
    
                    if is_in_lane and forward_dist > 0:  # Only consider vehicles ahead
                        other_velocity = vehicle.get_velocity()
                        ttc = self._calculate_time_to_collision(ego_velocity, other_velocity, forward_dist)
    
                        detected_obstacles.append((vehicle, forward_dist, ttc))
                        min_distance = min(min_distance, forward_dist)
                        min_ttc = min(min_ttc, ttc)
    
                        if self.debug:
                            self._world.debug.draw_box(
                                vehicle.bounding_box,
                                vehicle.get_transform().rotation,
                                thickness=0.5,
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )
                            self._world.debug.draw_string(
                                vehicle_location + carla.Location(z=2.0),
                                f'!!! {forward_dist:.1f}m, TTC: {ttc:.1f}s !!!',
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )
    
            # FIXED: Improved pedestrian detection to ignore those on footpaths
            for pedestrian in all_pedestrians:
                pedestrian_location = pedestrian.get_location()
                distance = ego_location.distance(pedestrian_location)
    
                # Use a narrower detection range for pedestrians
                if distance < self.detection_distance:
                    # Get the waypoint for the pedestrian to check if they're on a sidewalk
                    pedestrian_waypoint = self._world.get_map().get_waypoint(pedestrian_location, 
                                                                             lane_type=carla.LaneType.Driving)
                    
                    # Get the distance from pedestrian to nearest driving lane
                    if pedestrian_waypoint:
                        sidewalk_distance = pedestrian_location.distance(pedestrian_waypoint.transform.location)
                    else:
                        sidewalk_distance = float('inf')
                    
                    # Only consider pedestrians that are close to or on the road
                    # Typical sidewalk width is 1.5-2m, so pedestrians >2m from the road are on footpaths
                    if sidewalk_distance < 2.0:
                        to_ped = pedestrian_location - ego_location
                        forward_dot = ego_forward.x * to_ped.x + ego_forward.y * to_ped.y
    
                        # Only consider pedestrians ahead of us
                        if forward_dot > 0:
                            # Calculate lateral distance
                            lateral_vector = to_ped - ego_forward * forward_dot
                            lateral_distance = math.sqrt(lateral_vector.x**2 + lateral_vector.y**2)
    
                            # Only consider pedestrians that are on or very close to our driving lane
                            if lateral_distance < self.lane_width:
                                other_velocity = pedestrian.get_velocity()
                                ttc = self._calculate_time_to_collision(ego_velocity, other_velocity, forward_dot)
    
                                detected_obstacles.append((pedestrian, forward_dot, ttc))
                                min_distance = min(min_distance, forward_dot)
                                min_ttc = min(min_ttc, ttc)
    
                                if self.debug:
                                    self._world.debug.draw_point(
                                        pedestrian_location,
                                        size=0.2,
                                        color=carla.Color(255, 0, 0, 255),
                                        life_time=0.1
                                    )
                                    self._world.debug.draw_string(
                                        pedestrian_location + carla.Location(z=2.0),
                                        f'!!! Pedestrian on road {forward_dot:.1f}m, TTC: {ttc:.1f}s !!!',
                                        color=carla.Color(255, 0, 0, 255),
                                        life_time=0.1
                                    )
                    elif self.debug:
                        # Visual debug for pedestrians that are safely on footpaths
                        self._world.debug.draw_point(
                            pedestrian_location,
                            size=0.1,
                            color=carla.Color(0, 255, 0, 255),  # Green for safe pedestrians
                            life_time=0.1
                        )
                        self._world.debug.draw_string(
                            pedestrian_location + carla.Location(z=2.0),
                            f'Safe: {sidewalk_distance:.1f}m from road',
                            color=carla.Color(0, 255, 0, 255),
                            life_time=0.1
                        )
    
            if detected_obstacles:
                print(f"\nObstacle detected - Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s, Speed: {speed:.1f} km/h")
    
                # Apply gradual braking based on distance and TTC
                if self._apply_gradual_braking(min_distance, min_ttc):
                    self.last_detected = True
                else:
                    # Not braking, proceed normally
                    if self.last_detected:
                        print("Path clear - resuming normal operation")
                        self._resume_path()
                    self.last_detected = False
            else:
                if self.last_detected:
                    print("Path clear - resuming normal operation")
                    self._resume_path()
                self.last_detected = False
    
            # Update controller with obstacles
            if self._controller and hasattr(self._controller, 'update_obstacles'):
                self._controller.update_obstacles([v[0].get_location() for v in detected_obstacles])
    
        except Exception as e:
            print(f"Error in safety check: {str(e)}")
            import traceback
            traceback.print_exc()
            self._emergency_stop()

    def _emergency_stop(self):
        """Maximum braking force"""
        control = carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
            steer=self._maintain_path_steer()  # Keep steering while emergency braking
        )
        self._parent.apply_control(control)
        self.last_brake_time = time.time()

    def _maintain_path(self):
        """Enhanced path maintenance during braking"""
        try:
            if not self._controller:
                return
                
            current_time = time.time()
            
            # Check if we need to enter recovery mode
            if not self.recovery_mode:
                velocity = self._parent.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                
                if speed < self.min_speed_threshold:
                    self.recovery_mode = True
                    self.recovery_start_time = current_time
                    print("Entering path recovery mode")
            
            # Handle recovery mode
            if self.recovery_mode:
                if current_time - self.recovery_start_time > self.max_recovery_time:
                    self.recovery_mode = False
                    print("Exiting recovery mode - timeout")
                else:
                    # Force path recovery
                    if hasattr(self._controller, 'force_path_recovery'):
                        control = self._controller.force_path_recovery(self._parent)
                        self._parent.apply_control(control)
                    return
            
            # Normal path maintenance
            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    vehicle_transform = self._parent.get_transform()
                    
                    if hasattr(self._controller, '_calculate_steering'):
                        steer = self._controller._calculate_steering(vehicle_transform, target_wp.transform)
                        
                        # Get current control and maintain brake while updating steering
                        control = self._parent.get_control()
                        control.steer = steer
                        self._parent.apply_control(control)
            
        except Exception as e:
            print(f"Error in path maintenance: {str(e)}")

    def _resume_path(self):
        """Improved path resumption after braking with better path memory"""
        try:
            current_time = time.time()

            # Check if we're still in brake cooldown
            if self.last_brake_time and current_time - self.last_brake_time < self.brake_cooldown:
                # Maintain current path but release brake gradually
                control = self._parent.get_control()
                control.brake *= 0.3  # Aggressive brake release
                control.throttle = 0.2  # Add some throttle to ensure movement
                control.steer = self._maintain_path_steer()  # Crucial: maintain correct steering
                self._parent.apply_control(control)
                return

            # Reset recovery mode
            self.recovery_mode = False
            self.recovery_start_time = None

            # NEW: Store the last known path index before stopping
            if hasattr(self._controller, 'current_waypoint_index') and hasattr(self._controller, 'last_valid_waypoint_index'):
                if self._controller.current_waypoint_index < self._controller.last_valid_waypoint_index:
                    print(f"Restoring waypoint index from {self._controller.current_waypoint_index} to {self._controller.last_valid_waypoint_index}")
                    self._controller.current_waypoint_index = self._controller.last_valid_waypoint_index

            # NEW: If car seems stuck (not moving), apply more thrust but maintain steering direction
            ego_velocity = self._parent.get_velocity()
            speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            if speed < 0.5:  # Car is basically stopped
                # NEW: Get the target waypoint for correct steering direction
                steer = self._maintain_path_steer()

                control = carla.VehicleControl(
                    throttle=0.7,  # Strong throttle to get moving
                    brake=0.0,
                    steer=steer,  # Use the correct steering value from path
                    hand_brake=False
                )
                self._parent.apply_control(control)
                print("Vehicle stopped - applying extra throttle to resume while maintaining path direction")

                # Apply control multiple times to overcome inertia while keeping steering direction
                for _ in range(3):
                    control.throttle *= 0.9  # Slightly reduce throttle each time
                    self._parent.apply_control(control)
                    time.sleep(0.05)  # Short delay between control applications

            # NEW: Check if we need path recovery after stopping
            elif hasattr(self._controller, 'force_path_recovery') and hasattr(self._controller, 'waypoints'):
                # Get current vehicle position
                vehicle_transform = self._parent.get_transform()
                vehicle_loc = vehicle_transform.location

                # Check if we're far from our intended path
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    current_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    distance_to_path = vehicle_loc.distance(current_wp.transform.location)

                    if distance_to_path > 2.0:  # If we're more than 2 meters from the path
                        print(f"Vehicle deviated from path after stop ({distance_to_path:.2f}m) - initiating path recovery")
                        recovery_control = self._controller.force_path_recovery(self._parent)
                        self._parent.apply_control(recovery_control)
                        return

            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()

            # Resume normal control
            if hasattr(self._controller, 'get_control'):
                control = self._controller.get_control(self._parent, self._world)
                self._parent.apply_control(control)

        except Exception as e:
            print(f"Error resuming path: {str(e)}")
            import traceback
            traceback.print_exc()

    def _force_resume_path(self):
        """Force vehicle to resume movement after stopping for a traffic light with path preservation"""
        try:
            # Reset brake and recovery state
            self.last_brake_time = None
            self.recovery_mode = False
            self.recovery_start_time = None

            # ENHANCED: Get current velocity to determine needed thrust
            ego_velocity = self._parent.get_velocity()
            current_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # NEW: Critical - Make sure we maintain correct steering direction when resuming
            steer = self._maintain_path_steer()

            # Apply stronger initial acceleration when completely stopped
            throttle_value = 0.9 if current_speed < 0.5 else 0.7

            control = carla.VehicleControl(
                throttle=throttle_value,
                brake=0.0,
                steer=steer,  # Use the correct steering value from path
                hand_brake=False
            )
            self._parent.apply_control(control)

            # NEW: Apply control multiple times to overcome inertia while keeping steering direction
            for _ in range(3):
                self._parent.apply_control(control)
                time.sleep(0.05)  # Short delay between control applications

            # NEW: Check if we need path recovery
            if current_speed < 0.5 and hasattr(self._controller, 'force_path_recovery'):
                print("Applying path recovery after traffic light")
                recovery_control = self._controller.force_path_recovery(self._parent)
                self._parent.apply_control(recovery_control)

            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()

        except Exception as e:
            print(f"Error forcing resume: {str(e)}")
            import traceback
            traceback.print_exc()

    def _maintain_path_steer(self):
        """Enhanced steering calculation to maintain path while braking"""
        try:
            if not self._controller:
                return 0.0

            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                # NEW: Look ahead for better steering when stopped
                look_ahead = 0
                if hasattr(self._parent, 'get_velocity'):
                    velocity = self._parent.get_velocity()
                    speed = math.sqrt(velocity.x**2 + velocity.y**2)
                    if speed < 1.0:  # If almost stopped, look further ahead
                        look_ahead = 1  # Look at least one waypoint ahead

                # Make sure we don't go out of bounds
                target_idx = min(self._controller.current_waypoint_index + look_ahead, 
                                len(self._controller.waypoints) - 1)

                target_wp = self._controller.waypoints[target_idx]
                vehicle_transform = self._parent.get_transform()

                if hasattr(self._controller, '_calculate_steering'):
                    # NEW: Store current location and waypoint for debugging
                    if self.debug:
                        ego_location = self._parent.get_location()
                        wp_location = target_wp.transform.location
                        self._world.debug.draw_line(
                            ego_location,
                            wp_location,
                            thickness=0.2,
                            color=carla.Color(0, 255, 0),
                            life_time=0.1
                        )

                    return self._controller._calculate_steering(vehicle_transform, target_wp.transform)

            return 0.0

        except Exception as e:
            print(f"Error calculating steering: {str(e)}")
            return 0.0
    def destroy(self):
        """Clean up sensors"""
        if hasattr(self, 'collision_sensor'):
            self.collision_sensor.destroy()