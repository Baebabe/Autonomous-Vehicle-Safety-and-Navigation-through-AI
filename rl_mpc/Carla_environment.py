import os
import sys
import glob
import time
import random
import numpy as np
import cv2
import weakref
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import requests 
import pygame
from pygame import surfarray
import traceback
from mpc import MPCController

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import threading

# Constants
IM_WIDTH = 1080
IM_HEIGHT = 720
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_spatial_info(bbox, class_id, image_width=IM_WIDTH, image_height=IM_HEIGHT, fov=90):
    """
    Calculate depth and relative position using bounding box and class-specific dimensions
    
    Returns:
        - depth: estimated distance to object in meters
        - confidence: confidence in the depth estimate
        - relative_angle: angle to object in degrees (0 = straight ahead, negative = left, positive = right)
        - normalized_x_pos: horizontal position in normalized coordinates (-1 to 1, where 0 is center)
        - lane_position: estimated lane position relative to ego vehicle (-1: left lane, 0: same lane, 1: right lane)
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    # Center of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Define typical widths for different object classes (in meters)
    REAL_WIDTHS = {
        0: 0.45,   # person - average shoulder width
        
        # Vehicles
        1: 0.8,    # bicycle - typical handlebar width
        2: 0.8,    # motorcycle - typical handlebar width
        3: 1.8,    # car - average car width
        4: 2.5,    # truck - average truck width
        5: 2.9,    # bus - average bus width
        6: 3.0,    # train - typical train car width
        
        # Outdoor objects
        7: 0.6,    # fire hydrant - typical width
        8: 0.3,    # stop sign - standard width
        9: 0.3,    # parking meter - typical width
        10: 0.4,   # bench - typical seat width
    }
    
    # Get real width based on class, default to car width if class not found
    real_width = REAL_WIDTHS.get(class_id, 1.8)
    
    # Calculate focal length using camera parameters
    focal_length = (image_width / 2) / np.tan(np.radians(fov / 2))
    
    # Calculate depth using similar triangles principle
    if bbox_width > 0:  # Avoid division by zero
        depth = (real_width * focal_length) / bbox_width
    else:
        depth = float('inf')
    
    # Horizontal position normalized to [-1, 1] where 0 is center
    normalized_x_pos = (center_x - (image_width / 2)) / (image_width / 2)
    
    # Calculate relative angle in degrees
    relative_angle = np.degrees(np.arctan2(normalized_x_pos * np.tan(np.radians(fov / 2)), 1))
    
    # Estimate lane position
    # This is a simple heuristic based on horizontal position and object width
    # More sophisticated lane detection would use road markings
    if abs(normalized_x_pos) < 0.2:
        # Object is roughly centered - likely in same lane
        lane_position = 0
    elif normalized_x_pos < 0:
        # Object is to the left
        lane_position = -1
    else:
        # Object is to the right
        lane_position = 1
    
    # For vehicles (class 1-6), refine lane estimation based on size and position
    if 1 <= class_id <= 6:
        # Calculate expected width at this depth if in same lane
        expected_width_in_px = (real_width * focal_length) / depth
        
        # Ratio of actual width to expected width if centered
        width_ratio = bbox_width / expected_width_in_px
        
        # If object seems too small for its position, might be in adjacent lane
        if width_ratio < 0.7 and abs(normalized_x_pos) < 0.4:
            # Object appears smaller than expected for this position
            if normalized_x_pos < 0:
                lane_position = -1
            else:
                lane_position = 1
    
    # Calculate confidence based on multiple factors
    size_confidence = min(1.0, bbox_width / (image_width * 0.5))  # Higher confidence for larger objects
    center_confidence = 1.0 - abs(normalized_x_pos)  # Higher confidence for centered objects
    aspect_confidence = min(1.0, bbox_height / (bbox_width + 1e-6) / 0.75)  # Expected aspect ratio
    
    # Combined confidence score
    confidence = (size_confidence * 0.5 + center_confidence * 0.3 + aspect_confidence * 0.2)
    
    return {
        'depth': depth,
        'confidence': confidence,
        'relative_angle': relative_angle,
        'normalized_x_pos': normalized_x_pos,
        'lane_position': lane_position
    }

class CarEnv:
    def __init__(self):
        # Existing attributes
        self.client = None
        self.world = None
        self.camera = None
        self.vehicle = None
        self.collision_hist = []
        self.collision_sensor = None
        self.yolo_model = None
        self.max_objects = 10
        self.last_location = None
        self.stuck_time = 0
        self.episode_start = 0
        self.display_lock = threading.Lock()
        self.current_surface = None
        self.controller = None
        self.front_camera = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []
        self.prev_speed=0.0
        self.prev_acc=0.0
        # Add control source tracking
        self.control_source = "MPC_FULL"    
        self.detection_lock = threading.Lock()
        self.latest_detections = None
        self.vehicle_lock = threading.Lock()
        pygame.init()
        self.display = None
        self.clock = None
        self.init_pygame_display()  

        self.camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),
            carla.Rotation(pitch=0)
        )   

        self.setup_world()
        self._init_yolo()

    def init_pygame_display(self):
        """Initialize pygame display with error handling"""
        try:
            if self.display is None:
                pygame.init()
                self.display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT))
                pygame.display.set_caption("CARLA + YOLO View")
                self.clock = pygame.time.Clock()
        except Exception as e:
            print(f"Error initializing pygame display: {e}")
            traceback.print_exc()

    def _process_image(self, weak_self, image):
        self = weak_self()
        if self is not None:
            try:
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                self.front_camera = array
    
                detections = self.process_yolo_detection(array, image.transform)
                with self.detection_lock:
                    self.latest_detections = detections
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
                for obj in detections:
                    x1 = int(obj['position'][0] - obj['bbox_width']/2)
                    y1 = int(obj['position'][1] - obj['bbox_height']/2)
                    x2 = int(obj['position'][0] + obj['bbox_width']/2)
                    y2 = int(obj['position'][1] + obj['bbox_height']/2)
    
                    scale_x = IM_WIDTH / image.width
                    scale_y = IM_HEIGHT / image.height
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
    
                    pygame.draw.rect(surface, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 2)
    
                    if pygame.font.get_init():
                        font = pygame.font.Font(None, 24)
                        label = f"{obj['class_name']}"
                        text = font.render(label, True, (255, 255, 255))
                        surface.blit(text, (x1, y1-20))
    
                # if pygame.font.get_init():
                #     font = pygame.font.Font(None, 24)
                #     control_text = font.render(f"Control: {self.control_source}", True, (255, 255, 255))
                #     surface.blit(control_text, (10, 10))
    
                with self.display_lock:
                    self.current_surface = surface
    
            except Exception as e:
                print(f"Error in image processing: {e}")
                traceback.print_exc()

    def cleanup_actors(self):
        try:
            print("Starting cleanup of actors...")
            # Stop and destroy sensors
            if hasattr(self, 'collision_sensor') and self.collision_sensor:
                if self.collision_sensor.is_listening:
                    self.collision_sensor.stop()
                if self.collision_sensor.is_alive:
                    self.collision_sensor.destroy()
                print("Collision sensor destroyed")
            if hasattr(self, 'camera') and self.camera:
                if self.camera.is_listening:
                    self.camera.stop()
                if self.camera.is_alive:
                    self.camera.destroy()
                print("Camera destroyed")
            # Destroy vehicle with lock
            with self.vehicle_lock:
                if self.vehicle and self.vehicle.is_alive:
                    self.vehicle.destroy()
                    print("Vehicle destroyed")
                self.vehicle = None

            # Clear references
            self.collision_sensor = None
            self.camera = None
            self.front_camera = None
            self.controller = None
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error cleaning up actors: {e}")
            import traceback
            traceback.print_exc()


    def setup_world(self):
        """Initialize CARLA world and settings"""
        try:
            print("Connecting to CARLA server...")
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(200.0)

            print("Getting world...")
            # Load world without map layer flags since they're not supported in 0.10.0
            self.world = self.client.load_world('Town10HD_Opt')
            am = self.client.get_available_maps()
            print("AVAILABLE MAPS ARE", am)

            # Manually remove parked vehicles instead of using unload_map_layer
            self._remove_parked_vehicles()

            # Set up traffic manager
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.global_percentage_speed_difference(10.0)

            # Set synchronous mode settings
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)

            # Wait for the world to stabilize
            for _ in range(10):
                self.world.tick()

            print("CARLA world setup completed successfully")

        except Exception as e:
            print(f"Error setting up CARLA world: {e}")
            raise   

    def _remove_parked_vehicles(self):
        """Manually remove parked vehicles from the world"""
        print("Removing parked vehicles...")
        # Get all vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')
        removed_count = 0

        for vehicle in vehicles:
            # You can determine if a vehicle is parked in different ways
            # For example, check if it's not moving or if it's not controlled by the traffic manager
            velocity = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

            # Simple check - if it's not moving, consider it parked
            if speed < 0.1:  # Almost stationary
                vehicle.destroy()
                removed_count += 1

        print(f"Removed {removed_count} parked vehicles")
        
    def _init_yolo(self):
        """Initialize YOLOv11 model from Ultralytics"""
        try:
            print("Loading YOLOv11 model...")
            
            # Define path to ultralytics YOLOv11
            ultralytics_path = r"C:\Users\msi\miniconda3\envs\sdcarAB\Carla-0.10.0-Win64-Shipping\PythonAPI\sdcar\ultralytics"
            if ultralytics_path not in sys.path:
                sys.path.append(ultralytics_path)
            
            # Define model path
            model_path = os.path.join(ultralytics_path, 'yolo11n.pt')
            
            # Check if the model exists, if not, download it
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}, downloading YOLOv11 model")
                self._download_yolov11_weights(model_path)
            
            # Import the YOLO model from Ultralytics package
            from ultralytics import YOLO
            
            print(f"Loading model from: {model_path}")
            self.yolo_model = YOLO(model_path)
            
            # Configure model settings
            self.yolo_model.conf = 0.25  # Confidence threshold
            self.yolo_model.iou = 0.45   # NMS IoU threshold
            
            print("YOLOv11 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            import traceback
            traceback.print_exc()
            raise   

    def _download_yolov11_weights(self, save_path):
        """Download YOLOv11 weights if they don't exist"""
        import requests
        
        # URL for YOLOv11 weights (you'll need to update this with the correct URL)
        # This is a placeholder URL - you'll need to replace it with the actual URL
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            print(f"Downloading YOLOv11 weights to {save_path}")
            response = requests.get(url, stream=True)
            
            # Check if the request was successful
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Print progress
                            if total_size > 0:
                                percent = downloaded * 100 / total_size
                                print(f"Download progress: {percent:.1f}%", end="\r")
                
                print(f"\nDownload complete! YOLOv11 weights saved to {save_path}")
                return True
            else:
                print(f"Failed to download YOLOv11 weights. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error downloading YOLOv11 weights: {e}")
            return False



    def process_yolo_detection(self, image, camera_transform):
        if image is None:
            return []
        try:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = self.yolo_model(img_bgr, verbose=False)
            objects = []

            with self.vehicle_lock:
                if self.vehicle is None:
                    return []
                ego_location = self.vehicle.get_location()
                ego_waypoint = self.world.get_map().get_waypoint(ego_location)
                ego_lane_id = ego_waypoint.lane_id if ego_waypoint else None

            # Get path waypoints solely from MPCController
            path_waypoints = None
            if hasattr(self, 'controller') and self.controller and hasattr(self.controller, 'waypoints'):
                path_waypoints = self.controller.waypoints

            # Camera intrinsic parameters
            image_width = IM_WIDTH
            image_height = IM_HEIGHT
            fov = 90
            focal_length = (image_width / 2) / np.tan(np.radians(fov / 2))
            cx = image_width / 2
            cy = image_height / 2

            # Get all CARLA actors for ground truth verification
            ego_location = self.vehicle.get_location()
            all_vehicles = self.world.get_actors().filter('vehicle.*')
            all_pedestrians = self.world.get_actors().filter('walker.*')
            all_actors = [
                actor for actor in list(all_vehicles) + list(all_pedestrians)
                if actor.get_location().distance(ego_location) < 50.0
            ]

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    cls = class_ids[i]
                    class_name = result.names[cls]

                    # Filter to only relevant classes
                    if class_name not in ['car', 'truck', 'bus', 'person']:
                        continue

                    # Get initial spatial info from bounding box
                    spatial_info = calculate_spatial_info([x1, y1, x2, y2], cls, image_width, image_height, fov)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    depth = spatial_info['depth']  # Initial depth estimate

                    # Calculate 3D point in camera coordinates
                    X = (center_x - cx) * depth / focal_length
                    Y = (center_y - cy) * depth / focal_length
                    Z = depth

                    # Transform to world coordinates
                    point_camera = carla.Vector3D(X, Y, Z)
                    point_world = camera_transform.transform(point_camera)

                    # Ground truth verification and correction
                    min_dist = float('inf')
                    closest_actor = None

                    for actor in all_actors:
                        actor_loc = actor.get_location()
                        dist = point_world.distance(actor_loc)

                        # Check if this actor is likely the detected object
                        threshold = min(depth * 0.3, 5.0)  # 30% of estimated depth or 5m max

                        if dist < min_dist and dist < threshold:
                            min_dist = dist
                            closest_actor = actor

                    # If we found a matching actor, use its ground truth information
                    if closest_actor:
                        true_distance = ego_location.distance(closest_actor.get_location())
                        depth_correction_factor = true_distance / depth if depth > 0 else 1.0

                        # Apply correction if within reasonable bounds
                        if 0.5 < depth_correction_factor < 2.0:
                            depth = true_distance
                            X = (center_x - cx) * depth / focal_length
                            Y = (center_y - cy) * depth / focal_length
                            Z = depth
                            point_camera = carla.Vector3D(X, Y, Z)
                            point_world = camera_transform.transform(point_camera)

                    # Enhanced lane position detection using lane IDs
                    # Enhanced lane position detection using lane IDs
                    lane_position = None
                    in_ego_path = False

                    if path_waypoints:
                        # Find the closest MPC waypoint to the object's position
                        closest_wp = None
                        min_dist = float('inf')
                        for wp in path_waypoints:
                            wp_loc = wp.transform.location if hasattr(wp, 'transform') else wp.location
                            dist = point_world.distance(wp_loc)
                            if dist < min_dist:
                                min_dist = dist
                                closest_wp = wp

                        if closest_wp:
                            # Get the waypoint at the object's actual location
                            object_wp = self.world.get_map().get_waypoint(point_world)
                            # Updated default lane width
                            lane_width = closest_wp.lane_width if hasattr(closest_wp, 'lane_width') else 3.5

                            if object_wp:
                                # Check if object is on the same road and lane as MPC waypoint
                                same_road = object_wp.road_id == closest_wp.road_id
                                same_lane = object_wp.lane_id == closest_wp.lane_id

                                # For same lane determination, we need both same road ID and lane ID
                                if same_road and same_lane:
                                    # Calculate lateral offset to confirm object is truly within lane boundaries
                                    object_loc = carla.Location(point_world.x, point_world.y, point_world.z)
                                    ego_transform = self.vehicle.get_transform()
                                    ego_forward = ego_transform.get_forward_vector()
                                    ego_right = ego_transform.get_right_vector()

                                    # Vector from ego to object
                                    to_object = object_loc - ego_location

                                    # Project onto ego's forward and right vectors
                                    forward_proj = to_object.x * ego_forward.x + to_object.y * ego_forward.y
                                    right_proj = to_object.x * ego_right.x + to_object.y * ego_right.y

                                    # Strict lane position - only if truly ahead and centered
                                    if forward_proj > 0 and abs(right_proj) < lane_width/2.5:  # Stricter than lane_width/2
                                        lane_position = 0
                                        # Object must be directly ahead to be in path
                                        in_ego_path = abs(right_proj) < lane_width/3.0  # Even stricter for in_path
                                    else:
                                        # Not precisely in our lane
                                        lane_position = 1 if right_proj > 0 else -1
                                        in_ego_path = False
                                else:
                                    # Different road or lane
                                    lane_position = 1 if object_wp.lane_id > closest_wp.lane_id else -1
                                    in_ego_path = False
                            else:
                                # Fallback to spatial info if waypoint mapping fails
                                lane_position = spatial_info['lane_position']
                                in_ego_path = False
                    else:
                        # Fallback when no MPC waypoints are available
                        object_wp = self.world.get_map().get_waypoint(point_world)
                        if object_wp and ego_waypoint:
                            # Check if object is on the same road and lane as ego vehicle
                            same_road = object_wp.road_id == ego_waypoint.road_id
                            same_lane = object_wp.lane_id == ego_waypoint.lane_id

                            # For same lane determination, we need both same road ID and lane ID
                            if same_road and same_lane:
                                # Calculate lateral offset to confirm object is truly within lane boundaries
                                object_loc = carla.Location(point_world.x, point_world.y, point_world.z)
                                ego_transform = self.vehicle.get_transform()
                                ego_forward = ego_transform.get_forward_vector()
                                ego_right = ego_transform.get_right_vector()

                                # Vector from ego to object
                                to_object = object_loc - ego_location

                                # Project onto ego's forward and right vectors
                                forward_proj = to_object.x * ego_forward.x + to_object.y * ego_forward.y
                                right_proj = to_object.x * ego_right.x + to_object.y * ego_right.y

                                # Strict lane position - only if truly ahead and centered
                                if forward_proj > 0 and abs(right_proj) < lane_width/2.5:  # Stricter than lane_width/2
                                    lane_position = 0
                                    # Object must be directly ahead to be in path
                                    in_ego_path = abs(right_proj) < lane_width/3.0  # Even stricter for in_path
                                else:
                                    # Not precisely in our lane
                                    lane_position = 1 if right_proj > 0 else -1
                                    in_ego_path = False
                            else:
                                # Different road or lane
                                lane_position = 1 if object_wp.lane_id > ego_waypoint.lane_id else -1
                                in_ego_path = False
                        else:
                            lane_position = spatial_info['lane_position']
                            in_ego_path = False

                    # Ground truth override if available
                    if closest_actor:
                        actor_wp = self.world.get_map().get_waypoint(closest_actor.get_location())
                        if actor_wp and ego_waypoint:
                            same_lane = (actor_wp.road_id == ego_waypoint.road_id and 
                                       actor_wp.lane_id == ego_waypoint.lane_id)
                            if same_lane:
                                lane_position = 0
                                # Only set in_ego_path True if within path corridor
                                in_ego_path = self._is_point_in_path_corridor(point_world, path_waypoints)
                            else:
                                lane_position = 1 if actor_wp.lane_id > ego_waypoint.lane_id else -1
                                in_ego_path = False

                    # Calculate time-to-collision
                    # Calculate time-to-collision
                    ttc = float('inf')
                    if self.vehicle and in_ego_path:
                        velocity = self.vehicle.get_velocity()
                        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                        if speed > 0.5:
                            ttc = depth / speed

                    # Create object dictionary
                    objects.append({
                        'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'depth': depth,
                        'depth_confidence': spatial_info['confidence'],
                        'relative_angle': spatial_info['relative_angle'],
                        'normalized_x_pos': spatial_info['normalized_x_pos'],
                        'lane_position': lane_position,
                        'in_ego_path': in_ego_path,
                        'time_to_collision': ttc,
                        'class': int(cls),
                        'class_name': class_name,
                        'detection_confidence': float(conf),
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        'risk_score': self._calculate_risk_score(
                            depth,
                            lane_position,
                            ttc,
                            int(cls),
                            in_ego_path
                        )
                    })

                # Sort by risk score and limit to max_objects
                objects.sort(key=lambda x: x['risk_score'], reverse=True)
                return objects[:self.max_objects]

        except Exception as e:
            print(f"Error in YOLOv11 detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _is_point_in_path_corridor(self, point, waypoints, corridor_width=3.0):  # Narrower corridor
        """
        Improved check if a point is within the corridor defined by the waypoints path.
        Stricter corridor to avoid false positives.
        """
        if not waypoints or len(waypoints) < 2:
            return False

        # Only consider waypoints ahead of the vehicle
        vehicle_location = self.vehicle.get_location()
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()

        # Convert to numpy arrays for easier vector operations
        vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_dir = np.array([vehicle_forward.x, vehicle_forward.y])
        vehicle_dir = vehicle_dir / np.linalg.norm(vehicle_dir)

        point_pos = np.array([point.x, point.y])
        point_vec = point_pos - vehicle_pos

        # Check if point is ahead of the vehicle
        if np.dot(point_vec, vehicle_dir) <= 0:
            return False  # Behind or beside the vehicle

        min_dist = float('inf')
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]

            wp1_loc = wp1.transform.location if hasattr(wp1, 'transform') else wp1.location
            wp2_loc = wp2.transform.location if hasattr(wp2, 'transform') else wp2.location

            # Use lane width from waypoint if available, or default width
            lane_width = wp1.lane_width if hasattr(wp1, 'lane_width') else corridor_width

            # Make corridor narrower to be more selective
            effective_corridor = lane_width * 0.85  # Narrower than the original

            segment_vec = carla.Vector3D(wp2_loc.x - wp1_loc.x, wp2_loc.y - wp1_loc.y, wp2_loc.z - wp1_loc.z)
            point_vec = carla.Vector3D(point.x - wp1_loc.x, point.y - wp1_loc.y, point.z - wp1_loc.z)

            segment_length = math.sqrt(segment_vec.x**2 + segment_vec.y**2 + segment_vec.z**2)
            if segment_length < 0.01:
                continue
            
            segment_vec.x /= segment_length
            segment_vec.y /= segment_length
            segment_vec.z /= segment_length

            projection = point_vec.x * segment_vec.x + point_vec.y * segment_vec.y + point_vec.z * segment_vec.z
            projection = max(0, min(segment_length, projection))

            closest_point = carla.Vector3D(
                wp1_loc.x + segment_vec.x * projection,
                wp1_loc.y + segment_vec.y * projection,
                wp1_loc.z + segment_vec.z * projection
            )

            dist = math.sqrt((point.x - closest_point.x)**2 + 
                            (point.y - closest_point.y)**2 + 
                            (point.z - closest_point.z)**2)

            if dist < min_dist:
                min_dist = dist

        return min_dist <= effective_corridor / 2

    def _calculate_risk_score(self, depth, lane_position, ttc, class_id, in_ego_path=False):
        """
        Calculate risk score for an object with higher emphasis on same-lane objects
        """
        risk = 0

        # Distance factor - more aggressive for close objects
        distance_factor = math.exp(-0.08 * depth) * 12  # Increased weight 
        risk += distance_factor

        # Lane position factor - much higher for same lane
        if lane_position == 0:  # Same lane
            lane_factor = 15    # Increased from 10 to 15
        elif lane_position in [-1, 1]:  # Adjacent lanes
            lane_factor = 2     # Reduced from 3 to 2
        else:
            lane_factor = 0
        risk += lane_factor

        # Time-to-collision factor - more aggressive
        if ttc < float('inf'):
            if ttc < 1.0:
                ttc_factor = 25      # Increased from 20 to 25
            elif ttc < 2.0:
                ttc_factor = 20      # Increased from 15 to 20
            elif ttc < 3.0:
                ttc_factor = 15      # Increased from 10 to 15
            elif ttc < 5.0:
                ttc_factor = 8       # Increased from 5 to 8
            else:
                ttc_factor = 4       # Increased from 3 to 4
            risk += ttc_factor

        # Class factor - unchanged
        if class_id == 0:  # person
            class_factor = 8
        elif class_id in [3, 4, 5]:  # car, truck, bus
            class_factor = 5
        else:
            class_factor = 3
        risk += class_factor

        # Path factor - increased for in-path objects
        if in_ego_path:
            path_factor = 25    # Increased from 20 to 25
            risk += path_factor

        return risk

     
    def get_waypoint_info(self):
        """Get waypoint information"""
        location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(location)
        
        # Calculate distance and angle to waypoint
        distance = location.distance(waypoint.transform.location)
        
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        waypoint_forward = waypoint.transform.get_forward_vector()
        
        dot = vehicle_forward.x * waypoint_forward.x + vehicle_forward.y * waypoint_forward.y
        cross = vehicle_forward.x * waypoint_forward.y - vehicle_forward.y * waypoint_forward.x
        angle = math.atan2(cross, dot)
        
        return {
            'distance': distance,
            'angle': angle
        }
    
    def get_vehicle_state(self):
        """Get vehicle state information"""
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h
        steering = self.vehicle.get_control().steer
        
        return {
            'speed': speed,
            'steering': steering
        }
    
    def get_state(self):
        """Get state for RL focusing on throttle/brake control"""
        if self.front_camera is None:
            return None

        state_array = []
        with self.detection_lock:
            detections = self.latest_detections if self.latest_detections is not None else []

        for obj in detections:
            state_array.extend([
                obj['normalized_x_pos'],
                obj['depth'] / 100.0,
                obj['relative_angle'] / 90.0,
                float(obj['lane_position']),
                min(1.0, 10.0 / max(0.1, obj['time_to_collision'])),
                obj['risk_score'] / 10.0
            ])

        remaining_objects = self.max_objects - len(detections)
        state_array.extend([0.0] * (remaining_objects * 6))

        # Vehicle state (speed only, steering handled by MPC)
        vehicle_state = self.get_vehicle_state()
        state_array.extend([vehicle_state['speed'] / 50.0])

        return np.array(state_array, dtype=np.float16)
    

    def calculate_reward(self):
        try:
            reward = 0.0
            done = False
            info = {}

            # Get current state information
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            location = self.vehicle.get_location()
            detections = self.process_yolo_detection(self.front_camera, self.camera.get_transform())
            safety_info = self._analyze_safety(detections)

            nearest_in_path_dist = safety_info['nearest_in_path_dist']
            nearest_same_lane_dist = safety_info['nearest_same_lane_dist']
            min_ttc = safety_info['min_ttc']

            # Base reward for being active
            reward += 0.1

            # Dynamic target speed based on obstacles
            if nearest_in_path_dist < 15.0:
                # More conservative speed reduction when objects are closer
                target_speed = max(0, min(30.0, nearest_in_path_dist * 1.5))
            else:
                target_speed = 15.0

            # Enhanced speed control reward with collision risk consideration
            speed_diff = abs(speed - target_speed)
            collision_risk = self._calculate_collision_risk(speed, nearest_same_lane_dist, min_ttc)

            # Dynamic speed reward based on collision risk
            if collision_risk > 0.7:  # High risk
                if speed < target_speed:
                    speed_reward = 3.0  # Reward for slowing down in high-risk situations
                else:
                    speed_reward = -5.0 * collision_risk  # Penalize maintaining speed in high-risk
            elif collision_risk > 0.3:  # Medium risk
                if speed <= target_speed:
                    speed_reward = 2.0
                else:
                    speed_reward = -2.0 * collision_risk
            else:  # Low risk
                if speed_diff < 5.0:
                    speed_reward = 2.0
                elif speed_diff < 10.0:
                    speed_reward = 1.0 - (speed_diff / 10.0)
                else:
                    speed_reward = -1.0 * (speed_diff / 10.0)

            reward += speed_reward * 2.0

            # Enhanced same-lane object handling
            if nearest_same_lane_dist < 20.0:
                safe_following_distance = max(speed * 0.5, 5.0)  # Dynamic safe distance based on speed
                if nearest_same_lane_dist < safe_following_distance:
                    # Penalize not maintaining safe distance
                    distance_penalty = -3.0 * ((safe_following_distance - nearest_same_lane_dist) / safe_following_distance)
                    reward += distance_penalty
                elif speed > target_speed:
                    # Additional penalty for speeding when close to same-lane object
                    reward -= (speed - target_speed) * 0.5

            # Handle stopped cases with enhanced logic
            if speed < 1.0:
                near_obstacle = nearest_in_path_dist < 10.0
                if near_obstacle:
                    # Reward for appropriate stopping
                    stopping_reward = 2.0 * (1.0 - (nearest_in_path_dist / 10.0))
                    reward += stopping_reward
                    info['stop_reason'] = 'obstacle_in_path'
                else:
                    self.stuck_time += 0.2
                    if self.stuck_time > 3.0:
                        done = True
                        reward -= 20.0
                        info['termination_reason'] = 'stuck_no_obstacle'
            else:
                self.stuck_time = 0

            # Maximum episode time
            if time.time() - self.episode_start > 200.0:
                done = True
                reward -= 20.0
                info['termination_reason'] = 'max_time_exceeded'

            # Progressive distance reward
            if self.last_location is not None:
                distance_traveled = location.distance(self.last_location)
                # Scale distance reward based on safety
                safe_progress_factor = max(0, 1.0 - collision_risk)
                reward += distance_traveled * 0.5 * safe_progress_factor

            # Enhanced collision penalty with progression
            if len(self.collision_hist) > 0:
                # Calculate collision penalty based on the previous behavior
                collision_penalty = 200.0 * (1.0 + collision_risk)  # Higher penalty if collision was preventable
                reward -= collision_penalty
                done = True
                info['termination_reason'] = 'collision'

            self.last_location = location

            info.update({
                'speed': speed,
                'target_speed': target_speed,
                'speed_diff': speed_diff,
                'collision_risk': collision_risk,
                'reward': reward,
                'nearest_in_path_dist': nearest_in_path_dist,
                'nearest_same_lane_dist': nearest_same_lane_dist,
                'min_ttc': min_ttc
            })

            return reward, done, info

        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            traceback.print_exc()
            return 0.0, True, {'error': str(e)}

    def _calculate_collision_risk(self, speed, same_lane_dist, ttc):
        """Calculate collision risk factor between 0 and 1"""
        # Base risk on speed
        speed_risk = min(1.0, speed / 50.0)  # Normalize speed risk

        # Distance risk (higher risk at shorter distances)
        distance_risk = max(0, 1.0 - (same_lane_dist / 20.0))

        # TTC risk (higher risk at lower TTC)
        ttc_risk = max(0, 1.0 - (ttc / 10.0))

        # Combine risks with weights
        total_risk = (0.3 * speed_risk + 0.4 * distance_risk + 0.3 * ttc_risk)

        return min(1.0, max(0.0, total_risk))


    def _analyze_safety(self, detections):
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        nearest_same_lane_dist = float('inf')
        nearest_in_path_dist = float('inf')
        min_ttc = float('inf')

        # Important: We need to be stricter about what's in our lane
        relevant_classes = ['car', 'truck', 'bus', 'person']
        strict_in_lane_objects = []

        for obj in detections:
            if obj['class_name'] not in relevant_classes:
                continue

            obj_depth = obj['depth']
            obj_ttc = obj['time_to_collision']

            # Distance to objects in the same lane - using stricter criteria
            if obj['lane_position'] == 0:  # Same lane
                # Only consider objects with high confidence of being in the same lane
                if abs(obj['normalized_x_pos']) < 0.15:  # More centered in view
                    nearest_same_lane_dist = min(nearest_same_lane_dist, obj_depth)

                    # Only consider in-path objects that are truly in our path
                    if obj['in_ego_path'] and obj['depth_confidence'] > 0.6:
                        strict_in_lane_objects.append(obj)
                        nearest_in_path_dist = min(nearest_in_path_dist, obj_depth)
                        min_ttc = min(min_ttc, obj_ttc)

        # Default values if no objects detected
        if nearest_same_lane_dist == float('inf'):
            nearest_same_lane_dist = 100.0
        if nearest_in_path_dist == float('inf'):
            nearest_in_path_dist = 100.0
        if min_ttc == float('inf'):
            min_ttc = 100.0

        # Dynamic threshold based on speed
        threshold = max(5.0, (speed / 3.6) * 2)  # Minimum 5m, scales with speed

        return {
            'nearest_same_lane_dist': nearest_same_lane_dist,
            'nearest_in_path_dist': nearest_in_path_dist,
            'min_ttc': min_ttc,
            'threshold': threshold,
            'strict_in_lane_objects': strict_in_lane_objects
        }

    def step(self, rl_action=None):
        try:
            # Process YOLO detections from the front camera
            detections = self.process_yolo_detection(self.front_camera, self.camera.get_transform())

            # Analyze safety based on detections
            safety_info = self._analyze_safety(detections)

            # Calculate vehicle speed in km/h
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            # More conservative TTC threshold for early reaction
            ttc_threshold = 3.5  # 3.5 seconds TTL for same-lane objects

            # Distance threshold based on speed (stopping distance)
            # Modified to be more conservative: stopping distance = reaction distance + braking distance
            reaction_distance = (speed / 3.6) * 1.0  # 1.0 second reaction time
            braking_distance = (speed / 3.6) * (speed / 20.0)  # Simplified braking physics
            distance_threshold = max(8.0, reaction_distance + braking_distance)

            # Check if we have strict in-lane objects
            strict_in_lane_objects = safety_info.get('strict_in_lane_objects', [])
            # Get MPC control as the default
            mpc_control = self.controller.get_control(self.vehicle, self.world)
            control = mpc_control
            self.control_source = "MPC_FULL"  # Internal tracking only

            if complex_environment and rl_action is not None:
                rl_value = float(rl_action[0])
                throttle = float(np.clip(rl_value, 0.0, 1.0)) if rl_value >= 0 else 0.0
                brake = float(np.clip(-rl_value, 0.0, 1.0)) if rl_value < 0 else 0.0
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=mpc_control.steer,
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                )
                self.control_source = "RL_THROTTLE_MPC_STEER"  # Internal tracking only

            # Apply the selected control to the vehicle
            self.vehicle.apply_control(control)

            # Gather control and safety information for info dict
            control_info = {
                'throttle': control.throttle,
                'steer': control.steer,
                'brake': control.brake,
                'control_source': "RL" if complex_environment else "MPC",  # Simplified visible source
                'complex_environment': complex_environment,
                'safety_threshold': safety_info['threshold']
            }
            control_info.update(safety_info)

            # Update the simulation by ticking the world
            for _ in range(4):
                self.world.tick()

            # Get the new state after applying control
            new_state = self.get_state()

            # Calculate reward, done flag, and info
            reward, done, info = self.calculate_reward()
            info.update(control_info)

            if self.display is not None:
                with self.display_lock:
                    if self.current_surface is not None:
                        self.display.blit(self.current_surface, (0, 0))
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        raise Exception("User closed the window")

            # Return the new state, reward, done flag, and info
            return new_state, reward, done, info

        except Exception as e:
            # Handle any errors during the step
            print(f"Error in step: {e}")
            traceback.print_exc()
            return None, 0, True, {'error': str(e)}

    def run(self):
        """Main game loop using MPC controller"""
        try:
            running = True
            while running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_r:
                            # Reset simulation
                            self.reset()
                        elif event.key == pygame.K_p:
                            # Reset path
                            self.reset_path()

                # Execute step with MPC controller
                _, reward, done, info = self.step()

                # Handle termination
                if done:
                    print(f"Episode terminated: {info.get('termination_reason', 'unknown')}")
                    self.reset()

                # Maintain fps
                if self.clock:
                    self.clock.tick(20)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup
            self.cleanup_actors()
            pygame.quit()

            # Disable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def reset_path(self):
        """Reset the MPC controller path with a new destination"""
        if self.vehicle and self.controller:
            # Get current location
            start_location = self.vehicle.get_location()

            # Get random destination point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                return False

            end_point = random.choice(spawn_points)
            while end_point.location.distance(start_location) < 100:
                end_point = random.choice(spawn_points)

            # Set path using A*
            success = self.controller.set_path(
                self.world,
                start_location,
                end_point.location
            )

            return success

        return False
    
    def setup_vehicle(self):
        try:
            print("Starting vehicle setup...")
            blueprint_library = self.world.get_blueprint_library()
            print("Got blueprint library")
    
            vehicle_names = [
                'vehicle.audi.a2',
                'vehicle.audi.tt',
                'vehicle.bmw.grandtourer',
                'vehicle.dodge.charger_2020',
                'vehicle.lincoln.mkz',
                'vehicle.mercedes.coupe',
                'vehicle.mercedes.coupe_2020',
                'vehicle.seat.leon',
                'vehicle.tesla.model3',
                'vehicle.toyota.prius'
            ]
            vehicle_bp = None
            for vehicle_name in vehicle_names:
                vehicle_filters = blueprint_library.filter(vehicle_name)
                if vehicle_filters:
                    vehicle_bp = vehicle_filters[0]
                    print(f"Selected vehicle: {vehicle_name}")
                    break
            if vehicle_bp is None:
                available_vehicles = blueprint_library.filter('vehicle.*')
                if available_vehicles:
                    vehicle_bp = random.choice(available_vehicles)
                    print(f"Fallback to available vehicle: {vehicle_bp.id}")
                else:
                    raise Exception("No vehicle blueprints found")
    
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points found")
            spawn_point = random.choice(spawn_points)
            print(f"Selected spawn point: {spawn_point}")
    
            # Spawn vehicle with lock
            with self.vehicle_lock:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is None:
                    for i in range(10):
                        spawn_point = random.choice(spawn_points)
                        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                        if self.vehicle is not None:
                            break
                    if self.vehicle is None:
                        raise Exception("Failed to spawn vehicle after multiple attempts")
                print("Vehicle spawned successfully")
    
            # Set up camera and sensors outside the lock to minimize lock duration
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
            camera_bp.set_attribute('fov', '90')
            self.camera = self.world.spawn_actor(camera_bp, self.camera_transform, attach_to=self.vehicle)
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: self._process_image(weak_self, image))
            collision_bp = blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
    
            self.controller = MPCController()
            end_point = random.choice(spawn_points)
            while end_point.location.distance(spawn_point.location) < 100:
                end_point = random.choice(spawn_points)
            success = self.controller.set_path(self.world, spawn_point.location, end_point.location)
            if not success:
                for _ in range(5):
                    end_point = random.choice(spawn_points)
                    success = self.controller.set_path(self.world, spawn_point.location, end_point.location)
                    if success:
                        break
                if not success:
                    raise Exception("Failed to find a valid path after multiple attempts")
    
            for _ in range(10):
                self.world.tick()
                if self.front_camera is not None:
                    print("Sensors initialized successfully")
                    return True
            if self.front_camera is None:
                raise Exception("Camera failed to initialize")
            return True
        except Exception as e:
            print(f"Error setting up vehicle: {e}")
            self.cleanup_actors()
            return False


    def reset(self):
        """Reset environment with improved movement verification and visualization cleanup"""
        print("Starting environment reset...")

        # Clear all visualizations before cleanup
        if hasattr(self, 'world') and self.world is not None:
            try:
                # Clear all debug visualizations
                self.world.debug.draw_line(
                    carla.Location(0, 0, 0),
                    carla.Location(0, 0, 0.1),
                    thickness=0.0,
                    color=carla.Color(0, 0, 0),
                    life_time=0.0,
                    persistent_lines=False
                )

                # A workaround to clear persistent visualization is to call 
                # clear_all_debug_shapes which is available in CARLA 0.10.0
                if hasattr(self.world.debug, 'clear_all_debug_shapes'):
                    self.world.debug.clear_all_debug_shapes()
                    print("Cleared all debug visualizations")
            except Exception as e:
                print(f"Warning: Failed to clear visualizations: {e}")

        # Reset path visualization flag in MPC controller if it exists
        if hasattr(self, 'controller') and self.controller is not None:
            if hasattr(self.controller, 'path_visualization_done'):
                self.controller.path_visualization_done = False
                print("Reset MPC visualization flags")

        # Cleanup actors and NPCs
        self.cleanup_actors()
        self.cleanup_npcs()

        # Clear state variables
        self.collision_hist = []  # Clear collision history
        self.stuck_time = 0
        self.episode_start = time.time()
        self.last_location = None

        # Set up new episode
        if not self.setup_vehicle():
            raise Exception("Failed to setup vehicle")
        #self.controller.clear_path_visualization(self.world)
        # Wait for initial camera frame
        print("Waiting for camera initialization...")
        timeout = time.time() + 10.0
        while self.front_camera is None:
            self.world.tick()
            if time.time() > timeout:
                raise Exception("Camera initialization timeout")
            time.sleep(0.1)

        # Spawn NPCs after camera is ready
        self.spawn_npcs()

        # Let the physics settle and NPCs initialize
        for _ in range(20):
            self.world.tick()
            time.sleep(0.05)

        # Get initial state after everything is set up
        state = self.get_state()
        if state is None:
            raise Exception("Failed to get initial state")

        print("Environment reset complete")
        return state




    def spawn_npcs(self):
        """Spawn NPC vehicles and pedestrians near the training vehicle"""
        try:
            number_of_vehicles = 60
            spawn_radius = 40.0

            if self.vehicle is None:
                print("Training vehicle not found! Cannot spawn NPCs.")
                return

            # Get training vehicle's location
            vehicle_location = self.vehicle.get_location()

            # Configure traffic manager
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(1.5)
            traffic_manager.global_percentage_speed_difference(-50.0)

            # Spawn strategic vehicles
            self._spawn_strategic_npcs(close_npcs=20, far_npcs=number_of_vehicles)

            # Spawn strategic pedestrians
            self._spawn_strategic_pedestrians(close_peds=0, far_peds=0)

            # Visualize spawn radius using points and lines
            try:
                debug = self.world.debug
                if debug:
                    # Draw points around the radius (approximating a circle)
                    num_points = 36  # Number of points to approximate circle
                    for i in range(num_points):
                        angle = (2 * math.pi * i) / num_points
                        point = carla.Location(
                            x=vehicle_location.x + spawn_radius * math.cos(angle),
                            y=vehicle_location.y + spawn_radius * math.sin(angle),
                            z=vehicle_location.z
                        )
                        debug.draw_point(
                            point,
                            size=0.1,
                            color=carla.Color(r=0, g=255, b=0),
                            life_time=1.0
                        )
                        # Draw lines between points
                        if i > 0:
                            prev_point = carla.Location(
                                x=vehicle_location.x + spawn_radius * math.cos((2 * math.pi * (i-1)) / num_points),
                                y=vehicle_location.y + spawn_radius * math.sin((2 * math.pi * (i-1)) / num_points),
                                z=vehicle_location.z
                            )
                            debug.draw_line(
                                prev_point,
                                point,
                                thickness=0.1,
                                color=carla.Color(r=0, g=255, b=0),
                                life_time=1.0
                            )
            except Exception as debug_error:
                print(f"Warning: Could not draw debug visualization: {debug_error}")

        except Exception as e:
            print(f"Error spawning NPCs: {e}")
            traceback.print_exc()
            self.cleanup_npcs()

    def _spawn_strategic_npcs(self, close_npcs=20, far_npcs=60):
        """
        Spawn NPC vehicles with some specifically placed in front of the player vehicle
        """
        try:
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')

            # Filter for cars (no bikes/motorcycles)
            car_blueprints = [
                bp for bp in blueprints 
                if any(car_type in bp.id.lower() 
                    for car_type in ['car', 'tesla', 'audi', 'bmw', 'mercedes', 'toyota', 'ford'])
            ]

            if not car_blueprints:
                print("Warning: Using all vehicle blueprints as fallback")
                car_blueprints = blueprints

            spawn_points = self.world.get_map().get_spawn_points()

            # Get player's transform
            player_transform = self.vehicle.get_transform()
            player_location = player_transform.location
            player_forward = player_transform.get_forward_vector()

            # Setup traffic manager
            traffic_manager = self.client.get_trafficmanager()

            # First, spawn NPCs close to the player
            close_spawn_points = []
            for spawn_point in spawn_points:
                # Calculate vector from player to spawn point
                to_spawn_x = spawn_point.location.x - player_location.x
                to_spawn_y = spawn_point.location.y - player_location.y
                to_spawn_z = spawn_point.location.z - player_location.z

                # Calculate distance using pythagorean theorem
                distance = math.sqrt(to_spawn_x**2 + to_spawn_y**2 + to_spawn_z**2)

                # Calculate dot product manually
                dot_product = (to_spawn_x * player_forward.x + 
                            to_spawn_y * player_forward.y + 
                            to_spawn_z * player_forward.z)

                # Check if point is in front of player and within distance range
                if (distance < 60.0 and distance > 20.0 and  # Between 30m and 50m
                    dot_product > 0):                        # In front of player
                    close_spawn_points.append(spawn_point)

            # Spawn close NPCs
            print(f"\nSpawning {close_npcs} NPCs near player...")
            random.shuffle(close_spawn_points)
            for i in range(min(close_npcs, len(close_spawn_points))):
                try:
                    blueprint = random.choice(car_blueprints)
                    if blueprint.has_attribute('color'):
                        blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').    recommended_values))

                    vehicle = self.world.spawn_actor(blueprint, close_spawn_points[i])
                    vehicle.set_autopilot(True)
                    self.npc_vehicles.append(vehicle)

                    # Set traffic manager parameters
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(0, 10))
                    traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 2.0))

                    print(f"Spawned close NPC {i+1}/{close_npcs}")

                    # Give time for physics to settle
                    self.world.tick()

                    # Draw debug line to show spawn location
                    debug = self.world.debug
                    if debug:
                        # Draw a line from ego vehicle to spawned vehicle
                        debug.draw_line(
                            player_location,
                            vehicle.get_location(),
                            thickness=0.1,
                            color=carla.Color(r=0, g=255, b=0),
                            life_time=5.0
                        )

                except Exception as e:
                    print(f"Failed to spawn close NPC: {e}")
                    continue
                
            # Then spawn far NPCs randomly in the map
            print(f"\nSpawning {far_npcs} NPCs around the map...")
            random.shuffle(spawn_points)
            for i in range(far_npcs):
                try:
                    blueprint = random.choice(car_blueprints)
                    if blueprint.has_attribute('color'):
                        blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').    recommended_values))

                    # Try to find a spawn point not too close to player
                    spawn_point = None
                    for point in spawn_points:
                        # Calculate distance manually
                        dx = point.location.x - player_location.x
                        dy = point.location.y - player_location.y
                        dz = point.location.z - player_location.z
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)

                        if distance > 50.0:  # More than 50m away
                            spawn_point = point
                            spawn_points.remove(point)
                            break
                        
                    if spawn_point is None:
                        print("Couldn't find suitable spawn point for far NPC")
                        continue

                    vehicle = self.world.spawn_actor(blueprint, spawn_point)
                    vehicle.set_autopilot(True)
                    self.npc_vehicles.append(vehicle)

                    # Set traffic manager parameters
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(0, 10))
                    traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 2.0))

                    print(f"Spawned far NPC {i+1}/{far_npcs}")

                    # Give time for physics to settle
                    self.world.tick()

                except Exception as e:
                    print(f"Failed to spawn far NPC: {e}")
                    continue
                
            print(f"\nSuccessfully spawned {len(self.npc_vehicles)} NPC vehicles total")

        except Exception as e:
            print(f"Error in spawn_strategic_npcs: {e}")

    def _spawn_strategic_pedestrians(self, close_peds=0, far_peds=0):
        """
        Spawn pedestrians with robust error handling and careful placement
        to avoid simulation crashes
        """
        try:
            # Filter walker blueprints
            walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')

            if len(walker_bps) == 0:
                print("Warning: No pedestrian blueprints found!")
                return

            # Get player's transform
            player_transform = self.vehicle.get_transform()
            player_location = player_transform.location

            # Create spawn points manually instead of using road spawn points
            print(f"\nSpawning {close_peds} pedestrians near player...")

            # For close pedestrians, use sidewalks near player
            spawn_attempts = 0
            close_peds_spawned = 0

            while close_peds_spawned < close_peds and spawn_attempts < 20:
                spawn_attempts += 1

                try:
                    # Get a waypoint near the player
                    player_waypoint = self.world.get_map().get_waypoint(player_location)

                    # Find nearby random location within 30-60m, biased to sidewalks
                    distance = random.uniform(20.0, 80.0)
                    angle = random.uniform(-45, 45)  # Degrees, roughly in front of player
                    angle_rad = math.radians(angle)

                    # Calculate offset position based on player forward direction
                    # Create right vector manually from forward vector
                    forward = player_transform.get_forward_vector()

                    # Calculate right vector using cross product with up vector (0,0,1)
                    right_x = -forward.y  # Cross product with up vector
                    right_y = forward.x

                    # Calculate the target position
                    target_x = player_location.x + forward.x * distance * math.cos(angle_rad) + right_x * distance *    math.sin(angle_rad)
                    target_y = player_location.y + forward.y * distance * math.cos(angle_rad) + right_y * distance *    math.sin(angle_rad)
                    target_location = carla.Location(x=target_x, y=target_y, z=player_location.z)

                    # Get waypoint near this location
                    waypoint = self.world.get_map().get_waypoint(target_location)
                    if not waypoint:
                        continue
                    
                    # Try to find nearby sidewalk
                    sidewalk_wp = None
                    for _ in range(5):  # Try both sides and offsets
                        try:
                            # Try right side first (usually where sidewalks are)
                            temp_wp = waypoint.get_right_lane()
                            if temp_wp and temp_wp.lane_type == carla.LaneType.Sidewalk:
                                sidewalk_wp = temp_wp
                                break
                            
                            # Try left side
                            temp_wp = waypoint.get_left_lane()
                            if temp_wp and temp_wp.lane_type == carla.LaneType.Sidewalk:
                                sidewalk_wp = temp_wp
                                break
                            
                            # Move the waypoint forward and try again
                            next_wps = waypoint.next(5.0)
                            if next_wps:
                                waypoint = next_wps[0]
                        except:
                            pass
                        
                    # If no sidewalk found, use original waypoint but offset from road
                    if sidewalk_wp:
                        spawn_transform = sidewalk_wp.transform
                        # Raise slightly to avoid ground collision
                        spawn_transform.location.z += 0.5
                    else:
                        # Offset from road - Use manual right vector calculation
                        spawn_transform = waypoint.transform
                        # Get right vector from waypoint transform
                        wp_forward = waypoint.transform.get_forward_vector()
                        # Calculate right vector from forward vector
                        wp_right_x = -wp_forward.y
                        wp_right_y = wp_forward.x

                        # Apply offset
                        spawn_transform.location.x += wp_right_x * 3.0
                        spawn_transform.location.y += wp_right_y * 3.0
                        spawn_transform.location.z += 0.5

                    # Choose random walker blueprint
                    walker_bp = random.choice(walker_bps)

                    # Make sure pedestrian is not invincible
                    if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')

                    # Simplified attribute handling
                    try:
                        # Just try common attributes without checking if they exist
                        for attr_name in ['color', 'texture']:
                            attr = walker_bp.get_attribute(attr_name)
                            if attr and attr.recommended_values:
                                walker_bp.set_attribute(attr_name, random.choice(attr.recommended_values))
                    except:
                        # If this fails, just continue with default appearance
                        pass
                    
                    # Spawn the walker directly instead of using batch command
                    try:
                        walker = self.world.spawn_actor(walker_bp, spawn_transform)
                        if not walker:
                            print("Failed to spawn walker actor")
                            continue
                        
                        # Add to the class's walker list
                        if not hasattr(self, 'pedestrians'):
                            self.pedestrians = []
                        self.pedestrians.append(walker)
                    except Exception as e:
                        print(f"Failed to spawn pedestrian: {e}")
                        continue
                    
                    # Wait for physics to settle
                    for _ in range(5):
                        self.world.tick()

                    # Create walker controller directly instead of using batch
                    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

                    try:
                        controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                        if not controller:
                            print("Failed to spawn controller")
                            walker.destroy()
                            self.pedestrians.pop()
                            continue
                        
                        # Add to the class's controller list
                        self.pedestrian_controllers.append(controller)
                    except Exception as e:
                        print(f"Failed to spawn controller: {e}")
                        walker.destroy()
                        self.pedestrians.pop()
                        continue
                    
                    # Give time for actors to initialize properly
                    self.world.tick()

                    # Initialize controller to make pedestrian walk randomly
                    try:
                        # Start walking to random destination
                        controller.start()
                        controller.go_to_location(self.world.get_random_location_from_navigation())

                        # Set walking speed (lower speed for safety)
                        controller.set_max_speed(1.2)

                        close_peds_spawned += 1
                        print(f"Spawned close pedestrian {close_peds_spawned}/{close_peds}")
                    except Exception as e:
                        print(f"Error initializing pedestrian controller: {e}")
                        controller.destroy()
                        walker.destroy()
                        self.pedestrians.pop()
                        self.pedestrian_controllers.pop()
                        continue
                    
                    # Additional delay to ensure stability
                    for _ in range(3):
                        self.world.tick()

                except Exception as e:
                    print(f"Error during close pedestrian spawn attempt {spawn_attempts}: {e}")
                    continue
                
            # For far pedestrians, use navigation system
            print(f"\nSpawning {far_peds} pedestrians around the map...")
            spawn_attempts = 0
            far_peds_spawned = 0

            while far_peds_spawned < far_peds and spawn_attempts < 20:
                spawn_attempts += 1

                try:
                    # Get random location from navigation
                    nav_location = None
                    for _ in range(10):
                        try:
                            test_location = self.world.get_random_location_from_navigation()
                            # Check distance from player
                            dx = test_location.x - player_location.x
                            dy = test_location.y - player_location.y
                            dz = test_location.z - player_location.z
                            distance = math.sqrt(dx**2 + dy**2 + dz**2)

                            if distance > 70.0:  # Farther away for safety
                                nav_location = test_location
                                break
                        except:
                            continue
                        
                    if nav_location is None:
                        continue
                    
                    # Create spawn transform
                    spawn_transform = carla.Transform(nav_location)

                    # Choose random walker blueprint
                    walker_bp = random.choice(walker_bps)

                    # Configure blueprint
                    if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')

                    # Simplified attribute handling
                    try:
                        # Just try common attributes without checking if they exist
                        for attr_name in ['color', 'texture']:
                            attr = walker_bp.get_attribute(attr_name)
                            if attr and attr.recommended_values:
                                walker_bp.set_attribute(attr_name, random.choice(attr.recommended_values))
                    except:
                        # If this fails, just continue with default appearance
                        pass
                    
                    # Spawn walker directly
                    try:
                        walker = self.world.spawn_actor(walker_bp, spawn_transform)
                        if not walker:
                            print("Failed to spawn far walker actor")
                            continue
                        
                        # Add to the class's walker list
                        if not hasattr(self, 'pedestrians'):
                            self.pedestrians = []
                        self.pedestrians.append(walker)
                    except Exception as e:
                        print(f"Failed to spawn far pedestrian: {e}")
                        continue
                    
                    # Wait for physics to settle
                    for _ in range(5):
                        self.world.tick()

                    # Create walker controller directly
                    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

                    try:
                        controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                        if not controller:
                            print("Failed to retrieve far controller actor")
                            walker.destroy()
                            self.pedestrians.pop()
                            continue
                        
                        self.pedestrian_controllers.append(controller)
                    except Exception as e:
                        print(f"Failed to spawn far controller: {e}")
                        walker.destroy()
                        self.pedestrians.pop()
                        continue
                    
                    # Give time for actors to initialize properly
                    self.world.tick()

                    # Initialize controller
                    try:
                        controller.start()
                        controller.go_to_location(self.world.get_random_location_from_navigation())
                        controller.set_max_speed(1.2)

                        far_peds_spawned += 1
                        print(f"Spawned far pedestrian {far_peds_spawned}/{far_peds}")
                    except Exception as e:
                        print(f"Error initializing far pedestrian controller: {e}")
                        controller.destroy()
                        walker.destroy()
                        self.pedestrians.pop()
                        self.pedestrian_controllers.pop()
                        continue
                    
                    # Additional delay for stability
                    for _ in range(3):
                        self.world.tick()

                except Exception as e:
                    print(f"Error during far pedestrian spawn attempt {spawn_attempts}: {e}")
                    continue
                
            total_peds = len(self.pedestrians) if hasattr(self, 'pedestrians') else 0
            print(f"\nSuccessfully spawned {total_peds} pedestrians total")

        except Exception as e:
            print(f"Error in spawn_strategic_pedestrians: {e}")

    def cleanup_npcs(self):
        """Clean up all spawned NPCs, pedestrians, and controllers"""
        try:
            # Stop pedestrian controllers
            for controller in self.pedestrian_controllers:
                if controller and controller.is_alive:
                    controller.stop()
                    controller.destroy()
            self.pedestrian_controllers.clear()

            # Destroy pedestrians
            if hasattr(self, 'pedestrians'):
                for pedestrian in self.pedestrians:
                    if pedestrian and pedestrian.is_alive:
                        pedestrian.destroy()
                self.pedestrians.clear()

            # Destroy vehicles
            for vehicle in self.npc_vehicles:
                if vehicle and vehicle.is_alive:
                    vehicle.destroy()
            self.npc_vehicles.clear()

            print("Successfully cleaned up all NPCs")

        except Exception as e:
            print(f"Error cleaning up NPCs: {e}")
            traceback.print_exc()

    def close(self):
        """Close environment and cleanup"""
        try:
            self.cleanup_actors()
            self.cleanup_npcs()

            if pygame.get_init():
                pygame.quit()

            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

            print("Environment closed successfully")

        except Exception as e:
            print(f"Error closing environment: {e}")
            traceback.print_exc()













