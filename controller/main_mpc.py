






import os
import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import carla
import random
import math
from sklearn.cluster import DBSCAN
from safety_controller import SafetyController
from vehicle_detector import VehicleDetector 
from mpc_controller import MPCController  # Import the MPC controller
from queue import PriorityQueue  # Import PriorityQueue for A* implementation


# Pygame window settings
WINDOW_WIDTH =1920
WINDOW_HEIGHT = 1080
FPS = 30

class CameraManager:
    def __init__(self, parent_actor, world):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self._world = world
        
        # Set up camera blueprint
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        blueprint.set_attribute('fov', '110')
        
        # Find camera spawn point (behind and above the vehicle)
        spawn_point = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        
        # Spawn camera
        self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=self._parent)
        
        # Setup callback for camera data
        self.sensor.listen(self._parse_image)
    
    def _parse_image(self, image):
        """Convert CARLA raw image to Pygame surface"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def render(self, display):
        """Render camera image to Pygame display"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    def destroy(self):
        """Clean up camera sensor"""
        if self.sensor is not None:
            self.sensor.destroy()


def connect_to_carla(retries=10, timeout=5.0):
    """Attempt to connect to CARLA with retries"""
    for attempt in range(retries):
        try:
            print(f"Attempting to connect to CARLA (Attempt {attempt + 1}/{retries})")
            client = carla.Client('localhost', 2000)
            client.set_timeout(timeout)
            world = client.get_world()
            print("Successfully connected to CARLA")
            return client, world
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
    raise ConnectionError("Failed to connect to CARLA after multiple attempts")

def find_spawn_points(world, min_distance=30.0):
    """Find suitable spawn points with minimum distance between them"""
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < 2:
        raise ValueError("Not enough spawn points found!")
    
    for _ in range(50):
        start_point = random.choice(spawn_points)
        end_point = random.choice(spawn_points)
        distance = start_point.location.distance(end_point.location)
        if distance >= min_distance:
            return start_point, end_point
    return spawn_points[0], spawn_points[1]

def spawn_strategic_npcs(world, player_vehicle, close_npcs=0, far_npcs=15):
    """
    Spawn NPCs with some specifically placed in front of the player vehicle
    """
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    
    # Filter for cars (no bikes/motorcycles)
    car_blueprints = [
        bp for bp in blueprints 
        if any(car_type in bp.id.lower() 
               for car_type in ['car', 'tesla', 'audi', 'bmw', 'mercedes', 'toyota', 'ford'])
    ]
    
    if not car_blueprints:
        print("Warning: Using all vehicle blueprints as fallback")
        car_blueprints = blueprints
    
    spawn_points = world.get_map().get_spawn_points()
    
    # Get player's transform
    player_transform = player_vehicle.get_transform()
    player_location = player_transform.location
    player_forward = player_transform.get_forward_vector()
    
    vehicles = []
    
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
        if (distance < 50.0 and distance > 30.0 and  # Between 30m and 50m
            dot_product > 0):                        # In front of player
            close_spawn_points.append(spawn_point)
    
    # Spawn close NPCs
    print(f"\nSpawning {close_npcs} NPCs near player...")
    random.shuffle(close_spawn_points)
    for i in range(min(close_npcs, len(close_spawn_points))):
        try:
            blueprint = random.choice(car_blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            
            vehicle = world.spawn_actor(blueprint, close_spawn_points[i])
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"Spawned close NPC {i+1}/{close_npcs}")
            
            # Give time for physics to settle
            world.tick(2.0)
            time.sleep(0.1)
            
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
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            
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
                
            vehicle = world.spawn_actor(blueprint, spawn_point)
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"Spawned far NPC {i+1}/{far_npcs}")
            
            # Give time for physics to settle
            world.tick(2.0)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to spawn far NPC: {e}")
            continue
    
    print(f"\nSuccessfully spawned {len(vehicles)} NPCs total")
    return vehicles

def spawn_strategic_pedestrians(world, player_vehicle, close_peds=5, far_peds=10):
    """
    Spawn pedestrians with robust error handling and careful placement
    to avoid simulation crashes - Fixed version for older CARLA API
    """
    import random
    import math
    import time
    import carla
    
    # Lists to store spawned walkers and controllers
    walkers = []
    walker_controllers = []
    
    try:
        # Filter walker blueprints
        walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
        
        if len(walker_bps) == 0:
            print("Warning: No pedestrian blueprints found!")
            return [], []
        
        # Get player's transform
        player_transform = player_vehicle.get_transform()
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
                player_waypoint = world.get_map().get_waypoint(player_location)
                
                # Find nearby random location within 30-60m, biased to sidewalks
                distance = random.uniform(10.0, 100.0)
                angle = random.uniform(-45, 45)  # Degrees, roughly in front of player
                angle_rad = math.radians(angle)
                
                # Calculate offset position based on player forward direction
                # Create right vector manually from forward vector
                forward = player_transform.get_forward_vector()
                
                # Calculate right vector using cross product with up vector (0,0,1)
                right_x = -forward.y  # Cross product with up vector
                right_y = forward.x
                
                # Calculate the target position
                target_x = player_location.x + forward.x * distance * math.cos(angle_rad) + right_x * distance * math.sin(angle_rad)
                target_y = player_location.y + forward.y * distance * math.cos(angle_rad) + right_y * distance * math.sin(angle_rad)
                target_location = carla.Location(x=target_x, y=target_y, z=player_location.z)
                
                # Get waypoint near this location
                waypoint = world.get_map().get_waypoint(target_location)
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
                
                # FIX: Replace batch spawning with direct spawn_actor
                # Spawn the walker directly instead of using batch command
                try:
                    walker = world.spawn_actor(walker_bp, spawn_transform)
                    if not walker:
                        print("Failed to spawn walker actor")
                        continue
                    
                    walkers.append(walker)
                except Exception as e:
                    print(f"Failed to spawn pedestrian: {e}")
                    continue
                
                # Wait for physics to settle
                for _ in range(5):
                    world.tick()
                
                # Create walker controller directly instead of using batch
                controller_bp = world.get_blueprint_library().find('controller.ai.walker')
                
                try:
                    controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                    if not controller:
                        print("Failed to spawn controller")
                        walker.destroy()
                        walkers.pop()
                        continue
                        
                    walker_controllers.append(controller)
                except Exception as e:
                    print(f"Failed to spawn controller: {e}")
                    walker.destroy()
                    walkers.pop()
                    continue
                
                # Give time for actors to initialize properly
                world.tick()
                
                # Initialize controller to make pedestrian walk randomly
                try:
                    # Start walking to random destination
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    
                    # Set walking speed (lower speed for safety)
                    controller.set_max_speed(1.2)
                    
                    close_peds_spawned += 1
                    print(f"Spawned close pedestrian {close_peds_spawned}/{close_peds}")
                except Exception as e:
                    print(f"Error initializing pedestrian controller: {e}")
                    controller.destroy()
                    walker.destroy()
                    walkers.pop()
                    walker_controllers.pop()
                    continue
                
                # Additional delay to ensure stability
                for _ in range(3):
                    world.tick()
                
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
                        test_location = world.get_random_location_from_navigation()
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
                
                # FIX: Replace batch spawning with direct spawn_actor 
                # Spawn walker directly
                try:
                    walker = world.spawn_actor(walker_bp, spawn_transform)
                    if not walker:
                        print("Failed to spawn far walker actor")
                        continue
                        
                    walkers.append(walker)
                except Exception as e:
                    print(f"Failed to spawn far pedestrian: {e}")
                    continue
                
                # Wait for physics to settle
                for _ in range(5):
                    world.tick()
                
                # Create walker controller directly
                controller_bp = world.get_blueprint_library().find('controller.ai.walker')
                
                try:
                    controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                    if not controller:
                        print("Failed to retrieve far controller actor")
                        walker.destroy()
                        walkers.pop()
                        continue
                        
                    walker_controllers.append(controller)
                except Exception as e:
                    print(f"Failed to spawn far controller: {e}")
                    walker.destroy()
                    walkers.pop()
                    continue
                
                # Give time for actors to initialize properly
                world.tick()
                
                # Initialize controller
                try:
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    controller.set_max_speed(1.2)
                    
                    far_peds_spawned += 1
                    print(f"Spawned far pedestrian {far_peds_spawned}/{far_peds}")
                except Exception as e:
                    print(f"Error initializing far pedestrian controller: {e}")
                    controller.destroy()
                    walker.destroy()
                    walkers.pop()
                    walker_controllers.pop()
                    continue
                
                # Additional delay for stability
                for _ in range(3):
                    world.tick()
                
            except Exception as e:
                print(f"Error during far pedestrian spawn attempt {spawn_attempts}: {e}")
                continue
        
        total_peds = len(walkers)
        print(f"\nSuccessfully spawned {total_peds} pedestrians total")
        
    except Exception as e:
        print(f"Error in spawn_strategic_pedestrians: {e}")
        # Clean up any partially created pedestrians
        safe_cleanup_pedestrians(world, walkers, walker_controllers)
        return [], []
    
    return walkers, walker_controllers

def safe_cleanup_pedestrians(world, walkers, walker_controllers):
    """
    Safely clean up pedestrians and their controllers
    """
    print("Performing safe pedestrian cleanup...")
    
    # First stop all controllers
    for controller in walker_controllers:
        try:
            if controller is not None and controller.is_alive:
                controller.stop()
        except:
            pass
    
    # Give time for controllers to stop
    for _ in range(5):
        try:
            world.tick()
        except:
            pass
    
    # Remove controllers
    for controller in walker_controllers:
        try:
            if controller is not None and controller.is_alive:
                controller.destroy()
        except:
            pass
    
    # Remove walkers
    for walker in walkers:
        try:
            if walker is not None and walker.is_alive:
                walker.destroy()
        except:
            pass
# Modify main() to include this function and update the cleanup logic
def main():
    try:
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("CARLA Navigation with MPC")  # Updated window caption
        display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()
        
        # Connect to CARLA
        print("Connecting to CARLA...")
        client, world = connect_to_carla()
        
        # Set synchronous mode with enhanced physics settings
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        settings.substepping = True  # Enable physics substepping
        settings.max_substep_delta_time = 0.01  # Maximum physics substep size
        settings.max_substeps = 10  # Maximum number of substeps
        world.apply_settings(settings)
        
        # Wait for the world to stabilize
        print("Waiting for world to stabilize...")
        for _ in range(20):
            world.tick(2.0)
            time.sleep(0.1)
        
        # Find suitable spawn points
        print("Finding suitable spawn points...")
        start_point, end_point = find_spawn_points(world)
        print(f"Start point: {start_point.location}")
        print(f"End point: {end_point.location}")
        
        # Spawn vehicle
                # Get vehicle blueprint with priority order (compatible with CARLA 0.10.0)
        print("Selecting vehicle blueprint...")
        blueprint_library = world.get_blueprint_library()
        
        # Try to get vehicle blueprints in order of priority
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
            # Fallback to any available vehicle if none of the preferred ones exist
            available_vehicles = blueprint_library.filter('vehicle.*')
            if available_vehicles:
                vehicle_bp = random.choice(available_vehicles)
                print(f"Fallback to available vehicle: {vehicle_bp.id}")
            else:
                raise Exception("No vehicle blueprints found")
        vehicle = None
        camera = None
        detector = None
        safety_controller = None
        
        try:
            vehicle = world.spawn_actor(vehicle_bp, start_point)
            print("Vehicle spawned successfully")
            
            # Set up camera
            camera = CameraManager(vehicle, world)
            
            # Set up Vehicle Detector and Controllers
            # Use MPC controller instead of NavigationController
            controller = MPCController()  # Replace with MPC controller
            detector = VehicleDetector(vehicle, world, controller)
            safety_controller = SafetyController(vehicle, world, controller)
            
            # Allow everything to settle
            world.tick()
            time.sleep(0.5)
            
            # Plan path using MPC controller
            print("Planning path using MPC controller...")
            success = controller.set_path(world, start_point.location, end_point.location)
            
            if not success:
                print("Failed to plan path!")
                return
            
            print(f"Path planned with {len(controller.waypoints)} waypoints")
            
            # Spawn NPC vehicles
            print("Spawning NPC vehicles...")
            npcs = spawn_strategic_npcs(world, vehicle, close_npcs=0, far_npcs=15)
            print(f"Spawned {len(npcs)} total NPCs")
            
            # Wait for physics to settle after vehicle spawning
            for _ in range(10):
                world.tick()
                time.sleep(0.05)
            
            # Spawn pedestrians
            print("Spawning pedestrians...")
            walkers, walker_controllers = spawn_strategic_pedestrians(world, vehicle, close_peds=10, far_peds=2)
            
            # Main simulation loop
            with tqdm(total=5000, desc="MPC Navigation") as pbar:
                # Inside the main simulation loop
                try:
                    while True:
                        try:
                            # Tick the world with correct parameter
                            start_time = time.time()
                            while True:
                                try:
                                    world.tick(2.0)
                                    break
                                except RuntimeError as e:
                                    if time.time() - start_time > 10.0:  # Overall timeout
                                        raise
                                    time.sleep(0.1)
                                    continue
                                
                            # Update Pygame display
                            display.fill((0, 0, 0))
                            camera.render(display)
                            if detector is not None:
                                detector.detect_vehicles()
                                detector.render(display)
                            pygame.display.flip()

                            # Safety check before applying control
                            safety_controller.check_safety()

                            # Apply the updated safety control logic
                            if safety_controller.last_detected:
                                # Apply emergency brake
                                vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))
                                print("Safety stop active")
                            else:
                                # Use MPC controller to get control commands
                                control = controller.get_control(vehicle, world)
                                vehicle.apply_control(control)
                                # Debug output for normal operation
                                print(f"MPC Control commands - Throttle: {control.throttle:.2f}, "
                                      f"Brake: {control.brake:.2f}, "
                                      f"Steer: {control.steer:.2f}")

                            # Debug vehicle state
                            velocity = vehicle.get_velocity()
                            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
                            print(f"Vehicle speed: {speed:.2f} km/h")

                            # Update progress (works with both controller types)
                            if controller.waypoints:
                                progress = (len(controller.visited_waypoints) / 
                                          len(controller.waypoints)) * 100
                            else:
                                progress = 0

                            pbar.update(1)
                            pbar.set_postfix({
                                'speed': f"{speed:.1f}km/h",
                                'progress': f"{progress:.1f}%",
                                'safety': 'ACTIVE' if safety_controller.last_detected else 'OK'
                            })

                            # Check if destination reached
                            if (controller.current_waypoint_index >= 
                                len(controller.waypoints) - 1):
                                print("\nDestination reached!")
                                time.sleep(2)
                                break
                            
                            # Handle pygame events
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    return
                                elif event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_ESCAPE:
                                        return

                            clock.tick(FPS)

                        except Exception as e:
                            print(f"\nError during simulation step: {str(e)}")
                            if "time-out" in str(e).lower():
                                print("Attempting to recover from timeout...")
                                time.sleep(1.0)  # Give the simulator time to recover
                                continue
                            else:
                                raise
                
                except KeyboardInterrupt:
                    print("\nNavigation interrupted by user")
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                        
        finally:
            print("Cleaning up...")
            try:
                # First safely clean up pedestrians
                if 'walkers' in locals() and 'walker_controllers' in locals():
                    safe_cleanup_pedestrians(world, walkers, walker_controllers)
                
                # Then clean up the rest
                if detector is not None:
                    detector.destroy()
                if camera is not None:
                    camera.destroy()
                if safety_controller is not None:
                    safety_controller.destroy()
                if vehicle is not None:
                    vehicle.destroy()
                
                # Destroy NPC vehicles
                if 'npcs' in locals() and npcs:
                    for npc in npcs:
                        if npc is not None and npc.is_alive:
                            try:
                                npc.set_autopilot(False)  # Disable autopilot before destroying
                                npc.destroy()
                            except:
                                pass
                
                # Restore original settings
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.substepping = False  # Disable physics substepping
                world.apply_settings(settings)
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
                
            print("Cleanup complete")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        pygame.quit()
        print("Pygame quit")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user.')
    except Exception as e:
        print(f'Error occurred: {e}')
        import traceback
        traceback.print_exc()


























