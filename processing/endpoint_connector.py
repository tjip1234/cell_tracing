import numpy as np
from PySide6.QtCore import QThread, Signal
import cv2

class EndpointConnector(QThread):
    finished = Signal(np.ndarray, list)  # connected_skeleton, connections_made
    progress = Signal(str)
    
    def __init__(self, skeleton, max_distance=25, num_iterations=3):
        super().__init__()
        self.skeleton = skeleton
        self.max_distance = max_distance
        self.num_iterations = num_iterations
        
    def run(self):
        try:
            self.progress.emit("Starting endpoint connection...")
            
            current_skeleton = self.skeleton.copy()
            all_connections = []
            
            for iteration in range(self.num_iterations):
                self.progress.emit(f"Iteration {iteration + 1}/{self.num_iterations}")
                
                connected_skeleton, connections = self.connect_endpoints_single_iteration(
                    current_skeleton
                )
                
                all_connections.extend(connections)
                current_skeleton = connected_skeleton
                
                if len(connections) == 0:
                    self.progress.emit(f"No more connections possible after iteration {iteration + 1}")
                    break
                    
                self.progress.emit(f"Made {len(connections)} connections in iteration {iteration + 1}")
                
            self.progress.emit(f"Endpoint connection completed. Total connections: {len(all_connections)}")
            self.finished.emit(current_skeleton, all_connections)
            
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            
    def find_endpoints(self, skeleton):
        """Find endpoints of the skeleton."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = (skeleton == 1) & (neighbor_count == 1)
        y_coords, x_coords = np.where(endpoints)
        return list(zip(x_coords, y_coords))
        
    def find_intersections(self, skeleton):
        """Find intersection points."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        intersections = (skeleton == 1) & (neighbor_count > 2)
        y_coords, x_coords = np.where(intersections)
        return list(zip(x_coords, y_coords))
        
    def get_neighbors(self, point, skeleton):
        """Get all skeleton neighbors of a point."""
        x, y = point
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and 
                    skeleton[ny, nx] == 1):
                    neighbors.append((nx, ny))
        return neighbors
        
    def trace_branch_from_endpoint(self, endpoint, skeleton, max_length=50):
        """Trace a branch from an endpoint to get its direction and length."""
        path = [endpoint]
        current = endpoint
        visited = {current}
        
        while len(path) < max_length:
            neighbors = self.get_neighbors(current, skeleton)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if len(unvisited_neighbors) == 0:
                break
            elif len(unvisited_neighbors) == 1:
                current = unvisited_neighbors[0]
                visited.add(current)
                path.append(current)
            else:
                # Hit a junction, stop here
                break
                
        return path
        
    def get_branch_direction(self, branch_path):
        """Get the direction vector of a branch from its path."""
        if len(branch_path) < 2:
            return None
            
        if len(branch_path) <= 3:
            # Short branch - use simple direction
            start = branch_path[0]
            end = branch_path[-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
        else:
            # Longer branch - use weighted direction
            dx_sum = 0
            dy_sum = 0
            total_weight = 0
            
            for i in range(len(branch_path) - 1):
                weight = (i + 1) ** 2  # Quadratic weighting towards endpoint
                p1 = branch_path[i]
                p2 = branch_path[i + 1]
                
                segment_dx = p2[0] - p1[0]
                segment_dy = p2[1] - p1[1]
                
                dx_sum += segment_dx * weight
                dy_sum += segment_dy * weight
                total_weight += weight
                
            if total_weight > 0:
                dx = dx_sum / total_weight
                dy = dy_sum / total_weight
            else:
                return None
                
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return None
            
        return (-dx / length, -dy / length)
        
    def is_in_search_cone(self, from_point, to_point, direction, cone_angle_degrees=120):
        """Check if to_point is within the search cone."""
        if direction is None:
            return True
            
        # Vector from from_point to to_point
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return False
            
        dx /= length
        dy /= length
        
        # Calculate angle between direction and vector to target
        dot_product = direction[0] * dx + direction[1] * dy
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees <= cone_angle_degrees / 2
        
    def create_connection_line(self, point1, point2):
        """Create a line between two points using Bresenham's algorithm."""
        x1, y1 = point1
        x2, y2 = point2
        
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return points
        
    def connect_endpoints_single_iteration(self, skeleton):
        """Connect endpoints in a single iteration."""
        endpoints = self.find_endpoints(skeleton)
        intersections = self.find_intersections(skeleton)
        
        if len(endpoints) < 1:
            return skeleton, []
            
        # Get branch information for each endpoint
        endpoint_info = {}
        for endpoint in endpoints:
            branch_path = self.trace_branch_from_endpoint(endpoint, skeleton)
            endpoint_info[endpoint] = {
                'path': branch_path,
                'length': len(branch_path),
                'direction': self.get_branch_direction(branch_path)
            }
            
        connected_skeleton = skeleton.copy()
        connections_made = []
        used_endpoints = set()
        
        # Sort endpoints by branch length (longer branches get priority)
        sorted_endpoints = sorted(endpoints, key=lambda ep: endpoint_info[ep]['length'], reverse=True)
        
        for endpoint in sorted_endpoints:
            if endpoint in used_endpoints:
                continue
                
            branch_info = endpoint_info[endpoint]
            target_point, distance, target_type = self.find_closest_target_in_cone(
                endpoint, branch_info, endpoints, intersections, used_endpoints
            )
            
            if target_point:
                # Create connection
                connection_line = self.create_connection_line(endpoint, target_point)
                
                # Add connection to skeleton
                for x, y in connection_line:
                    if (0 <= y < connected_skeleton.shape[0] and 
                        0 <= x < connected_skeleton.shape[1]):
                        connected_skeleton[y, x] = 1
                        
                connections_made.append({
                    'from': endpoint,
                    'to': target_point,
                    'distance': distance,
                    'target_type': target_type,
                    'line': connection_line
                })
                
                # Mark endpoints as used
                used_endpoints.add(endpoint)
                if target_type == "endpoint":
                    used_endpoints.add(target_point)
                    
        return connected_skeleton, connections_made
        
    def find_closest_target_in_cone(self, endpoint, branch_info, all_endpoints, all_intersections, used_endpoints):
        """Find the closest valid target within the search cone."""
        direction = branch_info['direction']
        branch_length = branch_info['length']
        
        # Calculate search distance based on branch length
        max_search_distance = max(self.max_distance, int(branch_length * 1.5))
        
        best_target = None
        min_distance = float('inf')
        target_type = None
        
        # Check all endpoints
        for other_endpoint in all_endpoints:
            if other_endpoint == endpoint or other_endpoint in used_endpoints:
                continue
                
            distance = np.sqrt((other_endpoint[0] - endpoint[0])**2 + 
                             (other_endpoint[1] - endpoint[1])**2)
                             
            if distance > max_search_distance:
                continue
                
            if not self.is_in_search_cone(endpoint, other_endpoint, direction):
                continue
                
            if distance < min_distance:
                min_distance = distance
                best_target = other_endpoint
                target_type = "endpoint"
                
        # Check intersections
        for intersection in all_intersections:
            distance = np.sqrt((intersection[0] - endpoint[0])**2 + 
                             (intersection[1] - endpoint[1])**2)
                             
            if distance > max_search_distance:
                continue
                
            if not self.is_in_search_cone(endpoint, intersection, direction):
                continue
                
            if distance < min_distance:
                min_distance = distance
                best_target = intersection
                target_type = "intersection"
                
        return best_target, min_distance if best_target else None, target_type