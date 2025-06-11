from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import skeletonize
import cv2
import glob
import os

def find_endpoints(skeleton):
    """Find endpoints of the skeleton."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = (skeleton == 1) & (neighbor_count == 1)
    y_coords, x_coords = np.where(endpoints)
    return list(zip(x_coords, y_coords))

def find_intersection_points(skeleton):
    """Find intersection points (junctions) in the skeleton."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    intersections = (skeleton == 1) & (neighbor_count > 2)
    y_coords, x_coords = np.where(intersections)
    return list(zip(x_coords, y_coords))

def get_neighbors(point, skeleton):
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

def trace_branch_from_endpoint(endpoint, skeleton, max_length=50):
    """Trace a branch from an endpoint to get its direction and length."""
    path = [endpoint]
    current = endpoint
    visited = {current}
    
    while len(path) < max_length:
        neighbors = get_neighbors(current, skeleton)
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

def get_branch_direction(branch_path):
    """Get the direction vector of a branch from its path."""
    if len(branch_path) < 2:
        return None
    
    # Use weighted average focusing on the last part of the branch
    if len(branch_path) <= 3:
        # Short branch - use simple direction
        start = branch_path[0]
        end = branch_path[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
    else:
        # Longer branch - use weighted direction with emphasis on endpoint
        dx_sum = 0
        dy_sum = 0
        total_weight = 0
        
        for i in range(len(branch_path) - 1):
            # Weight increases towards the endpoint
            weight = (i + 1) ** 2  # Quadratic weighting
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
    
    return (dx / length, dy / length)

def is_in_search_cone(from_point, to_point, direction, cone_angle_degrees=120):
    """Check if to_point is within the search cone from from_point in the given direction."""
    if direction is None:
        return True  # If no direction, allow all angles
    
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
    
    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees <= cone_angle_degrees / 2

def find_closest_endpoint_in_cone(endpoint, branch_path, all_endpoints, all_intersections, skeleton, baseline_distance=25):
    """Find the closest endpoint or intersection within the search cone."""
    branch_length = len(branch_path)
    direction = get_branch_direction(branch_path)
    
    # Calculate maximum search distance based on branch length
    max_search_distance = max(baseline_distance, int(branch_length * 1.5))
    
    print(f"    Endpoint at {endpoint}: branch length {branch_length}, max search {max_search_distance}")
    
    best_target = None
    min_distance = float('inf')
    target_type = None
    
    # Check all endpoints
    for other_endpoint in all_endpoints:
        if other_endpoint == endpoint:
            continue
        
        # Calculate distance
        distance = np.sqrt((other_endpoint[0] - endpoint[0])**2 + 
                          (other_endpoint[1] - endpoint[1])**2)
        
        # Skip if too far
        if distance > max_search_distance:
            continue
        
        # Check if in search cone (120 degrees)
        if not is_in_search_cone(endpoint, other_endpoint, direction, cone_angle_degrees=120):
            continue
        
        # This is a valid target
        if distance < min_distance:
            min_distance = distance
            best_target = other_endpoint
            target_type = "endpoint"
    
    # Also check all intersections
    for intersection in all_intersections:
        # Calculate distance
        distance = np.sqrt((intersection[0] - endpoint[0])**2 + 
                          (intersection[1] - endpoint[1])**2)
        
        # Skip if too far
        if distance > max_search_distance:
            continue
        
        # Check if in search cone (120 degrees)
        if not is_in_search_cone(endpoint, intersection, direction, cone_angle_degrees=120):
            continue
        
        # This is a valid target
        if distance < min_distance:
            min_distance = distance
            best_target = intersection
            target_type = "intersection"
    
    if best_target:
        print(f"      Found {target_type} at {best_target}, distance: {min_distance:.1f}")
    else:
        print(f"      No valid target found in cone")
    
    return best_target, min_distance if best_target else None, target_type

def create_connection_line(point1, point2):
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

def connect_endpoints(skeleton, baseline_distance=15):
    """Connect endpoints to other nearby endpoints or intersections within search cones."""
    endpoints = find_endpoints(skeleton)
    intersections = find_intersection_points(skeleton)
    print(f"Found {len(endpoints)} endpoints and {len(intersections)} intersections to process")
    
    if len(endpoints) < 1:
        print("No endpoints to connect")
        return skeleton, []
    
    # Get branch information for each endpoint
    endpoint_info = {}
    for endpoint in endpoints:
        branch_path = trace_branch_from_endpoint(endpoint, skeleton)
        endpoint_info[endpoint] = {
            'path': branch_path,
            'length': len(branch_path),
            'direction': get_branch_direction(branch_path)
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
        
        # Find closest valid target (endpoint or intersection)
        target_point, distance, target_type = find_closest_endpoint_in_cone(
            endpoint, branch_info['path'], endpoints, intersections, skeleton, baseline_distance
        )
        
        if target_point:
            # Don't connect to already used endpoints, but intersections are always fair game
            if target_type == "endpoint" and target_point in used_endpoints:
                continue
            
            # Create connection
            connection_line = create_connection_line(endpoint, target_point)
            
            # Add connection to skeleton
            for x, y in connection_line:
                if (0 <= y < connected_skeleton.shape[0] and 
                    0 <= x < connected_skeleton.shape[1]):
                    connected_skeleton[y, x] = 1
            
            connections_made.append({
                'from': endpoint,
                'to': target_point,
                'distance': distance,
                'from_length': branch_info['length'],
                'target_type': target_type,
                'line': connection_line
            })
            
            # Mark endpoint as used (but not intersections, they can have multiple connections)
            used_endpoints.add(endpoint)
            if target_type == "endpoint":
                used_endpoints.add(target_point)
            
            print(f"  ✓ Connected {endpoint} to {target_type} {target_point} (distance: {distance:.1f})")
    
    print(f"Made {len(connections_made)} connections")
    return connected_skeleton, connections_made

def process_skeleton_file(skeleton_path, num_iterations=3):
    """Process a single skeleton file to connect endpoints with multiple iterations."""
    # Load skeleton
    skeleton_img = Image.open(skeleton_path).convert('L')
    skeleton_array = np.array(skeleton_img)
    skeleton = (skeleton_array > 128).astype(np.uint8)
    
    print(f"\nProcessing: {skeleton_path}")
    
    # Get initial statistics
    initial_endpoints = find_endpoints(skeleton)
    initial_intersections = find_intersection_points(skeleton)
    print(f"Initial: {len(initial_endpoints)} endpoints, {len(initial_intersections)} intersections")
    
    # Create figure for all iterations
    fig, axes = plt.subplots(num_iterations + 1, 3, figsize=(15, 5*(num_iterations + 1)))
    
    # Show original in first row
    axes[0, 0].imshow(skeleton, cmap='gray')
    if initial_endpoints:
        end_y, end_x = zip(*[(p[1], p[0]) for p in initial_endpoints])
        axes[0, 0].scatter(end_x, end_y, c='blue', s=30, marker='s', alpha=0.8, label='Endpoints')
    if initial_intersections:
        int_y, int_x = zip(*[(p[1], p[0]) for p in initial_intersections])
        axes[0, 0].scatter(int_x, int_y, c='red', s=30, marker='o', alpha=0.8, label='Intersections')
    axes[0, 0].set_title(f'Original\n{len(initial_endpoints)} endpoints, {len(initial_intersections)} intersections')
    axes[0, 0].legend()
    axes[0, 0].axis('off')
    
    # Hide unused plots in first row
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    # Process multiple iterations
    current_skeleton = skeleton.copy()
    all_connections = []
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Connect endpoints for this iteration
        connected_skeleton, connections = connect_endpoints(current_skeleton)
        all_connections.extend(connections)
        
        # Get statistics for this iteration
        iter_endpoints = find_endpoints(connected_skeleton)
        iter_intersections = find_intersection_points(connected_skeleton)
        
        # Show results for this iteration
        row = iteration + 1
        
        # Original with current endpoints
        axes[row, 0].imshow(current_skeleton, cmap='gray')
        cur_endpoints = find_endpoints(current_skeleton)
        cur_intersections = find_intersection_points(current_skeleton)
        if cur_endpoints:
            end_y, end_x = zip(*[(p[1], p[0]) for p in cur_endpoints])
            axes[row, 0].scatter(end_x, end_y, c='blue', s=30, marker='s', alpha=0.8)
        if cur_intersections:
            int_y, int_x = zip(*[(p[1], p[0]) for p in cur_intersections])
            axes[row, 0].scatter(int_x, int_y, c='red', s=30, marker='o', alpha=0.8)
        axes[row, 0].set_title(f'Before Iteration {iteration + 1}\n{len(cur_endpoints)} endpoints')
        axes[row, 0].axis('off')
        
        # Connections made this iteration
        axes[row, 1].imshow(current_skeleton, cmap='gray')
        endpoint_connections = [c for c in connections if c['target_type'] == 'endpoint']
        intersection_connections = [c for c in connections if c['target_type'] == 'intersection']
        
        # Draw connections
        for i, conn in enumerate(endpoint_connections):
            from_point = conn['from']
            to_point = conn['to']
            color = plt.cm.Set3(i % 12)
            axes[row, 1].plot([from_point[0], to_point[0]], [from_point[1], to_point[1]], 
                        color=color, linewidth=3, alpha=0.8, linestyle='-')
        
        for i, conn in enumerate(intersection_connections):
            from_point = conn['from']
            to_point = conn['to']
            color = plt.cm.Set1(i % 9)
            axes[row, 1].plot([from_point[0], to_point[0]], [from_point[1], to_point[1]], 
                        color=color, linewidth=3, alpha=0.8, linestyle='--')
        
        axes[row, 1].set_title(f'Iteration {iteration + 1} Connections\n{len(connections)} made')
        axes[row, 1].axis('off')
        
        # Result after this iteration
        axes[row, 2].imshow(connected_skeleton, cmap='gray')
        if iter_endpoints:
            end_y, end_x = zip(*[(p[1], p[0]) for p in iter_endpoints])
            axes[row, 2].scatter(end_x, end_y, c='blue', s=30, marker='s', alpha=0.8)
        if iter_intersections:
            int_y, int_x = zip(*[(p[1], p[0]) for p in iter_intersections])
            axes[row, 2].scatter(int_x, int_y, c='red', s=30, marker='o', alpha=0.8)
        axes[row, 2].set_title(f'After Iteration {iteration + 1}\n{len(iter_endpoints)} endpoints')
        axes[row, 2].axis('off')
        
        # Update for next iteration
        current_skeleton = connected_skeleton.copy()
        
        # Break if no more connections can be made
        if len(connections) == 0:
            print(f"No more connections possible after iteration {iteration + 1}")
            break
    
    plt.suptitle(f'Endpoint Connection ({num_iterations} iterations): {os.path.basename(skeleton_path)}')
    plt.tight_layout()
    plt.show()
    
    return current_skeleton, all_connections

def save_connected_results(identifier, connected_skeleton, connections):
    """Save connected results to a folder."""
    output_folder = "connected_skeletons"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save connected skeleton
    skeleton_path = os.path.join(output_folder, f"{identifier}_connected.png")
    skeleton_img = Image.fromarray((connected_skeleton * 255).astype(np.uint8))
    skeleton_img.save(skeleton_path)
    
    # Save connection metadata
    metadata_path = os.path.join(output_folder, f"{identifier}_connections.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Connections made: {len(connections)}\n\n")
        
        endpoint_connections = [c for c in connections if c['target_type'] == 'endpoint']
        intersection_connections = [c for c in connections if c['target_type'] == 'intersection']
        
        f.write(f"Endpoint-to-endpoint connections: {len(endpoint_connections)}\n")
        f.write(f"Endpoint-to-intersection connections: {len(intersection_connections)}\n\n")
        
        for i, conn in enumerate(connections):
            f.write(f"Connection {i+1}:\n")
            f.write(f"  From: {conn['from']}\n")
            f.write(f"  To: {conn['to']} ({conn['target_type']})\n")
            f.write(f"  Distance: {conn['distance']:.2f}\n")
            f.write(f"  Branch length: {conn['from_length']}\n\n")
    
    print(f"Connected results saved to '{output_folder}/' folder")

def main():
    """Process all skeleton files from the skeletized folder."""
    # Changed pattern to look in subfolders for final_skeleton.png
    skeleton_pattern = "skeletized/*/final_skeleton.png"
    skeleton_files = glob.glob(skeleton_pattern)
    
    if not skeleton_files:
        print("No skeleton files found in skeletized/*/ folders!")
        print("Make sure to run blob_to_trace.py first to generate skeletons.")
        return

    total_initial_endpoints = 0
    total_final_endpoints = 0
    total_connections = 0
    
    print(f"Found {len(skeleton_files)} skeleton files to process.")
    print(f"Connected results will be saved to 'connected_skeletons/' folder.\n")
    
    for skeleton_file in skeleton_files:
        try:
            connected_skeleton, connections = process_skeleton_file(skeleton_file, num_iterations=3)
            
            if connected_skeleton is not None:
                # Count endpoints before and after
                initial_endpoints = len(find_endpoints(np.array(Image.open(skeleton_file).convert('L')) > 128))
                final_endpoints = len(find_endpoints(connected_skeleton))
                
                total_initial_endpoints += initial_endpoints
                total_final_endpoints += final_endpoints
                total_connections += len(connections)
                
                # Extract identifier from folder name instead of filename
                folder_name = os.path.basename(os.path.dirname(skeleton_file))
                identifier = folder_name  # Uses the folder name as identifier
                
                save_connected_results(identifier, connected_skeleton, connections)
                print(f"Completed processing: {identifier}")
                print(f"Reduced endpoints from {initial_endpoints} to {final_endpoints}")
                
        except Exception as e:
            print(f"Error processing {skeleton_file}: {e}")
            continue
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total initial endpoints: {total_initial_endpoints}")
    print(f"Total final endpoints: {total_final_endpoints}")
    print(f"Total connections made: {total_connections}")
    print(f"Overall reduction: {total_initial_endpoints - total_final_endpoints} endpoints ({((total_initial_endpoints - total_final_endpoints) / total_initial_endpoints * 100):.1f}%)")

if __name__ == "__main__":
    main()