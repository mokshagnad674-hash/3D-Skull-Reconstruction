import trimesh
import numpy as np
import sys

def remove_spikes_and_artifacts(mesh):
    """Remove spikes, plates, and artifacts from mesh"""
    print("Removing spikes and artifacts...")
    
    # 1. Remove isolated vertices and small components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Keep only the largest component
        largest = max(components, key=lambda c: len(c.vertices))
        mesh = largest
        print(f"Removed {len(components)-1} disconnected components")
    
    # 2. Remove long thin triangles (spikes)
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Calculate edge lengths for each face
    valid_faces = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = np.linalg.norm(v1 - v0)
        edge2 = np.linalg.norm(v2 - v1)
        edge3 = np.linalg.norm(v0 - v2)
        
        # Calculate aspect ratio (longest edge / shortest edge)
        edges = [edge1, edge2, edge3]
        aspect_ratio = max(edges) / (min(edges) + 1e-7)
        
        # Remove faces with extreme aspect ratios (spikes)
        if aspect_ratio < 10.0:  # Keep only reasonable triangles
            valid_faces.append(face)
    
    removed = len(faces) - len(valid_faces)
    if removed > 0:
        print(f"Removed {removed} spike/artifact faces")
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(valid_faces))
    
    # 3. Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    
    return mesh

def remove_outlier_vertices(mesh, std_threshold=3.0):
    """Remove vertices that are statistical outliers (far from main mesh)"""
    print("Removing outlier vertices...")
    
    vertices = mesh.vertices
    center = vertices.mean(axis=0)
    distances = np.linalg.norm(vertices - center, axis=1)
    
    mean_dist = distances.mean()
    std_dist = distances.std()
    
    # Find outliers (vertices too far from center)
    outlier_threshold = mean_dist + std_threshold * std_dist
    valid_mask = distances < outlier_threshold
    
    # Get faces that only use valid vertices
    valid_vertex_indices = np.where(valid_mask)[0]
    valid_vertex_set = set(valid_vertex_indices)
    
    valid_faces = []
    for face in mesh.faces:
        if all(v in valid_vertex_set for v in face):
            valid_faces.append(face)
    
    removed = len(mesh.vertices) - valid_mask.sum()
    if removed > 0:
        print(f"Removed {removed} outlier vertices")
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(valid_faces))
        mesh.remove_unreferenced_vertices()
    
    return mesh

def fix_mesh(input_path, output_path=None):
    """Complete mesh fixing pipeline"""
    print(f"Loading mesh from {input_path}...")
    mesh = trimesh.load(input_path)
    
    print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 1: Basic cleanup
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    
    # Step 2: Remove spikes and artifacts
    mesh = remove_spikes_and_artifacts(mesh)
    
    # Step 3: Remove outlier vertices
    mesh = remove_outlier_vertices(mesh, std_threshold=2.5)
    
    # Step 4: Fill holes
    mesh.fill_holes()
    
    # Step 5: Smooth the mesh
    print("Smoothing mesh...")
    trimesh.smoothing.filter_laplacian(mesh, iterations=5, lamb=0.5)
    
    # Step 6: Fix normals
    mesh.fix_normals()
    
    print(f"Final: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if output_path is None:
        output_path = input_path.replace('.stl', '_fixed.stl')
    
    mesh.export(output_path)
    print(f"Fixed mesh saved to {output_path}")
    return mesh

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        fix_mesh(input_file, output_file)
    else:
        print("Usage: python fix_mesh.py <input.stl> [output.stl]")
