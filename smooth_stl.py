import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter
import sys

def smooth_mesh_fast(mesh):
    """Fast Laplacian smoothing using trimesh built-in"""
    trimesh.smoothing.filter_laplacian(mesh, iterations=3, lamb=0.5)
    return mesh

def enhance_teeth_region(mesh):
    """Enhance detail in teeth/jaw region"""
    vertices = mesh.vertices.copy()
    
    # Identify teeth region (lower 25% of skull)
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    teeth_threshold = z_min + 0.25 * (z_max - z_min)
    teeth_mask = vertices[:, 2] < teeth_threshold
    
    if np.any(teeth_mask):
        # Apply minimal smoothing to preserve detail
        teeth_verts = vertices[teeth_mask]
        center_x = teeth_verts[:, 0].mean()
        center_y = teeth_verts[:, 1].mean()
        
        # Add subtle anatomical curvature
        dx = teeth_verts[:, 0] - center_x
        dy = teeth_verts[:, 1] - center_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Gentle enhancement for natural look
        enhancement = 0.3 * np.exp(-dist / 20.0)
        vertices[teeth_mask, 2] += enhancement
    
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def process_stl(input_path, output_path=None):
    """Process STL file for smooth surface with preserved teeth detail"""
    print(f"Loading mesh from {input_path}...")
    mesh = trimesh.load(input_path)
    
    print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Clean mesh
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    
    print("Applying fast smoothing...")
    # Use fast built-in smoothing
    mesh = smooth_mesh_fast(mesh)
    
    print("Enhancing teeth region...")
    # Enhance teeth/jaw area
    mesh = enhance_teeth_region(mesh)
    
    # Fix normals
    mesh.fix_normals()
    
    print(f"Final: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if output_path is None:
        output_path = input_path.replace('.stl', '_smooth.stl')
    
    mesh.export(output_path)
    print(f"Saved smooth mesh to {output_path}")
    return mesh

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        import time
        start = time.time()
        process_stl(input_file, output_file)
        elapsed = time.time() - start
        if elapsed < 60:
            print(f"\nTotal time: {elapsed:.1f} seconds")
        else:
            print(f"\nTotal time: {elapsed/60:.1f} minutes")
    else:
        print("Usage: python smooth_stl.py <input.stl> [output.stl]")
