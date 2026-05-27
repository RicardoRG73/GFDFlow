import numpy as np
import scipy.sparse as sp
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from GFDFlow.GFDM import GFDMI_2D_problem

def test_initialization():
    print("Testing initialization...")
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    triangles = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    L = np.array([0, 0, 0, 1, 0, 1])
    source = lambda p: 0
    
    problem = GFDMI_2D_problem(coords, triangles, L, source)
    assert problem.coords.shape == (5, 2)
    assert problem.triangles.shape == (4, 3)
    print("Initialization OK.")

def test_support_nodes():
    print("Testing support nodes...")
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    triangles = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    L = np.array([0, 0, 0, 1, 0, 1])
    source = lambda p: 0
    
    problem = GFDMI_2D_problem(coords, triangles, L, source)
    s_nodes = problem.support_nodes(4)
    assert len(s_nodes) >= 5
    assert 4 in s_nodes
    print("Support nodes OK.")

def test_simple_laplacian():
    print("Testing simple Laplacian assembly...")
    # Simple rectangular mesh
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Simple manual triangles for a grid
    triangles = []
    for j in range(4):
        for i in range(4):
            # Two triangles per square
            n1 = j*5 + i
            n2 = j*5 + i + 1
            n3 = (j+1)*5 + i
            n4 = (j+1)*5 + i + 1
            triangles.append([n1, n2, n4])
            triangles.append([n1, n4, n3])
    triangles = np.array(triangles)
    
    L = np.array([0, 0, 0, 1, 0, 1]) # Laplacian
    source = lambda p: 0
    
    problem = GFDMI_2D_problem(coords, triangles, L, source)
    
    # Material
    interior_nodes = np.arange(len(coords))
    problem.material("mat", lambda p: 1.0, interior_nodes)
    
    # Boundaries (all Dirichlet for simplicity)
    problem.dirichlet_boundary("all", np.arange(len(coords)), lambda p: p[0] + p[1])
    
    K, F = problem.discretization_K_F(continuous=True)
    assert K.shape == (25, 25)
    assert len(F) == 25
    print("Laplacian assembly OK.")

if __name__ == "__main__":
    try:
        test_initialization()
        test_support_nodes()
        test_simple_laplacian()
        print("\nAll verification tests passed!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
