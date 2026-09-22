import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from GFDFlow.GFDM import GFDMI_2D_problem

def run_example():
    # 1. Create a simple mesh
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    # 2. Define triangles (Delaunay)
    from scipy.spatial import Delaunay
    tri = Delaunay(coords)
    triangles = tri.simplices
    
    # 3. Define problem coefficients (Laplacian: D=1, F=1)
    # L = [A, B, C, D, E, F]
    L = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    source = lambda p: 0.0
    
    problem = GFDMI_2D_problem(coords, triangles, L, source)
    
    # 4. Define material
    interior_nodes = np.arange(len(coords))
    problem.material("main_material", lambda p: 1.0, interior_nodes)
    
    # 5. Define boundary conditions
    # Find boundary indices
    left_nodes = np.where(coords[:, 0] == 0)[0]
    right_nodes = np.where(coords[:, 0] == 1)[0]
    bottom_nodes = np.where(coords[:, 1] == 0)[0]
    top_nodes = np.where(coords[:, 1] == 1)[0]
    
    # Dirichlet conditions: u(0,y)=0, u(1,y)=100
    problem.dirichlet_boundary("left", left_nodes, lambda p: 0.0)
    problem.dirichlet_boundary("right", right_nodes, lambda p: 100.0)
    
    # Neumann conditions: du/dn=0 at top and bottom
    problem.neumann_boundary("top", lambda p: 1.0, top_nodes, lambda p: 0.0)
    problem.neumann_boundary("bottom", lambda p: 1.0, bottom_nodes, lambda p: 0.0)
    
    # 6. Assemble and solve
    print("Assembling system...")
    K, F = problem.discretization_K_F(continuous=True)
    
    print("Solving system...")
    U = sp.linalg.spsolve(K, F)
    
    # 7. Plot results
    plt.figure(figsize=(8, 6))
    cont = plt.tricontourf(coords[:, 0], coords[:, 1], triangles, U, cmap="inferno", levels=20)
    plt.colorbar(cont, label="u")
    plt.title("GFDM Solution for 2D Laplacian")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    run_example()
