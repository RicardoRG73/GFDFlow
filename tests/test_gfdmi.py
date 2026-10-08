import numpy as np
import pytest
from GFDFlow.GFDM import GFDMI_2D_problem

def test_gfdmi_initialization():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    triangles = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    normal_vectors = np.zeros_like(coords)
    L = np.array([0, 0, 0, 1, 0, 1])
    source = lambda p: 0
    
    problem = GFDMI_2D_problem(coords, triangles, normal_vectors, L, source)
    assert problem.coords.shape == (5, 2)
    assert problem.triangles.shape == (4, 3)

def test_support_nodes():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    triangles = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    normal_vectors = np.zeros_like(coords)
    L = np.array([0, 0, 0, 1, 0, 1])
    source = lambda p: 0
    
    problem = GFDMI_2D_problem(coords, triangles, normal_vectors, L, source)
    s_nodes = problem.support_stencils[4]
    assert len(s_nodes) >= 5
    assert 4 in s_nodes

def test_simple_laplacian():
    # Simple rectangular mesh
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    from scipy.spatial import Delaunay
    tri = Delaunay(coords)
    triangles = tri.simplices
    normal_vectors = np.zeros_like(coords)
    
    L = np.array([0, 0, 0, 1, 0, 1]) # Laplacian
    source = lambda p: 0
    
    problem = GFDMI_2D_problem(coords, triangles, normal_vectors, L, source)
    
    # Material
    interior_nodes = np.arange(len(coords))
    problem.material("mat", lambda p: 1.0, interior_nodes)
    
    # Boundaries (all Dirichlet for simplicity)
    problem.dirichlet_boundary("all", np.arange(len(coords)), lambda p: p[0] + p[1])
    
    K, F = problem.discretization_K_F(continuous=True)
    assert K.shape == (25, 25)
    assert len(F) == 25
