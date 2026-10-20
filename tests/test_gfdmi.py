import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from GFDFlow.GFDM import GFDMI_2D_problem
from GFDFlow.utils import compute_M_matrix, compute_M_matrix_neumann

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


def test_neumann_stencils_and_m_pinv():
    """Verify that Neumann boundary nodes compute M_pinv via compute_M_matrix_neumann."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    triangles = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    normal_vectors = np.zeros((5, 2))
    # Assign outward normal to node 1 (Neumann boundary)
    normal_vectors[1] = [1.0, 0.0]
    L = np.array([0, 0, 0, 1, 0, 1])
    source = lambda p: 0.0

    problem = GFDMI_2D_problem(
        coords,
        triangles,
        normal_vectors,
        L,
        source,
        support_stencils=None,
        M_pinv=None,
    )

    # Node 1 is a Neumann boundary node (norm > 1e-10)
    stencil_1 = problem.support_stencils[1]
    expected_M_neu = compute_M_matrix_neumann(1, stencil_1, coords, normal_vectors[1])
    expected_pinv_neu = np.linalg.pinv(expected_M_neu)
    # Stencil size + 1 (for ghost node column in M, row in pinv)
    assert problem.M_pinv[1].shape == (len(stencil_1) + 1, 6)
    np.testing.assert_allclose(problem.M_pinv[1], expected_pinv_neu)

    # Node 0 is a regular / non-Neumann node (norm == 0)
    stencil_0 = problem.support_stencils[0]
    expected_M_std = compute_M_matrix(0, stencil_0, coords)
    expected_pinv_std = np.linalg.pinv(expected_M_std)
    assert problem.M_pinv[0].shape == (len(stencil_0), 6)
    np.testing.assert_allclose(problem.M_pinv[0], expected_pinv_std)


if __name__ == "__main__":
    test_gfdmi_initialization()
    test_support_nodes()
    test_simple_laplacian()
    test_neumann_stencils_and_m_pinv()
    print("All GFDMI tests passed successfully.")
