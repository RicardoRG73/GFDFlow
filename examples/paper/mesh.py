"""Mesh Generation for Five-Layer Dam Seepage Problem (Chavez et al. 2023).

This module constructs the computational geometry and generates the unstructured
triangular mesh for the five-layered embankment dam with clay core and foundation
described in Section 4.2.3 of Chavez et al. (2023).

The domain models three distinct materials across five subdomains:
- Rockfill shells (upstream and downstream)
- Central clay core with cutoff wall
- Soil foundation

It precomputes the outward normal vectors at boundaries, extracts internal material
interfaces and triple junction nodes, and precalculates the GFDM support stencils
and pseudo-inverse matrices (M_pinv) for each node.

Outputs
-------
- JSON: `examples/paper/mesh1.json`
- Figures: `examples/paper/figures/{geometry,mesh,nodes,normal_vectors}.png`
"""

import json
import os
from typing import Dict, List, Tuple

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from GFDFlow.utils import (
    compute_M_matrix,
    compute_M_matrix_neumann,
    compute_normal_vectors,
    get_support_nodes_2D,
)
from GFDFlow.visualization import (
    plot_geometry,
    plot_mesh,
    plot_nodes,
    plot_normal_vectors,
)


def create_dam_geometry() -> Tuple[cfg.Geometry, Dict[str, int]]:
    """Create the CalfEM geometry model for the five-layer dam.

    The coordinates and dimensions strictly match Section 4.2.3 of
    Chavez et al. (2023):
    - Height: 10 m, Crest width: 4 m (x = 23 to 27 m)
    - Slopes: 2:1 upstream and downstream
    - Reservoir level: y = 8 m (x = 19 m)
    - Foundation depth: 4 m (y = -4 to 0 m), horizontal extent: x = 3 to 60 m
    - Clay core cutoff: base width 2 m (x = 24 to 26 m) at y = -4 m

    Returns
    -------
    g : calfem.geometry.Geometry
        The constructed CalfEM geometry object.
    markers : dict of str to int
        Dictionary of boundary and material marker IDs.
    """
    g = cfg.Geometry()

    # Control points
    g.point([3, 0])      # P0: Upstream toe
    g.point([20, 0])     # P1: Triple junction 1 (upstream core-foundation)
    g.point([30, 0])     # P2: Triple junction 2 (downstream core-foundation)
    g.point([40, 0])     # P3: Downstream filter/drain start
    g.point([47, 0])     # P4: Downstream dam toe
    g.point([27, 10])    # P5: Crest right corner
    g.point([23, 10])    # P6: Crest left corner
    g.point([19, 8])     # P7: Upstream reservoir water surface level
    g.point([3, -4])     # P8: Foundation bottom-left corner
    g.point([24, -4])    # P9: Core cutoff base left
    g.point([26, -4])    # P10: Core cutoff base right
    g.point([60, -4])    # P11: Foundation bottom-right corner
    g.point([60, 0])     # P12: Foundation top-right ground surface

    # Boundary and interface markers
    markers = {
        "left": 10,
        "right": 11,
        "bottom": 12,
        "top": 13,
        "interface_a": 14,
        "interface_b": 15,
        "interface_c": 16,
        "interface_d": 17,
        "interface_e": 18,
        "interface_f": 19,
        "rock": 1,
        "clay": 2,
        "foundation": 3,
    }

    # Splines: Boundaries and internal interfaces
    g.spline([0, 1], marker=markers["interface_a"])
    g.spline([2, 3], marker=markers["interface_b"])
    g.spline([3, 4], marker=markers["right"])
    g.spline([4, 5], marker=markers["top"])
    g.spline([5, 6], marker=markers["top"])
    g.spline([6, 7], marker=markers["top"])
    g.spline([7, 0], marker=markers["left"])

    # Foundation boundaries
    g.spline([0, 8], marker=markers["left"])
    g.spline([8, 9], marker=markers["bottom"])
    g.spline([9, 10], marker=markers["bottom"])
    g.spline([10, 11], marker=markers["bottom"])
    g.spline([11, 12], marker=markers["bottom"])
    g.spline([12, 4], marker=markers["right"])

    # Core internal interfaces
    g.spline([6, 1], marker=markers["interface_c"])
    g.spline([1, 9], marker=markers["interface_d"])
    g.spline([5, 2], marker=markers["interface_f"])
    g.spline([2, 10], marker=markers["interface_e"])

    # Surfaces
    g.surface([0, 13, 5, 6], marker=markers["rock"])               # Subdomain 1: Rockfill upstream
    g.surface([1, 2, 3, 15], marker=markers["rock"])               # Subdomain 2: Rockfill downstream
    g.surface([7, 8, 14, 0], marker=markers["foundation"])         # Subdomain 3: Foundation upstream
    g.surface([10, 11, 12, 2, 1, 16], marker=markers["foundation"])# Subdomain 4: Foundation downstream
    g.surface([13, 14, 9, 16, 15, 4], marker=markers["clay"])      # Subdomain 5: Clay core & cutoff

    return g, markers


def generate_mesh(
    g: cfg.Geometry,
    el_size_factor: float = 0.655
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], List, Dict, List]:
    """Generate the 2D triangular mesh using CalfEM and Gmsh.

    Parameters
    ----------
    g : calfem.geometry.Geometry
        The geometric model.
    el_size_factor : float, optional
        Mesh element size scaling factor, by default 0.655 to yield ~1500 nodes.

    Returns
    -------
    coords : npt.NDArray[np.float64]
        Nodal coordinates with shape (N, 2).
    triangles : npt.NDArray[np.int_]
        Triangle vertex indices with shape (M, 3).
    edof : list
        Element degrees of freedom from CalfEM.
    bdofs : dict
        Boundary degrees of freedom per curve marker.
    element_markers : list
        Material marker assigned to each triangle element.
    """
    mesh = cfm.GmshMesh(g)
    mesh.el_type = 2          # 3-node triangle
    mesh.dofs_per_node = 1
    mesh.el_size_factor = el_size_factor

    coords, edof, dofs, bdofs, element_markers = mesh.create()
    verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
        coords,
        edof,
        mesh.dofs_per_node,
        mesh.el_type,
    )
    triangles = faces.copy()
    return coords, triangles, edof, bdofs, element_markers


def extract_node_groups(
    coords: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    bdofs: Dict[int, List[int]],
    element_markers: List[int],
    markers: Dict[str, int],
) -> Dict[str, npt.NDArray[np.int_]]:
    """Partition mesh nodes into outer boundaries, interfaces, and material domains.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Nodal coordinates (N, 2).
    triangles : npt.NDArray[np.int_]
        Triangle connectivity (M, 3).
    bdofs : dict
        Boundary DOFs extracted from CalfEM.
    element_markers : list
        Material markers for each element.
    markers : dict
        Mapping of semantic names to marker integers.

    Returns
    -------
    node_groups : dict of str to npt.NDArray[np.int_]
        Dictionary containing grouped node indices.
    """
    left_nodes = np.asarray(bdofs[markers["left"]]) - 1
    right_nodes = np.asarray(bdofs[markers["right"]]) - 1
    top_nodes = np.asarray(bdofs[markers["top"]]) - 1
    bottom_nodes = np.asarray(bdofs[markers["bottom"]]) - 1

    ia_nodes = np.asarray(bdofs[markers["interface_a"]]) - 1
    ib_nodes = np.asarray(bdofs[markers["interface_b"]]) - 1
    ic_nodes = np.asarray(bdofs[markers["interface_c"]]) - 1
    id_nodes = np.asarray(bdofs[markers["interface_d"]]) - 1
    ie_nodes = np.asarray(bdofs[markers["interface_e"]]) - 1
    if_nodes = np.asarray(bdofs[markers["interface_f"]]) - 1

    # Remove corner duplicates from outer boundaries
    top_nodes = np.setdiff1d(top_nodes, left_nodes)
    top_nodes = np.setdiff1d(top_nodes, right_nodes)
    bottom_nodes = np.setdiff1d(bottom_nodes, left_nodes)
    bottom_nodes = np.setdiff1d(bottom_nodes, right_nodes)

    outer_boundaries = np.hstack((left_nodes, right_nodes, bottom_nodes, top_nodes))

    # Strip boundary nodes from internal interfaces
    ia_nodes = np.setdiff1d(ia_nodes, outer_boundaries)
    ib_nodes = np.setdiff1d(ib_nodes, outer_boundaries)
    ic_nodes = np.setdiff1d(ic_nodes, outer_boundaries)
    id_nodes = np.setdiff1d(id_nodes, outer_boundaries)
    ie_nodes = np.setdiff1d(ie_nodes, outer_boundaries)
    if_nodes = np.setdiff1d(if_nodes, outer_boundaries)

    all_boundaries = np.hstack((
        outer_boundaries,
        ia_nodes,
        ib_nodes,
        ic_nodes,
        id_nodes,
        ie_nodes,
        if_nodes,
    ))

    # Extract interior material nodes
    elem_markers_arr = np.array(element_markers)
    rock_nodes = np.setdiff1d(
        triangles[elem_markers_arr == markers["rock"]].flatten(),
        all_boundaries,
    )
    clay_nodes = np.setdiff1d(
        triangles[elem_markers_arr == markers["clay"]].flatten(),
        all_boundaries,
    )
    foundation_nodes = np.setdiff1d(
        triangles[elem_markers_arr == markers["foundation"]].flatten(),
        all_boundaries,
    )

    # Detect triple junction nodes at [20, 0] and [30, 0]
    tj1_idx = int(np.where(np.linalg.norm(coords - np.array([20.0, 0.0]), axis=1) < 1e-4)[0][0])
    tj2_idx = int(np.where(np.linalg.norm(coords - np.array([30.0, 0.0]), axis=1) < 1e-4)[0][0])

    # Remove triple junction nodes from regular interface arrays
    ia_nodes = np.setdiff1d(ia_nodes, [tj1_idx])
    ic_nodes = np.setdiff1d(ic_nodes, [tj1_idx])
    id_nodes = np.setdiff1d(id_nodes, [tj1_idx])

    ib_nodes = np.setdiff1d(ib_nodes, [tj2_idx])
    ie_nodes = np.setdiff1d(ie_nodes, [tj2_idx])
    if_nodes = np.setdiff1d(if_nodes, [tj2_idx])

    return {
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "bottom_nodes": bottom_nodes,
        "top_nodes": top_nodes,
        "interface_a_nodes": ia_nodes,
        "interface_b_nodes": ib_nodes,
        "interface_c_nodes": ic_nodes,
        "interface_d_nodes": id_nodes,
        "interface_e_nodes": ie_nodes,
        "interface_f_nodes": if_nodes,
        "rock_nodes": rock_nodes,
        "clay_nodes": clay_nodes,
        "foundation_nodes": foundation_nodes,
        "tj1": np.array([tj1_idx]),
        "tj2": np.array([tj2_idx]),
    }


def compute_stencils_and_normals(
    coords: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    node_groups: Dict[str, npt.NDArray[np.int_]],
) -> Tuple[npt.NDArray[np.float64], Dict[int, npt.NDArray[np.int_]], Dict[int, npt.NDArray[np.float64]]]:
    """Compute outward unit normal vectors, support stencils, and star matrices.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Nodal coordinates (N, 2).
    triangles : npt.NDArray[np.int_]
        Triangle connectivity (M, 3).
    node_groups : dict of str to npt.NDArray[np.int_]
        Identified node groups.

    Returns
    -------
    normal_vecs : npt.NDArray[np.float64]
        Outward unit normal vectors with shape (N, 2).
    support_stencils : dict of int to npt.NDArray[np.int_]
        Support node indices for each central node.
    M_pinv : dict of int to npt.NDArray[np.float64]
        Precomputed pseudo-inverse matrices for each central node.
    """
    boundaries_with_normals = (
        node_groups["bottom_nodes"],
        node_groups["top_nodes"],
        node_groups["interface_a_nodes"],
        node_groups["interface_b_nodes"],
        node_groups["interface_c_nodes"],
        node_groups["interface_d_nodes"],
        node_groups["interface_e_nodes"],
        node_groups["interface_f_nodes"],
    )

    normal_vecs = np.zeros((coords.shape[0], 2))
    for nodes in boundaries_with_normals:
        if len(nodes) > 0:
            normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

    # Prescribed normal direction for triple junction nodes
    tj1_idx = int(node_groups["tj1"][0])
    tj2_idx = int(node_groups["tj2"][0])
    normal_vecs[tj1_idx] = np.array([-1.0, 0.0])
    normal_vecs[tj2_idx] = np.array([-1.0, 0.0])

    support_stencils: Dict[int, npt.NDArray[np.int_]] = {}
    M_pinv: Dict[int, npt.NDArray[np.float64]] = {}

    for i in range(coords.shape[0]):
        support_stencils[i] = get_support_nodes_2D(i, triangles)
        if np.linalg.norm(normal_vecs[i]) > 1e-10:
            M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vecs[i])
        else:
            M = compute_M_matrix(i, support_stencils[i], coords)
        M_pinv[i] = np.linalg.pinv(M)

    return normal_vecs, support_stencils, M_pinv


def export_mesh_data(
    filepath: str,
    coords: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    normal_vecs: npt.NDArray[np.float64],
    node_groups: Dict[str, npt.NDArray[np.int_]],
    support_stencils: Dict[int, npt.NDArray[np.int_]],
    M_pinv: Dict[int, npt.NDArray[np.float64]],
) -> None:
    """Save the serialized mesh, stencils, and matrices to a JSON file.

    Parameters
    ----------
    filepath : str
        Destination path for JSON file.
    coords : npt.NDArray[np.float64]
        Nodal coordinates.
    triangles : npt.NDArray[np.int_]
        Triangle connectivity.
    normal_vecs : npt.NDArray[np.float64]
        Normal vectors array.
    node_groups : dict
        Grouped node index arrays.
    support_stencils : dict
        Star stencils dictionary.
    M_pinv : dict
        Pseudo-inverse matrices dictionary.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data: Dict[str, object] = {
        "coords": coords.tolist(),
        "triangles": triangles.tolist(),
        "normal_vecs": normal_vecs.tolist(),
        "support_stencils": {str(k): v.tolist() for k, v in support_stencils.items()},
        "M_pinv": {str(k): v.tolist() for k, v in M_pinv.items()},
    }

    for k, v in node_groups.items():
        data[k] = v.tolist()

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Mesh data successfully saved to: {filepath}")


def save_diagnostic_figures(
    g: cfg.Geometry,
    coords: npt.NDArray[np.float64],
    edof: List,
    triangles: npt.NDArray[np.int_],
    normal_vecs: npt.NDArray[np.float64],
    node_groups: Dict[str, npt.NDArray[np.int_]],
    output_dir: str = "examples/paper/figures",
) -> None:
    """Generate and save publication-grade diagnostic plots.

    Parameters
    ----------
    g : calfem.geometry.Geometry
        The CalfEM geometry.
    coords : npt.NDArray[np.float64]
        Nodal coordinates.
    edof : list
        Element DOFs.
    triangles : npt.NDArray[np.int_]
        Triangular faces.
    normal_vecs : npt.NDArray[np.float64]
        Normal vectors.
    node_groups : dict
        Node groupings.
    output_dir : str, optional
        Target directory for images, by default "examples/paper/figures".
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.autolayout"] = True

    # 1. Geometry plot
    plot_geometry(
        g,
        title="Dam Geometry (Chavez 2023)",
        figsize=(8, 3.5),
        savepath=os.path.join(output_dir, "geometry.png"),
    )

    # 2. Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=1,
        el_type=2,
        filled=True,
        figsize=(8, 3.5),
        title=f"Triangular Mesh ({coords.shape[0]} nodes)",
        savepath=os.path.join(output_dir, "mesh.png"),
    )

    # 3. Node classification plot matching FigMallaPresa5C
    nodes_to_plot = (
        node_groups["clay_nodes"],
        node_groups["rock_nodes"],
        node_groups["foundation_nodes"],
        np.hstack((node_groups["left_nodes"], node_groups["right_nodes"])),
        np.hstack((node_groups["bottom_nodes"], node_groups["top_nodes"])),
        np.hstack((
            node_groups["interface_a_nodes"],
            node_groups["interface_b_nodes"],
            node_groups["interface_c_nodes"],
            node_groups["interface_d_nodes"],
            node_groups["interface_e_nodes"],
            node_groups["interface_f_nodes"],
            node_groups["tj1"],
            node_groups["tj2"],
        )),
    )
    labels = (
        "Clay Core (Layer 1)",
        "Rockfill (Layer 2)",
        "Soil Foundation (Layer 3)",
        "Dirichlet BC (NFronD)",
        "Neumann BC (NFronN)",
        "Interfaces (NFronInter)",
    )
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"Mesh Node Distribution ($N = {coords.shape[0]}$)",
        alpha=0.85,
        point_size=12,
        figsize=(10, 3.8),
        savepath=os.path.join(output_dir, "nodes.png"),
    )

    # 4. Normal vectors plot
    all_normal_nodes = (
        node_groups["bottom_nodes"],
        node_groups["top_nodes"],
        node_groups["interface_a_nodes"],
        node_groups["interface_b_nodes"],
        node_groups["interface_c_nodes"],
        node_groups["interface_d_nodes"],
        node_groups["interface_e_nodes"],
        node_groups["interface_f_nodes"],
        node_groups["tj1"],
        node_groups["tj2"],
    )
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=all_normal_nodes,
        quiver_color="k",
        quiver_alpha=0.4,
        title="Boundary & Interface Normal Vectors",
        figsize=(10, 3.8),
        savepath=os.path.join(output_dir, "normal_vectors.png"),
    )
    plt.close("all")
    print(f"Diagnostic figures saved to: {output_dir}")


def main() -> None:
    """Run mesh generation pipeline, precompute stencils, and export artifacts."""
    print("==================================================================")
    print("GFDFlow — Mesh Generator for Five-Layer Dam Seepage (Chavez 2023)")
    print("==================================================================")

    # 1. Define geometry
    g, markers = create_dam_geometry()

    # 2. Generate mesh
    coords, triangles, edof, bdofs, element_markers = generate_mesh(g, el_size_factor=0.655)
    print(f"Generated mesh with {coords.shape[0]} nodes and {len(triangles)} triangles.")

    # 3. Extract node groups
    node_groups = extract_node_groups(coords, triangles, bdofs, element_markers, markers)

    # 4. Compute normal vectors, support stencils, and M_pinv
    print("Precomputing outward normal vectors, support stencils, and M_pinv...")
    normal_vecs, support_stencils, M_pinv = compute_stencils_and_normals(
        coords,
        triangles,
        node_groups,
    )

    # 5. Export JSON
    json_path = "examples/paper/mesh1.json"
    export_mesh_data(
        json_path,
        coords,
        triangles,
        normal_vecs,
        node_groups,
        support_stencils,
        M_pinv,
    )

    # 6. Save figures
    fig_dir = "examples/paper/figures"
    save_diagnostic_figures(
        g,
        coords,
        edof,
        triangles,
        normal_vecs,
        node_groups,
        output_dir=fig_dir,
    )

    print("==================================================================")
    print("Mesh generation completed successfully.")
    print("==================================================================")


if __name__ == "__main__":
    main()
