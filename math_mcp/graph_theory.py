# -*- coding: utf-8 -*-
"""
Graph Theory Computation Module
Provides comprehensive graph theory analysis functionality.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple

try:
    from .file_utils import generate_unique_filename
except ImportError:
    from math_mcp.file_utils import generate_unique_filename


class GraphTheoryCalculator:
    """Graph theory calculator class, provides comprehensive graph analysis functionality"""

    def __init__(self):
        """Initialize the graph theory calculator"""
        pass

    def graph_theory_suite_tool(
        self,
        operation: str,
        graph_data: Optional[Dict[str, Any]] = None,
        adjacency_matrix: Optional[List[List[Union[int, float]]]] = None,
        edge_list: Optional[List[List[Union[int, str]]]] = None,
        node_list: Optional[List[Union[int, str]]] = None,
        source_node: Optional[Union[int, str]] = None,
        target_node: Optional[Union[int, str]] = None,
        weight_attribute: str = "weight",
        directed: bool = False,
        algorithm: str = "auto",
        k_value: Optional[int] = None,
        threshold: Optional[float] = None,
        layout: str = "spring",
        filename: Optional[str] = None,
        node_colors: Optional[List[str]] = None,
        edge_colors: Optional[List[str]] = None,
        node_sizes: Optional[List[int]] = None,
        show_labels: bool = True,
        figsize: Tuple[float, float] = (10, 8),
    ) -> Dict[str, Any]:
        """
        Comprehensive graph theory analysis tool

        Args:
            operation: Type of operation
            graph_data: Graph data as dictionary
            adjacency_matrix: Adjacency matrix
            edge_list: List of edges
            node_list: List of nodes
            source_node: Source node
            target_node: Target node
            weight_attribute: Name of the weight attribute
            directed: Whether the graph is directed
            algorithm: Algorithm selection
            k_value: Parameter K
            threshold: Threshold parameter
            layout: Layout algorithm
            filename: File name for saving
            node_colors: Node colors
            edge_colors: Edge colors
            node_sizes: Node sizes
            show_labels: Whether to show labels
            figsize: Figure size

        Returns:
            Graph theory analysis result
        """
        try:
            # Build graph
            G = self._build_graph(
                graph_data,
                adjacency_matrix,
                edge_list,
                node_list,
                directed,
                weight_attribute,
            )
            if isinstance(G, dict) and "error" in G:
                return G

            if operation == "shortest_path":
                if source_node is None or target_node is None:
                    return {
                        "error": "Shortest path calculation requires source and target nodes"
                    }
                return self._shortest_path_analysis(
                    G, source_node, target_node, algorithm, weight_attribute
                )

            elif operation == "all_pairs_shortest_path":
                return self._all_pairs_shortest_path(G, algorithm, weight_attribute)

            elif operation == "maximum_flow":
                if source_node is None or target_node is None:
                    return {
                        "error": "Maximum flow calculation requires source and target nodes"
                    }
                return self._maximum_flow_analysis(
                    G, source_node, target_node, algorithm, weight_attribute
                )

            elif operation == "connectivity_analysis":
                return self._connectivity_analysis(G, algorithm)

            elif operation == "centrality_analysis":
                return self._centrality_analysis(G, algorithm)

            elif operation == "community_detection":
                return self._community_detection(G, algorithm, threshold)

            elif operation == "spectral_analysis":
                return self._spectral_analysis(G, k_value)

            elif operation == "graph_properties":
                return self._graph_properties_analysis(G)

            elif operation == "minimum_spanning_tree":
                return self._minimum_spanning_tree(G, algorithm, weight_attribute)

            elif operation == "graph_coloring":
                return self._graph_coloring(G, algorithm)

            elif operation == "clique_analysis":
                return self._clique_analysis(G, k_value)

            elif operation == "graph_visualization":
                return self._visualize_graph(
                    G,
                    layout,
                    filename,
                    node_colors,
                    edge_colors,
                    node_sizes,
                    show_labels,
                    figsize,
                )

            elif operation == "graph_comparison":
                if not isinstance(graph_data, list) or len(graph_data) < 2:
                    return {"error": "Graph comparison requires at least two graphs"}
                return self.compare_graphs(graph_data, directed)

            elif operation == "graph_generation":
                n_nodes = len(node_list) if node_list else None
                return self._generate_special_graphs(
                    algorithm, n_nodes, threshold, k_value
                )

            else:
                return {"error": f"Unsupported operation type: {operation}"}

        except Exception as e:
            return {"error": f"Graph theory calculation error: {str(e)}"}

    def _build_graph(
        self,
        graph_data: Optional[Dict[str, Any]],
        adjacency_matrix: Optional[List[List[Union[int, float]]]],
        edge_list: Optional[List[List[Union[int, str]]]],
        node_list: Optional[List[Union[int, str]]],
        directed: bool = False,
        weight_attribute: str = "weight",
    ):
        """Build NetworkX graph object"""
        try:
            if directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()

            # Build from graph data dict
            if graph_data:
                if "nodes" in graph_data:
                    G.add_nodes_from(graph_data["nodes"])
                if "edges" in graph_data:
                    if isinstance(graph_data["edges"][0], dict):
                        # Edges with attributes
                        for edge in graph_data["edges"]:
                            G.add_edge(
                                edge["source"],
                                edge["target"],
                                **{
                                    k: v
                                    for k, v in edge.items()
                                    if k not in ["source", "target"]
                                },
                            )
                    else:
                        # Simple edge list
                        G.add_edges_from(graph_data["edges"])

            # Build from adjacency matrix
            elif adjacency_matrix:
                matrix = np.array(adjacency_matrix)
                if node_list:
                    G = nx.from_numpy_array(matrix, create_using=type(G))
                    mapping = {i: node_list[i] for i in range(len(node_list))}
                    G = nx.relabel_nodes(G, mapping)
                else:
                    G = nx.from_numpy_array(matrix, create_using=type(G))

            # Build from edge list
            elif edge_list:
                if len(edge_list[0]) == 2:
                    # Unweighted edges
                    G.add_edges_from(edge_list)
                elif len(edge_list[0]) == 3:
                    # Weighted/capacity edges, save as user-defined attribute and as "capacity" for max flow
                    for u, v, w in edge_list:
                        G.add_edge(u, v, **{weight_attribute: w, "capacity": w})
                else:
                    return {"error": "Incorrect edge list format"}

            # Build from node list only
            elif node_list:
                G.add_nodes_from(node_list)

            else:
                return {"error": "Graph data is required"}

            return G

        except Exception as e:
            return {"error": f"Failed to build graph: {str(e)}"}

    def _shortest_path_analysis(
        self,
        G,
        source,
        target,
        algorithm: str = "auto",
        weight_attribute: str = "weight",
    ) -> Dict[str, Any]:
        """Shortest path analysis"""
        try:
            results = {"source": source, "target": target, "algorithm": algorithm}

            # Check if nodes exist
            if source not in G.nodes() or target not in G.nodes():
                return {"error": "Source or target node does not exist in the graph"}

            # Select algorithm
            if algorithm == "auto":
                if nx.is_weighted(G, weight=weight_attribute):
                    algorithm = "dijkstra"
                else:
                    algorithm = "bfs"

            # Calculate shortest path
            try:
                if algorithm == "dijkstra":
                    if nx.is_weighted(G, weight=weight_attribute):
                        path = nx.dijkstra_path(
                            G, source, target, weight=weight_attribute
                        )
                        length = nx.dijkstra_path_length(
                            G, source, target, weight=weight_attribute
                        )
                    else:
                        path = nx.shortest_path(G, source, target)
                        length = nx.shortest_path_length(G, source, target)

                elif algorithm == "bellman_ford":
                    path = nx.bellman_ford_path(
                        G, source, target, weight=weight_attribute
                    )
                    length = nx.bellman_ford_path_length(
                        G, source, target, weight=weight_attribute
                    )

                elif algorithm == "bfs":
                    path = nx.shortest_path(G, source, target)
                    length = nx.shortest_path_length(G, source, target)

                elif algorithm == "astar":
                    # A* requires a heuristic function, using Euclidean distance as a simple example
                    def heuristic(u, v):
                        return 0  # simplified version

                    path = nx.astar_path(
                        G, source, target, heuristic=heuristic, weight=weight_attribute
                    )
                    length = nx.astar_path_length(
                        G, source, target, heuristic=heuristic, weight=weight_attribute
                    )

                else:
                    return {"error": f"Unsupported algorithm: {algorithm}"}

                results.update(
                    {
                        "path": path,
                        "path_length": length,
                        "path_edges": [
                            (path[i], path[i + 1]) for i in range(len(path) - 1)
                        ],
                        "hop_count": len(path) - 1,
                    }
                )

            except nx.NetworkXNoPath:
                results["error"] = "No path exists between source and target"
            except Exception as e:
                results["error"] = f"Path calculation failed: {str(e)}"

            # Calculate all shortest paths (if multiple exist)
            try:
                all_paths = list(
                    nx.all_shortest_paths(
                        G,
                        source,
                        target,
                        weight=(
                            weight_attribute
                            if nx.is_weighted(G, weight=weight_attribute)
                            else None
                        ),
                    )
                )
                if len(all_paths) > 1:
                    results["all_shortest_paths"] = all_paths
                    results["path_count"] = len(all_paths)
            except:
                pass

            return results

        except Exception as e:
            return {"error": f"Shortest path analysis error: {str(e)}"}

    def _all_pairs_shortest_path(
        self, G, algorithm: str = "auto", weight_attribute: str = "weight"
    ) -> Dict[str, Any]:
        """All pairs shortest path analysis"""
        try:
            results = {"algorithm": algorithm}

            # Select algorithm
            if algorithm == "auto":
                if nx.is_weighted(G, weight=weight_attribute):
                    algorithm = "dijkstra"
                else:
                    algorithm = "bfs"

            # Calculate all pairs shortest paths
            if algorithm == "floyd_warshall":
                paths = dict(nx.floyd_warshall(G, weight=weight_attribute))
                path_lengths = dict(nx.floyd_warshall(G, weight=weight_attribute))
            elif algorithm == "dijkstra":
                paths = dict(nx.all_pairs_dijkstra_path(G, weight=weight_attribute))
                path_lengths = dict(
                    nx.all_pairs_dijkstra_path_length(G, weight=weight_attribute)
                )
            else:
                paths = dict(nx.all_pairs_shortest_path(G))
                path_lengths = dict(nx.all_pairs_shortest_path_length(G))

            # Statistical information
            all_lengths = []
            for source in path_lengths:
                for target in path_lengths[source]:
                    if source != target:
                        all_lengths.append(path_lengths[source][target])

            results.update(
                {
                    "path_matrix": path_lengths,
                    "statistics": {
                        "average_path_length": (
                            np.mean(all_lengths) if all_lengths else 0
                        ),
                        "max_path_length": max(all_lengths) if all_lengths else 0,
                        "min_path_length": min(all_lengths) if all_lengths else 0,
                        "diameter": max(all_lengths) if all_lengths else 0,
                    },
                    "node_count": G.number_of_nodes(),
                    "edge_count": G.number_of_edges(),
                }
            )

            return results

        except Exception as e:
            return {"error": f"All pairs shortest path analysis error: {str(e)}"}

    def _maximum_flow_analysis(
        self,
        G,
        source,
        target,
        algorithm: str = "auto",
        weight_attribute: str = "capacity",
    ) -> Dict[str, Any]:
        """Maximum flow analysis"""
        try:
            results = {"source": source, "target": target, "algorithm": algorithm}

            # Check if directed
            if not G.is_directed():
                G = G.to_directed()

            # If algorithm is auto, use edmonds_karp by default
            if algorithm == "auto":
                algorithm = "edmonds_karp"

            # Ensure all edges have "capacity" attribute
            for u, v, data in G.edges(data=True):
                if "capacity" not in data:
                    cap_val = data.get(weight_attribute, 1)
                    data["capacity"] = cap_val

            # Calculate max flow according to algorithm
            if algorithm == "edmonds_karp":
                flow_value, flow_dict = nx.maximum_flow(
                    G,
                    source,
                    target,
                    capacity="capacity",
                    flow_func=nx.algorithms.flow.edmonds_karp,
                )
            elif algorithm == "preflow_push":
                flow_value, flow_dict = nx.maximum_flow(
                    G,
                    source,
                    target,
                    capacity="capacity",
                    flow_func=nx.algorithms.flow.preflow_push,
                )
            elif algorithm == "shortest_augmenting_path":
                flow_value, flow_dict = nx.maximum_flow(
                    G,
                    source,
                    target,
                    capacity="capacity",
                    flow_func=nx.algorithms.flow.shortest_augmenting_path,
                )
            else:
                return {"error": f"Unsupported maximum flow algorithm: {algorithm}"}

            # Calculate minimum cut
            cut_value, partition = nx.minimum_cut(G, source, target)

            # Flow network analysis
            flow_edges = []
            for u in flow_dict:
                for v in flow_dict[u]:
                    if flow_dict[u][v] > 0:
                        flow_edges.append(
                            {
                                "source": u,
                                "target": v,
                                "flow": flow_dict[u][v],
                                "capacity": G[u][v].get(
                                    "capacity", G[u][v].get("weight", 1)
                                ),
                            }
                        )

            results.update(
                {
                    "maximum_flow_value": flow_value,
                    "flow_distribution": flow_dict,
                    "flow_edges": flow_edges,
                    "minimum_cut": {
                        "cut_value": cut_value,
                        "partition": [list(partition[0]), list(partition[1])],
                    },
                    "flow_efficiency": (
                        flow_value
                        / sum(
                            G[u][v].get("capacity", G[u][v].get("weight", 1))
                            for u, v in G.edges()
                            if u in partition[0] and v in partition[1]
                        )
                        if partition[0] and partition[1]
                        else 0
                    ),
                }
            )

            return results

        except Exception as e:
            return {"error": f"Maximum flow analysis error: {str(e)}"}

    def _connectivity_analysis(self, G, algorithm: str = "auto") -> Dict[str, Any]:
        """Connectivity analysis"""
        try:
            results = {"algorithm": algorithm}

            # Basic connectivity
            if G.is_directed():
                results["strongly_connected"] = nx.is_strongly_connected(G)
                results["weakly_connected"] = nx.is_weakly_connected(G)

                # Strongly connected components
                scc = list(nx.strengthly_connected_components(G))
                results["strongly_connected_components"] = {
                    "count": len(scc),
                    "components": [list(component) for component in scc],
                    "largest_component_size": (
                        max(len(component) for component in scc) if scc else 0
                    ),
                }

                # Weakly connected components
                wcc = list(nx.weakly_connected_components(G))
                results["weakly_connected_components"] = {
                    "count": len(wcc),
                    "components": [list(component) for component in wcc],
                    "largest_component_size": (
                        max(len(component) for component in wcc) if wcc else 0
                    ),
                }
            else:
                results["connected"] = nx.is_connected(G)

                # Connected components
                cc = list(nx.connected_components(G))
                results["connected_components"] = {
                    "count": len(cc),
                    "components": [list(component) for component in cc],
                    "largest_component_size": (
                        max(len(component) for component in cc) if cc else 0
                    ),
                }

            # Connectivity measures
            results["connectivity_measures"] = {}

            # Node connectivity
            try:
                if nx.is_connected(G) or (
                    G.is_directed() and nx.is_strongly_connected(G)
                ):
                    results["connectivity_measures"]["node_connectivity"] = (
                        nx.node_connectivity(G)
                    )
                    results["connectivity_measures"]["edge_connectivity"] = (
                        nx.edge_connectivity(G)
                    )
            except:
                pass

            # Articulation points and bridges
            if not G.is_directed():
                articulation_points = list(nx.articulation_points(G))
                bridges = list(nx.bridges(G))

                results["structural_analysis"] = {
                    "articulation_points": articulation_points,
                    "articulation_points_count": len(articulation_points),
                    "bridges": [list(bridge) for bridge in bridges],
                    "bridges_count": len(bridges),
                }

            return results

        except Exception as e:
            return {"error": f"Connectivity analysis error: {str(e)}"}

    def _centrality_analysis(self, G, algorithm: str = "all") -> Dict[str, Any]:
        """Centrality analysis"""
        try:
            results = {"algorithm": algorithm}
            centralities = {}

            # Degree centrality
            if algorithm in ["all", "degree"]:
                degree_centrality = nx.degree_centrality(G)
                centralities["degree_centrality"] = degree_centrality
                centralities["degree_centrality_stats"] = {
                    "max_node": max(degree_centrality, key=degree_centrality.get),
                    "max_value": max(degree_centrality.values()),
                    "min_value": min(degree_centrality.values()),
                    "average": np.mean(list(degree_centrality.values())),
                }

            # Closeness centrality
            if algorithm in ["all", "closeness"]:
                try:
                    closeness_centrality = nx.closeness_centrality(G)
                    centralities["closeness_centrality"] = closeness_centrality
                    centralities["closeness_centrality_stats"] = {
                        "max_node": max(
                            closeness_centrality, key=closeness_centrality.get
                        ),
                        "max_value": max(closeness_centrality.values()),
                        "min_value": min(closeness_centrality.values()),
                        "average": np.mean(list(closeness_centrality.values())),
                    }
                except:
                    centralities["closeness_centrality"] = (
                        "Unable to calculate (graph is disconnected)"
                    )

            # Betweenness centrality
            if algorithm in ["all", "betweenness"]:
                betweenness_centrality = nx.betweenness_centrality(G)
                centralities["betweenness_centrality"] = betweenness_centrality
                centralities["betweenness_centrality_stats"] = {
                    "max_node": max(
                        betweenness_centrality, key=betweenness_centrality.get
                    ),
                    "max_value": max(betweenness_centrality.values()),
                    "min_value": min(betweenness_centrality.values()),
                    "average": np.mean(list(betweenness_centrality.values())),
                }

            # Eigenvector centrality
            if algorithm in ["all", "eigenvector"]:
                try:
                    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                    centralities["eigenvector_centrality"] = eigenvector_centrality
                    centralities["eigenvector_centrality_stats"] = {
                        "max_node": max(
                            eigenvector_centrality, key=eigenvector_centrality.get
                        ),
                        "max_value": max(eigenvector_centrality.values()),
                        "min_value": min(eigenvector_centrality.values()),
                        "average": np.mean(list(eigenvector_centrality.values())),
                    }
                except:
                    centralities["eigenvector_centrality"] = (
                        "Unable to calculate (possibly not converging)"
                    )

            # PageRank
            if algorithm in ["all", "pagerank"]:
                pagerank = nx.pagerank(G)
                centralities["pagerank"] = pagerank
                centralities["pagerank_stats"] = {
                    "max_node": max(pagerank, key=pagerank.get),
                    "max_value": max(pagerank.values()),
                    "min_value": min(pagerank.values()),
                    "average": np.mean(list(pagerank.values())),
                }

            # Katz centrality
            if algorithm in ["all", "katz"]:
                try:
                    katz_centrality = nx.katz_centrality(G)
                    centralities["katz_centrality"] = katz_centrality
                    centralities["katz_centrality_stats"] = {
                        "max_node": max(katz_centrality, key=katz_centrality.get),
                        "max_value": max(katz_centrality.values()),
                        "min_value": min(katz_centrality.values()),
                        "average": np.mean(list(katz_centrality.values())),
                    }
                except:
                    centralities["katz_centrality"] = "Unable to calculate"

            results["centralities"] = centralities

            # Centrality rankings
            rankings = {}
            for centrality_type, centrality_dict in centralities.items():
                if isinstance(centrality_dict, dict):
                    sorted_nodes = sorted(
                        centrality_dict.items(), key=lambda x: x[1], reverse=True
                    )
                    rankings[centrality_type] = sorted_nodes[:10]  # Top 10

            results["rankings"] = rankings

            return results

        except Exception as e:
            return {"error": f"Centrality analysis error: {str(e)}"}

    def _spectral_analysis(self, G, k_value: Optional[int] = None) -> Dict[str, Any]:
        """Spectral analysis"""
        try:
            results = {}

            # Adjacency matrix
            adj_matrix = nx.adjacency_matrix(G).todense()
            results["adjacency_matrix"] = adj_matrix.tolist()

            # Degree matrix
            degree_matrix = np.diag([G.degree(node) for node in G.nodes()])
            results["degree_matrix"] = degree_matrix.tolist()

            # Laplacian matrix
            laplacian_matrix = nx.laplacian_matrix(G).todense()
            results["laplacian_matrix"] = laplacian_matrix.tolist()

            # Normalized Laplacian matrix
            normalized_laplacian = nx.normalized_laplacian_matrix(G).todense()
            results["normalized_laplacian_matrix"] = normalized_laplacian.tolist()

            # Eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
            results["eigenvalues"] = eigenvalues.tolist()
            results["eigenvectors"] = eigenvectors.tolist()

            # Spectral properties
            results["spectral_properties"] = {
                "algebraic_connectivity": (
                    float(eigenvalues[1]) if len(eigenvalues) > 1 else 0
                ),
                "spectral_gap": (
                    float(eigenvalues[1] - eigenvalues[0])
                    if len(eigenvalues) > 1
                    else 0
                ),
                "largest_eigenvalue": float(eigenvalues[-1]),
                "smallest_eigenvalue": float(eigenvalues[0]),
                "eigenvalue_multiplicity_zero": int(
                    np.sum(np.abs(eigenvalues) < 1e-10)
                ),
            }

            # Spectral clustering
            if k_value:
                try:
                    from sklearn.cluster import SpectralClustering

                    spectral_clustering = SpectralClustering(
                        n_clusters=k_value, affinity="precomputed"
                    )
                    cluster_labels = spectral_clustering.fit_predict(adj_matrix)

                    # Organize cluster results
                    clusters = {}
                    for i, label in enumerate(cluster_labels):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(list(G.nodes())[i])

                    results["spectral_clustering"] = {
                        "k": k_value,
                        "clusters": clusters,
                        "cluster_sizes": [
                            len(cluster) for cluster in clusters.values()
                        ],
                    }
                except ImportError:
                    results["spectral_clustering"] = (
                        "scikit-learn is required for spectral clustering"
                    )
                except Exception as e:
                    results["spectral_clustering"] = (
                        f"Spectral clustering failed: {str(e)}"
                    )

            return results

        except Exception as e:
            return {"error": f"Spectral analysis error: {str(e)}"}

    def _graph_properties_analysis(self, G) -> Dict[str, Any]:
        """Graph properties analysis"""
        try:
            results = {}

            # Basic properties
            results["basic_properties"] = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "is_directed": G.is_directed(),
                "is_multigraph": G.is_multigraph(),
                "density": nx.density(G),
                "is_connected": (
                    nx.is_connected(G)
                    if not G.is_directed()
                    else nx.is_strongly_connected(G)
                ),
            }

            # Degree distribution
            degrees = [G.degree(node) for node in G.nodes()]
            results["degree_distribution"] = {
                "degrees": degrees,
                "average_degree": np.mean(degrees),
                "max_degree": max(degrees),
                "min_degree": min(degrees),
                "degree_variance": np.var(degrees),
                "degree_histogram": np.histogram(degrees, bins=10)[0].tolist(),
            }

            # Path-related properties
            if nx.is_connected(G) or (G.is_directed() and nx.is_strongly_connected(G)):
                try:
                    results["path_properties"] = {
                        "diameter": nx.diameter(G),
                        "radius": nx.radius(G),
                        "average_shortest_path_length": nx.average_shortest_path_length(
                            G
                        ),
                        "center": list(nx.center(G)),
                        "periphery": list(nx.periphery(G)),
                    }
                except:
                    results["path_properties"] = (
                        "Unable to calculate (graph is disconnected or calculation failed)"
                    )

            # Clustering coefficient
            results["clustering"] = {
                "average_clustering": nx.average_clustering(G),
                "clustering_coefficients": nx.clustering(G),
                "transitivity": nx.transitivity(G),
            }

            # Assortativity
            try:
                results["assortativity"] = {
                    "degree_assortativity": nx.degree_assortativity_coefficient(G)
                }
            except:
                results["assortativity"] = "Unable to calculate assortativity"

            # Other graph characteristics
            results["graph_characteristics"] = {
                "is_tree": nx.is_tree(G),
                "is_forest": nx.is_forest(G),
                "is_bipartite": nx.is_bipartite(G),
                "is_planar": (
                    nx.is_planar(G)
                    if G.number_of_nodes() <= 100
                    else "Graph too large, planar detection skipped"
                ),
                "is_eulerian": nx.is_eulerian(G),
                "is_semiconnected": (
                    nx.is_semiconnected(G)
                    if G.is_directed()
                    else "Applicable only for directed graphs"
                ),
            }

            return results

        except Exception as e:
            return {"error": f"Graph properties analysis error: {str(e)}"}

    def _visualize_graph(
        self,
        G,
        layout: str = "spring",
        filename: Optional[str] = None,
        node_colors: Optional[List[str]] = None,
        edge_colors: Optional[List[str]] = None,
        node_sizes: Optional[List[int]] = None,
        show_labels: bool = True,
        figsize: Tuple[float, float] = (10, 8),
    ) -> Dict[str, Any]:
        """Graph visualization"""
        try:
            plt.figure(figsize=figsize)

            # Select layout
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "random":
                pos = nx.random_layout(G)
            elif layout == "shell":
                pos = nx.shell_layout(G)
            elif layout == "spectral":
                pos = nx.spectral_layout(G)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)

            # Set default colors and size
            if node_colors is None:
                node_colors = ["lightblue"] * G.number_of_nodes()
            if node_sizes is None:
                node_sizes = [300] * G.number_of_nodes()
            if edge_colors is None:
                edge_colors = ["gray"] * G.number_of_edges()

            # Draw graph
            nx.draw_networkx_nodes(
                G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
            )
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, width=1.5)

            if show_labels:
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

            # Show edge weights if weighted
            if nx.is_weighted(G):
                edge_labels = nx.get_edge_attributes(G, "weight")
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

            plt.title(f"Graph Visualization (layout: {layout})", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

            # Save image
            if filename is None:
                filename = "graph_visualization"

            filepath, _ = generate_unique_filename(
                "graph_visualization", "png", filename
            )
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            return {
                "visualization_info": {
                    "layout": layout,
                    "node_count": G.number_of_nodes(),
                    "edge_count": G.number_of_edges(),
                    "figure_size": figsize,
                },
                "file_path": filepath,
                "layout_positions": {
                    str(node): [float(pos[node][0]), float(pos[node][1])]
                    for node in pos
                },
            }

        except Exception as e:
            return {"error": f"Graph visualization error: {str(e)}"}

    def _minimum_spanning_tree(
        self, G, algorithm: str = "kruskal", weight_attribute: str = "weight"
    ) -> Dict[str, Any]:
        """Minimum spanning tree"""
        try:
            if G.is_directed():
                return {
                    "error": "Minimum spanning tree is only applicable to undirected graphs"
                }

            if not nx.is_connected(G):
                return {
                    "error": "Graph is not connected, cannot construct spanning tree"
                }

            # Calculate minimum spanning tree
            if algorithm == "kruskal":
                mst = nx.minimum_spanning_tree(
                    G, weight=weight_attribute, algorithm="kruskal"
                )
            elif algorithm == "prim":
                mst = nx.minimum_spanning_tree(
                    G, weight=weight_attribute, algorithm="prim"
                )
            else:
                mst = nx.minimum_spanning_tree(G, weight=weight_attribute)

            # Calculate total weight
            total_weight = sum(
                mst[u][v].get(weight_attribute, 1) for u, v in mst.edges()
            )

            return {
                "algorithm": algorithm,
                "mst_edges": list(mst.edges(data=True)),
                "mst_nodes": list(mst.nodes()),
                "total_weight": total_weight,
                "edge_count": mst.number_of_edges(),
                "node_count": mst.number_of_nodes(),
                "is_tree": nx.is_tree(mst),
            }

        except Exception as e:
            return {"error": f"Minimum spanning tree calculation error: {str(e)}"}

    def _graph_coloring(self, G, algorithm: str = "greedy") -> Dict[str, Any]:
        """Graph coloring"""
        try:
            if algorithm == "greedy":
                coloring = nx.greedy_color(G, strategy="largest_first")
            elif algorithm == "welsh_powell":
                coloring = nx.greedy_color(G, strategy="largest_first")
            elif algorithm == "dsatur":
                coloring = nx.greedy_color(G, strategy="saturation_largest_first")
            else:
                coloring = nx.greedy_color(G)

            # Analyze coloring result
            num_colors = max(coloring.values()) + 1
            color_classes = {}
            for node, color in coloring.items():
                if color not in color_classes:
                    color_classes[color] = []
                color_classes[color].append(node)

            return {
                "algorithm": algorithm,
                "coloring": coloring,
                "chromatic_number": num_colors,
                "color_classes": color_classes,
                "is_proper_coloring": self._verify_proper_coloring(G, coloring),
            }

        except Exception as e:
            return {"error": f"Graph coloring error: {str(e)}"}

    def _verify_proper_coloring(self, G, coloring) -> bool:
        """Verify whether coloring is proper"""
        for u, v in G.edges():
            if coloring[u] == coloring[v]:
                return False
        return True

    def _clique_analysis(self, G, k_value: Optional[int] = None) -> Dict[str, Any]:
        """Clique analysis"""
        try:
            results = {}

            # Find all maximal cliques
            max_cliques = list(nx.find_cliques(G))
            results["max_cliques"] = [list(clique) for clique in max_cliques]
            results["max_clique_count"] = len(max_cliques)

            # Maximum clique size
            if max_cliques:
                clique_sizes = [len(clique) for clique in max_cliques]
                results["largest_clique_size"] = max(clique_sizes)
                results["clique_size_distribution"] = {
                    "sizes": clique_sizes,
                    "average_size": np.mean(clique_sizes),
                    "max_size": max(clique_sizes),
                    "min_size": min(clique_sizes),
                }

            # Clique number (size of the largest clique)
            results["clique_number"] = nx.graph_clique_number(G)

            # If k is specified, find k-cliques
            if k_value:
                k_cliques = [clique for clique in max_cliques if len(clique) >= k_value]
                results[f"k_cliques_{k_value}"] = {
                    "cliques": [list(clique) for clique in k_cliques],
                    "count": len(k_cliques),
                }

            return results

        except Exception as e:
            return {"error": f"Clique analysis error: {str(e)}"}

    def _community_detection(
        self, G, algorithm: str = "auto", threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Community detection"""
        try:
            results = {"algorithm": algorithm}

            if algorithm in ["auto", "louvain"]:
                try:
                    import community as community_louvain

                    if hasattr(community_louvain, "best_partition"):
                        partition = community_louvain.best_partition(G)
                    else:
                        raise AttributeError

                    modularity = community_louvain.modularity(partition, G)

                    # Organize communities
                    communities = {}
                    for node, comm_id in partition.items():
                        if comm_id not in communities:
                            communities[comm_id] = []
                        communities[comm_id].append(node)

                    results.update(
                        {
                            "communities": communities,
                            "community_count": len(communities),
                            "modularity": modularity,
                            "partition": partition,
                        }
                    )

                except ImportError:
                    results["error"] = (
                        "python-louvain package is required for Louvain community detection"
                    )
                except AttributeError:
                    comms = list(nx.community.greedy_modularity_communities(G))
                    communities = {i: list(c) for i, c in enumerate(comms)}
                    results.update(
                        {
                            "communities": communities,
                            "community_count": len(communities),
                        }
                    )

            elif algorithm == "girvan_newman":
                communities_generator = nx.community.girvan_newman(G)
                top_level_communities = next(communities_generator)
                communities = {
                    i: list(community)
                    for i, community in enumerate(top_level_communities)
                }

                results.update(
                    {"communities": communities, "community_count": len(communities)}
                )

            elif algorithm == "label_propagation":
                communities_generator = nx.community.label_propagation_communities(G)
                communities = {
                    i: list(community)
                    for i, community in enumerate(communities_generator)
                }

                results.update(
                    {"communities": communities, "community_count": len(communities)}
                )

            else:
                return {
                    "error": f"Unsupported community detection algorithm: {algorithm}"
                }

            # Community statistics
            if "communities" in results:
                community_sizes = [
                    len(comm) for comm in results["communities"].values()
                ]
                results["community_statistics"] = {
                    "sizes": community_sizes,
                    "average_size": np.mean(community_sizes),
                    "largest_community_size": max(community_sizes),
                    "smallest_community_size": min(community_sizes),
                }

            return results

        except Exception as e:
            return {"error": f"Community detection error: {str(e)}"}

    def _generate_special_graphs(
        self,
        graph_type: str,
        n: Optional[int] = None,
        p: Optional[float] = None,
        k_value: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate special graphs"""
        try:
            if n is None:
                n = 10

            if graph_type == "complete":
                G = nx.complete_graph(n)
                description = f"Complete graph K_{n}"
            elif graph_type == "cycle":
                G = nx.cycle_graph(n)
                description = f"Cycle graph C_{n}"
            elif graph_type == "path":
                G = nx.path_graph(n)
                description = f"Path graph P_{n}"
            elif graph_type == "star":
                G = nx.star_graph(n - 1)
                description = f"Star graph S_{n-1}"
            elif graph_type == "wheel":
                G = nx.wheel_graph(n)
                description = f"Wheel graph W_{n}"
            elif graph_type == "random":
                if p is None:
                    p = 0.5
                G = nx.erdos_renyi_graph(n, p)
                description = f"Random graph G({n}, {p})"
            elif graph_type == "barabasi_albert":
                if n is None:
                    n = 10
                m = k_value if k_value is not None else (min(3, n - 1) if n > 3 else 1)
                G = nx.barabasi_albert_graph(n, m)
                description = f"BA scale-free network ({n} nodes)"
            elif graph_type == "watts_strogatz":
                if n is None:
                    n = 10
                k = k_value if k_value is not None else (min(4, n - 1) if n > 4 else 2)
                p_ws = p if p else 0.3
                G = nx.watts_strogatz_graph(n, k, p_ws)
                description = f"WS small world network ({n} nodes)"
            else:
                return {"error": f"Unsupported graph type: {graph_type}"}

            # Analyze generated graph
            properties = self._graph_properties_analysis(G)

            return {
                "graph_type": graph_type,
                "description": description,
                "parameters": {"n": n, "p": p},
                "graph_data": {"nodes": list(G.nodes()), "edges": list(G.edges())},
                "properties": properties,
            }

        except Exception as e:
            return {"error": f"Special graph generation error: {str(e)}"}
