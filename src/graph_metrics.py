import glob
import math
from collections import defaultdict
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sknw
import tifffile as tif
import tqdm
from joblib import Parallel, delayed
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.spatial import distance_matrix
from skimage import filters
from skimage.morphology import skeletonize, convex_hull_image
import alphashape
from shapely.geometry import Point

def construct_graph(sknw_graph, nodes):

    network_graph = nx.Graph()

    # Adding nodes to the graph with positions
    for i in sknw_graph.nodes():
        network_graph.add_node(i, position=nodes[i]['o'])

    # Adding edges to the graph
    for (start_node, end_node) in sknw_graph.edges():
        network_graph.add_edge(
            start_node, 
            end_node, 
            position=sknw_graph[start_node][end_node]['pts']
            )

    return network_graph

def construct_vessel_network(binary_image):
    
    distance_transform = edt(binary_image)

    # Generate the skeleton of the binary image
    skeleton_image = skeletonize(binary_image)

    # Compute vessel diameters
    diameter_image = 2 * distance_transform * skeleton_image 

    # Build the sknw graph from the skeleton
    sknw_graph = sknw.build_sknw(skeleton_image)

    # Get the nodes of the graph
    graph_nodes = sknw_graph.nodes()

    # Construct a NetworkX graph from the sknw graph and nodes
    G = construct_graph(sknw_graph, graph_nodes)

    return G, diameter_image, sknw_graph

def calculate_branching_angles(sknw_graph, G, subgraph):

    def get_angle_one_direction(edge, edge_index):

        start_node, end_node = edge

        # Get the neighboring edges of the start node
        start_node_neighbors = G[start_node]

        all_branching_angles = {key: [] for key in branching_angles}

        # Find the edge that is not the current edge and that is connected to the start node
        for neighbor_edge_id in start_node_neighbors:
            if (start_node, neighbor_edge_id) != edge:
                neighbor_edge_end_node = neighbor_edge_id

                # Calculate the angle between the current edge and the neighboring edge
                current_edge_vector = sknw_graph.nodes[end_node]['o'] - sknw_graph.nodes[start_node]['o']
                neighbor_edge_vector = sknw_graph.nodes[neighbor_edge_end_node]['o'] - sknw_graph.nodes[start_node]['o']
                cosine_similarity = current_edge_vector.dot(neighbor_edge_vector) / (np.linalg.norm(current_edge_vector) * np.linalg.norm(neighbor_edge_vector))
                cosine_similarity = np.clip(cosine_similarity, -1, 1) # Clip the cosine similarity to avoid numerical errors
                angle = math.degrees(math.acos(cosine_similarity))

                # Store the angle in the dictionary
                all_branching_angles[edge_index].append(angle)

        # Only consider the minimum angle between the current edge and the neighboring edge
        [branching_angles[edge_index].append(np.min(value)) for value in all_branching_angles.values() if len(value) > 0]
        
    branching_angles = {}

    for edge_index, edge_id in enumerate(subgraph.edges()):

        branching_angles[edge_index] = []

        [get_angle_one_direction(edge=edge, edge_index=edge_index) for edge in [edge_id, tuple(np.flip(edge_id))]] 

    return branching_angles

def shortest_path(coordinates):

    return np.linalg.norm(coordinates[0] - coordinates[-1])

def edge_length(coordinates):

    differences = np.diff(coordinates, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return np.sum(segment_lengths)

def compute_vessel_metrics(sknw_graph, diameter_image, subgraph, G, vessel_metrics_df):
    
    vessel_metrics_df['x'] = [data[:, 0] for data in nx.get_edge_attributes(subgraph, 'position').values()]
    vessel_metrics_df['y'] = [data[:, 1] for data in nx.get_edge_attributes(subgraph, 'position').values()]

    vessel_metrics_df['diameter'] = np.array([
        diameter_image[sknw_graph[u][v]['pts'][:, 0], 
        sknw_graph[u][v]['pts'][:, 1]].mean() 
        for u, v in subgraph.edges()
        ])
        
    vessel_metrics_df['area'] = np.array([
        diameter_image[sknw_graph[u][v]['pts'][:, 0], 
        sknw_graph[u][v]['pts'][:, 1]].sum() 
        for u, v in subgraph.edges()
        ])

    vessel_metrics_df['orientation'] = np.array([
        np.arctan2(sknw_graph.nodes[v]['o'][1] - sknw_graph.nodes[u]['o'][1], 
        sknw_graph.nodes[v]['o'][0] - sknw_graph.nodes[u]['o'][0]) 
        for u, v in subgraph.edges()
        ])
        
    edge_aspect_ratio = np.abs(np.nan_to_num([
        np.max(sknw_graph.nodes[v]['o'] - sknw_graph.nodes[u]['o']) / 
        np.min(sknw_graph.nodes[v]['o'] - sknw_graph.nodes[u]['o']) 
        for u, v in subgraph.edges()]))

    edge_aspect_ratio[edge_aspect_ratio > 10] = 1 # self loops
    vessel_metrics_df['aspect_ratio'] = edge_aspect_ratio

    vessel_metrics_df['length'] = np.array([
        edge_length(sknw_graph[u][v]['pts']) 
        for u, v in subgraph.edges()
        ])

    vessel_metrics_df['shortest_path'] = np.array([
        shortest_path(sknw_graph[u][v]['pts']) 
        for u, v in subgraph.edges()
        ]) # - 2 as it includes the nodes coordinates

    edge_tortuosity = vessel_metrics_df['length'] / vessel_metrics_df['shortest_path']
    edge_tortuosity[edge_tortuosity > 10] = 1 # self loops
    vessel_metrics_df['tortuosity'] = edge_tortuosity

    vessel_metrics_df['branching_angle'] = np.nan_to_num([
        np.mean(angles) 
        for angles in calculate_branching_angles(sknw_graph, G, subgraph).values() 
        if len(angles) > 0
        ])
 
    return vessel_metrics_df

def compute_branching_metrics(subgraph, trimmed_subgraph, branching_metrics_df, distance_threshold=100):

    # Extracting the positions of nodes
    positions = np.array([data for data in nx.get_node_attributes(subgraph, 'position').values()])
    branching_metrics_df['x'] = positions[:, 0]
    branching_metrics_df['y'] = positions[:, 1]

    dist_matrix = distance_matrix(positions, positions)

    dist_to_nearest_neighbor = np.min(dist_matrix + np.eye(dist_matrix.shape[0]) * np.max(dist_matrix), axis=1)

    num_neighbors_within_distance = np.sum(dist_matrix <= distance_threshold, axis=1) - 1  # Subtract 1 to exclude the node itself

    branching_metrics_df['branching_distance_to_nearest_neighbor'] = dist_to_nearest_neighbor
    branching_metrics_df[f'number_of_branching_neighbors'] = num_neighbors_within_distance
    branching_metrics_df[f'number_of_vessel_per_branching'] = np.mean([subgraph.degree[node] for node in trimmed_subgraph.nodes() if subgraph.degree[node] > 1 and trimmed_subgraph.degree[node] > 0])

    return branching_metrics_df

def fractal_dimension_and_lacunarity(array, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):

    if max_box_size == None:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num = n_samples, base =2 ))
    scales = np.unique(scales)

    locs = np.array(np.where(array > 0))

    Ns = []
    lacunarity_values = []
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        for offset in offsets:
            bin_edges = [np.arange(0-offset, i+offset, scale) for i in array.shape]
            H1, _ = np.histogramdd(locs.T, bins=bin_edges)
            touched.append(np.sum(H1>0))
        # Calculate lacunarity
        non_zero_hist = H1[H1 > 0]
        lacunarity_values.append(np.var(non_zero_hist) / (np.mean(non_zero_hist) ** 2))
        Ns.append(min(touched))

    Ns = np.array(Ns)

    # Only keep scales at which Ns changed
    unique_Ns, indices = np.unique(Ns, return_index=True)
    unique_Ns = unique_Ns[unique_Ns > 0]
    unique_scales = scales[indices[:len(unique_Ns)]]
    unique_lacunarity = np.array(lacunarity_values)[indices[:len(unique_Ns)]]

    coeffs = np.polyfit(np.log(1 / unique_scales), np.log(unique_Ns), 1)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.log(1 / unique_scales), np.log(unique_Ns), c="teal", label="Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1 / unique_scales))
        ax.plot(np.log(1 / unique_scales), fitted_y_vals, "k--",
                 label=f"Fit: {np.round(coeffs[0], 3)}X+{coeffs[1]}")
        ax.legend()

    return coeffs[0], np.mean(unique_lacunarity)

def find_largest_connected_component(G):

    # Initialize variables to keep track of the largest component
    max_nodes = 0
    largest_component = None
    
    # Iterate over all connected components
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        num_nodes = nx.number_of_nodes(subgraph)
        
        # Update if this component is larger than what we've seen before
        if num_nodes > max_nodes:
            max_nodes = num_nodes
            largest_component = component
            
    # Create a subgraph from the largest component found
    largest_component_subgraph = G.subgraph(largest_component)
    
    return largest_component_subgraph

def collect_border_vicinity_edges(subgraph, vicinity=300):
    """
    Identifies edges that are within a given distance to the concave hull of the graph.
    
    Args:
    subgraph (nx.Graph): The graph representing the network.
    vicinity (int): The vicinity distance to the concave hull.
    
    Returns:
    set: A set of tuples representing the edges that are within the specified vicinity of the concave hull.
    """
    
    # Extract node positions
    coord_tuples = [(data['position'][0], data['position'][1]) for node, data in subgraph.nodes(data=True)]
    hull = alphashape.alphashape(coord_tuples, alpha=0.003)

    buffer_distance = -vicinity # Negative buffer to shrink the polygon
    hull_buffer = hull.buffer(buffer_distance)
    
    border_vicinity_edges = set()
    for u, v in subgraph.edges():
        pts = subgraph[u][v]['position']
        if any(not hull_buffer.contains(Point(pt)) for pt in pts):
            border_vicinity_edges.add((u, v))

    trimmed_subgraph = subgraph.copy()
    trimmed_subgraph.remove_edges_from(border_vicinity_edges)

    return border_vicinity_edges, trimmed_subgraph

def plot_vessel_graph(subgraph, binary_image):
    
    node_color = [1, 1, 1]
    border_color = [0.2, 0.2, 0.2]
    vessel_color = [1, 0, 0]
    sprout_color = [0, 1, 0]

    border_touching_edges, trimmed_subgraph = collect_border_vicinity_edges(subgraph)

    plt.style.use('dark_background')
    plt.figure(figsize=(15, 30))
    plt.imshow(binary_image, cmap='gray', alpha=0.35)

    # Plot nodes
    for node in trimmed_subgraph.nodes():
        if subgraph.degree[node] > 1 and trimmed_subgraph.degree[node] > 0:
            y, x = subgraph.nodes[node]['position']
            plt.plot(x, y, lw=3, marker='o', color=node_color, markersize=5, zorder=10)

    # Plot edges
    for u, v in subgraph.edges():
        if (u, v) in border_touching_edges or (v, u) in border_touching_edges:
            edge_color = border_color
        elif subgraph.degree(u) == 1 or subgraph.degree(v) == 1:
            edge_color = sprout_color
        else:
            edge_color = vessel_color

        pts = subgraph[u][v]['position']
        plt.plot(pts[:, 1], pts[:, 0], lw=3, color=edge_color)

    plt.axis('off')
    plt.tight_layout()

def calculate_graph_metrics(subgraph, sample_name, sknw_graph, G, diameter_image, binary_image):

    convex_hull = convex_hull_image(binary_image)

    border_touching_edges, trimmed_subgraph = collect_border_vicinity_edges(subgraph)

    global_metrics_df = pd.DataFrame({'sample' : [sample_name]})
    branching_metrics_df = pd.DataFrame()
    vessel_metrics_df = pd.DataFrame()

    fd, lacunarity = fractal_dimension_and_lacunarity(diameter_image>0)

    global_metrics_df.loc[0, 'explant_area'] = convex_hull.sum()
    global_metrics_df.loc[0, 'image_area'] = binary_image.shape[0]*binary_image.shape[1]
    global_metrics_df.loc[0, 'vessel_area'] = np.sum(binary_image)
    global_metrics_df.loc[0, 'vessel_density_explant'] = global_metrics_df.loc[0, 'vessel_area'] / global_metrics_df.loc[0, 'explant_area']
    global_metrics_df.loc[0, 'vessel_density_image'] = global_metrics_df.loc[0, 'vessel_area'] / global_metrics_df.loc[0, 'image_area']
    global_metrics_df.loc[0, 'total_vessel_length'] = np.sum([edge_length(sknw_graph[u][v]['pts']) for u, v in subgraph.edges() if (u, v) not in border_touching_edges])
    global_metrics_df.loc[0, 'total_number_of_sprouts'] = len([edge for edge in subgraph.edges() if ((subgraph.degree[edge[0]] == 1 or subgraph.degree[edge[1]] == 1) and edge not in border_touching_edges)])
    global_metrics_df.loc[0, 'total_number_of_vessels'] = len(trimmed_subgraph.edges()) - global_metrics_df.loc[0, 'total_number_of_sprouts'] 
    global_metrics_df.loc[0, 'total_number_of_branchings'] = len([node for node in trimmed_subgraph.nodes() if subgraph.degree[node] > 1 and trimmed_subgraph.degree[node] > 0])
    global_metrics_df.loc[0, 'total_number_of_vessels_touching_image_borders'] = len(border_touching_edges)
    global_metrics_df.loc[0, 'fractal_dimension'] = fd
    global_metrics_df.loc[0, 'lacunarity'] = lacunarity

    vessel_metrics_df = compute_vessel_metrics(sknw_graph, diameter_image, trimmed_subgraph, G, vessel_metrics_df)
    for key, value in vessel_metrics_df.items():
        if key not in ['x', 'y']:
            global_metrics_df.loc[0, 'mean_vessel_' + key] = np.nanmean(value)

    branching_metrics_df = compute_branching_metrics(trimmed_subgraph, trimmed_subgraph, branching_metrics_df, distance_threshold=100)
    for key, value in branching_metrics_df.items():
        if key not in ['x', 'y']:
            global_metrics_df.loc[0, 'mean_' + key] = np.nanmean(value)

    branching_metrics_df['sample'] = sample_name
    vessel_metrics_df['sample'] = sample_name
    
    return global_metrics_df, branching_metrics_df, vessel_metrics_df

def process_directory(masks_dir_path, suffix, save_local_metrics, visualization):

    masks_dir = Path(masks_dir_path)
    masks_paths = list(masks_dir.glob(f'*{suffix}.tif'))
    global_csv_path = masks_dir / 'global_metrics.csv'

    cumulative_global_metrics_df = pd.DataFrame()

    for mask_path in tqdm.tqdm(masks_paths, desc='Processing masks'):
        sample_name = mask_path.stem
        binary_image = tif.imread(str(mask_path))
        
        if np.sum(binary_image) > 0:
        
            G, diameter_image, sknw_graph = construct_vessel_network(binary_image)

            subgraph = find_largest_connected_component(G)

            global_metrics_df, branching_metrics_df, vessel_metrics_df = calculate_graph_metrics(
                subgraph, sample_name, sknw_graph, G, diameter_image, binary_image)
            
            if save_local_metrics:
                branching_csv_path = masks_dir / f'{sample_name}_branching_metrics.h5'
                vessel_csv_path = masks_dir / f'{sample_name}_vessel_metrics.h5'

                branching_metrics_df.to_hdf(branching_csv_path, index=False, key='df', mode='w')
                vessel_metrics_df.to_hdf(vessel_csv_path, index=False, key='df', mode='w')
            
            if visualization:
                plot_vessel_graph(subgraph, binary_image)
                plt.savefig(masks_dir / f'{sample_name}_graph.png', dpi=300)
                plt.close('all')
        else: 

            global_metrics_df = pd.DataFrame({'sample' : [sample_name]})
        
        cumulative_global_metrics_df = pd.concat([cumulative_global_metrics_df, global_metrics_df], ignore_index=True)
        cumulative_global_metrics_df.to_csv(global_csv_path, index=False) # Save every iteration as backup

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute graph metrics on the vessel segmentation masks.")

    parser.add_argument("--masks_dir_path", type=str, required=True,
                        help="Path to the directory containing masks.")
    
    parser.add_argument("--file_suffix", type=str, default='_mask',
                        help="Suffix of the mask files.")
    
    parser.add_argument("--save_local_metrics", action='store_true',
                        help="Save edge and node metrics for each mask, only save the global metrics otherwise.")

    parser.add_argument("--graph_visualization", action='store_true',
                        help="Save graph visualization.")
    
    # parser.add_argument("--exclude_border_vessels", default=True,
    #                     help="Exclude vessels touching the border of the image from the analysis.")

    args = parser.parse_args()

    process_directory(
        masks_dir_path=args.masks_dir_path,
        suffix=args.file_suffix,
        save_local_metrics=args.save_local_metrics, 
        graph_visualization=args.graph_visualization
        )