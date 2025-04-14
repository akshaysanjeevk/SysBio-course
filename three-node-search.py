import networkx as nx
import numpy as np
import random
from collections import defaultdict, Counter
import itertools
import time
import psutil
import os

def get_memory_usage():
    """Return the memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def load_network(edge_file_path):
    """
    Load network from an edge list file.
    Assumes format of each line: source target [weight]
    """
    print(f"Loading network from {edge_file_path}...")
    G = nx.DiGraph()
    
    with open(edge_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                source, target = parts[0], parts[1]
                G.add_edge(source, target)
    
    print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def generate_random_networks(G, num_random=1000, batch_size=10):
    """
    Generate random networks using edge-switching method in batches to save memory.
    Returns a generator that yields batches of random graphs.
    """
    print(f"Generating {num_random} random networks in batches of {batch_size}...")
    
    for batch_start in range(0, num_random, batch_size):
        batch_size_actual = min(batch_size, num_random - batch_start)
        random_graphs = []
        
        for _ in range(batch_size_actual):
            # Create a copy of the graph to preserve the original
            R = nx.DiGraph()
            R.add_nodes_from(G.nodes())
            
            # Create a list of edges to shuffle
            edges = list(G.edges())
            random.shuffle(edges)
            
            # Add edges to maintain the in/out degree of each node
            attempts = 0
            max_attempts = len(edges) * 10
            remaining_edges = edges.copy()
            
            while remaining_edges and attempts < max_attempts:
                attempts += 1
                i = random.randint(0, len(remaining_edges) - 1)
                edge = remaining_edges[i]
                
                if not R.has_edge(edge[0], edge[1]):
                    R.add_edge(edge[0], edge[1])
                    remaining_edges.pop(i)
                    
            # If we couldn't add all edges, just add the remaining ones
            for edge in remaining_edges:
                R.add_edge(edge[0], edge[1])
                
            random_graphs.append(R)
        
        yield random_graphs

def get_triad_type(edges):
    """
    Determine the triad type based on the edges between three nodes.
    Returns an ID number for the triad pattern (0-15 for all possible configurations).
    """
    # Each triad can be represented as a 6-bit binary number
    # where each bit represents the presence (1) or absence (0) of an edge
    # The order is: AB, BA, AC, CA, BC, CB
    binary = 0
    edge_positions = {'01': 0, '10': 1, '02': 2, '20': 3, '12': 4, '21': 5}
    
    for edge in edges:
        src, dst = edge
        key = f"{src}{dst}"
        if key in edge_positions:
            binary |= (1 << (5 - edge_positions[key]))  # Set the corresponding bit
    
    return binary

def count_triads_stream(G, max_samples=None):
    """
    Count triads using a streaming/sampling approach for memory efficiency.
    max_samples: limit the number of node triplets to sample (None = exhaustive)
    """
    print("Counting triads using streaming approach...")
    triad_counts = defaultdict(int)
    
    # Convert to more efficient representation
    node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}
    
    # Create an adjacency matrix representation for quick edge checking
    n = len(node_to_id)
    adj_matrix = np.zeros((n, n), dtype=bool)
    
    for u, v in G.edges():
        adj_matrix[node_to_id[u], node_to_id[v]] = True
    
    nodes = list(range(n))
    
    # Decide whether to sample or do exhaustive counting
    if max_samples is None or max_samples >= n*(n-1)*(n-2)//6:
        # Exhaustive enumeration
        triplets = itertools.combinations(nodes, 3)
        total = n*(n-1)*(n-2)//6
    else:
        # Random sampling
        total = max_samples
        triplets = (random.sample(nodes, 3) for _ in range(max_samples))
    
    # Process triplets
    processed = 0
    start_time = time.time()
    for triplet in triplets:
        processed += 1
        if processed % 100000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed}/{total} triplets ({processed/total*100:.1f}%), "
                  f"Time: {elapsed:.1f}s, Memory: {get_memory_usage():.1f} MB")
        
        # Standardize node order
        a, b, c = sorted(triplet)
        
        # Check for edges between the three nodes
        edges = []
        for i, j in [(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]:
            if adj_matrix[i, j]:
                edges.append((0 if i == a else (1 if i == b else 2),
                              0 if j == a else (1 if j == b else 2)))
        
        # Get the triad type and increment the counter
        triad_type = get_triad_type(edges)
        triad_counts[triad_type] += 1
    
    print(f"Triad counting completed. Found {sum(triad_counts.values())} triads.")
    return triad_counts

def find_network_motifs(G, num_random=1000, z_score_threshold=2.0, batch_size=10):
    """
    Find network motifs by comparing with random networks.
    Uses batching to reduce memory usage.
    """
    print("Finding network motifs...")
    
    # Count triads in the original network
    real_counts = count_triads_stream(G)
    
    # Initialize counters for random networks
    random_counts = defaultdict(list)
    
    # Process random networks in batches to save memory
    for batch_idx, random_graphs_batch in enumerate(generate_random_networks(G, num_random, batch_size)):
        print(f"Processing random network batch {batch_idx+1}/{(num_random+batch_size-1)//batch_size}")
        
        for i, R in enumerate(random_graphs_batch):
            r_counts = count_triads_stream(R)
            
            # Add counts to our running tallies
            for triad_type, count in r_counts.items():
                random_counts[triad_type].append(count)
            
            # Clear memory
            del R
    
    # Calculate statistics and identify motifs
    motifs = {}
    for triad_type, real_count in real_counts.items():
        # Get statistics from random networks
        rand_counts = random_counts.get(triad_type, [0] * num_random)
        rand_mean = np.mean(rand_counts)
        rand_std = np.std(rand_counts) if len(rand_counts) > 1 else 1.0
        
        # Calculate z-score
        if rand_std > 0:
            z_score = (real_count - rand_mean) / rand_std
        else:
            z_score = 0.0 if real_count == rand_mean else float('inf')
        
        # Record the results
        motifs[triad_type] = {
            'real_count': real_count,
            'random_mean': rand_mean,
            'random_std': rand_std,
            'z_score': z_score,
            'is_motif': abs(z_score) >= z_score_threshold
        }
    
    return motifs

def main():
    # Parameters
    edge_file_path = "network_edges.txt"  # Replace with your edge list file path
    num_random_networks = 100  # Increase for better statistical significance
    z_score_threshold = 2.0
    batch_size = 10  # Process this many random networks at once
    
    # Display initial memory usage
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Load the network
    G = load_network(edge_file_path)
    print(f"Memory after loading: {get_memory_usage():.1f} MB")
    
    # Find network motifs
    motifs = find_network_motifs(G, num_random_networks, z_score_threshold, batch_size)
    
    # Display results
    print("\nNetwork Motif Analysis Results:")
    print("------------------------------")
    print("Triad ID | Real Count | Random Mean ± Std | Z-score | Is Motif?")
    for triad_type, stats in sorted(motifs.items()):
        print(f"{triad_type:7d} | {stats['real_count']:10d} | {stats['random_mean']:.1f} ± {stats['random_std']:.1f} | {stats['z_score']:7.2f} | {'Yes' if stats['is_motif'] else 'No'}")
    
    # Save results to file
    with open("motif_results.txt", "w") as f:
        f.write("Triad ID,Real Count,Random Mean,Random Std,Z-score,Is Motif\n")
        for triad_type, stats in sorted(motifs.items()):
            f.write(f"{triad_type},{stats['real_count']},{stats['random_mean']},{stats['random_std']},{stats['z_score']},{stats['is_motif']}\n")
    
    print("\nResults saved to motif_results.txt")
    print(f"Final memory usage: {get_memory_usage():.1f} MB")

if __name__ == "__main__":
    main()