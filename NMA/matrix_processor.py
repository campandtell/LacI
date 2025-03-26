#!/usr/bin/env python3
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Dict
from scipy import linalg

def read_covariance_matrix(file_path: str) -> np.ndarray:
    """Read a covariance matrix from a .dat file."""
    return np.loadtxt(file_path)

def extract_standard_chains_vectorized(data: np.ndarray, source_ranges: List[Tuple[int, int]], 
                                    target_ranges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Extract standard chain length data from longer chains.
    Vectorized implementation for large matrices.
    
    This version handles non-square matrices properly by using the effective size,
    not relying on matrix shape assumptions.
    """
    if len(source_ranges) != len(target_ranges):
        raise ValueError("Source and target ranges must have same length")
    
    # Calculate total residues and dimensions
    n_target_residues = sum(end - start + 1 for start, end in target_ranges)
    n_source_residues = sum(end - start + 1 for start, end in source_ranges)
    
    # For covariance matrix data
    if data.ndim == 2:
        # Use minimum dimension if the matrix is not perfectly square
        min_dim = min(data.shape)
        
        # Calculate expected size based on source ranges (3 coords per residue)
        expected_size = 3 * n_source_residues
        
        # Handle cases where shape doesn't perfectly match (might have padding)
        if min_dim < expected_size:
            raise ValueError(f"Matrix size {min_dim} is smaller than expected size {expected_size} for source ranges")
        
        # For full matrix, create a square result matrix of the target size
        n_dim = 3 * n_target_residues
        result = np.zeros((n_dim, n_dim))
        
        # Process each chain separately
        target_offset = 0
        source_offset = 0
        
        for i, ((s_start, s_end), (t_start, t_end)) in enumerate(zip(source_ranges, target_ranges)):
            # Calculate sizes for this chain
            n_source_residues_chain = s_end - s_start + 1
            n_target_residues_chain = t_end - t_start + 1
            
            source_size = 3 * n_source_residues_chain
            target_size = 3 * n_target_residues_chain
            
            # Extract the block for this chain (ensuring we stay in bounds)
            end_source = min(source_offset + source_size, min_dim)
            
            # Take only the square part of this block
            block_size = min(end_source - source_offset, source_size)
            source_block = data[source_offset:source_offset+block_size, 
                               source_offset:source_offset+block_size]
            
            # Take the part we need (if source chain is longer than target)
            copy_size = min(target_size, source_block.shape[0])
            block_to_copy = source_block[:copy_size, :copy_size]
            
            # Copy to result
            result[target_offset:target_offset+copy_size, 
                  target_offset:target_offset+copy_size] = block_to_copy
            
            # Update offsets
            source_offset += source_size
            target_offset += target_size
        
        # Handle cross-correlations between chains
        for i, ((s_start_i, s_end_i), (t_start_i, t_end_i)) in enumerate(zip(source_ranges, target_ranges)):
            for j, ((s_start_j, s_end_j), (t_start_j, t_end_j)) in enumerate(zip(source_ranges, target_ranges)):
                if i < j:  # Only process each pair once
                    # Calculate sizes and offsets
                    s_size_i = 3 * (s_end_i - s_start_i + 1)
                    s_size_j = 3 * (s_end_j - s_start_j + 1)
                    t_size_i = 3 * (t_end_i - t_start_i + 1)
                    t_size_j = 3 * (t_end_j - t_start_j + 1)
                    
                    s_offset_i = 3 * sum(s_end - s_start + 1 for s_start, s_end in source_ranges[:i])
                    s_offset_j = 3 * sum(s_end - s_start + 1 for s_start, s_end in source_ranges[:j])
                    t_offset_i = 3 * sum(t_end - t_start + 1 for t_start, t_end in target_ranges[:i])
                    t_offset_j = 3 * sum(t_end - t_start + 1 for t_start, t_end in target_ranges[:j])
                    
                    # Check bounds for source indices
                    if s_offset_i < data.shape[0] and s_offset_j < data.shape[1]:
                        # Extract cross-correlation blocks with proper bounds checking
                        s_end_i = min(s_offset_i + s_size_i, data.shape[0])
                        s_end_j = min(s_offset_j + s_size_j, data.shape[1])
                        
                        block_ij = data[s_offset_i:s_end_i, s_offset_j:s_end_j]
                        
                        # Limit to target sizes
                        copy_size_i = min(t_size_i, block_ij.shape[0])
                        copy_size_j = min(t_size_j, block_ij.shape[1])
                        
                        block_to_copy_ij = block_ij[:copy_size_i, :copy_size_j]
                        
                        # Copy blocks and ensure symmetry
                        result[t_offset_i:t_offset_i+copy_size_i, 
                              t_offset_j:t_offset_j+copy_size_j] = block_to_copy_ij
                        
                        # Transpose for copying to the symmetric position
                        block_to_copy_ji = block_to_copy_ij.T
                        
                        result[t_offset_j:t_offset_j+copy_size_j, 
                              t_offset_i:t_offset_i+copy_size_i] = block_to_copy_ji
        
        return result
    
    # For eigenvector data (fixed condition checking, there was a duplicate check for data.ndim == 2)
    elif data.ndim == 1:
        n_modes = data.shape[1]
        result = np.zeros((n_target_residues, n_modes))
        
        target_idx = 0
        source_idx = 0
        
        for (s_start, s_end), (t_start, t_end) in zip(source_ranges, target_ranges):
            # Calculate number of residues in this range
            n_source_residues = s_end - s_start + 1
            n_target_residues_chain = t_end - t_start + 1
            
            # Copy the relevant range (only up to the target size)
            copy_size = min(n_target_residues_chain, data.shape[0] - source_idx)
            result[target_idx:target_idx+copy_size] = data[source_idx:source_idx+copy_size]
            
            target_idx += n_target_residues_chain
            source_idx += n_source_residues
    
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
        
    return result

def perform_svd(cov_matrix: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Perform SVD/eigendecomposition on a covariance matrix."""
    # Verify the matrix is square
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(f"Matrix is not square: shape {cov_matrix.shape}")
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Return the first n_modes
    return eigvals[:n_modes], eigvecs[:, :n_modes]

def map_eigenvectors_to_residues(eigvecs: np.ndarray) -> np.ndarray:
    """
    Map 3N-dimensional eigenvectors to per-residue magnitudes.
    Vectorized implementation for speed.
    """
    n_coords = eigvecs.shape[0]
    n_residues = n_coords // 3
    n_modes = eigvecs.shape[1]
    
    # Reshape to separate x, y, z components for each residue
    reshaped = eigvecs.reshape(n_residues, 3, n_modes)
    
    # Calculate magnitudes using numpy operations (much faster)
    magnitudes = np.sqrt(np.sum(reshaped**2, axis=1))
    
    return magnitudes

def create_residue_labels(ranges: List[Tuple[int, int]]) -> List[str]:
    """
    Create residue labels based on residue ranges.
    
    Args:
        ranges: List of residue ranges as (start, end) tuples
        
    Returns:
        List of residue labels in format 'X:N' where X is chain ID and N is residue number
    """
    labels = []
    
    for i, (start, end) in enumerate(ranges):
        chain_id = chr(65 + i)  # A, B, C, ...
        for res_num in range(start, end + 1):
            labels.append(f"{chain_id}{res_num}")
    
    return labels

def process_individual_matrix(file_path: str, n_modes: int = 10, use_long_chains: bool = False, 
                             source_ranges: List[Tuple[int, int]] = None, 
                             target_ranges: List[Tuple[int, int]] = None) -> Dict:
    """
    Process a single covariance matrix file.
    Designed for parallel processing.
    
    Args:
        file_path: Path to covariance matrix file
        n_modes: Number of modes to extract
        use_long_chains: Whether to extract from longer chains
        source_ranges: Source residue ranges if different from target
        target_ranges: Target residue ranges
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Read the matrix
        matrix = read_covariance_matrix(file_path)
        
        # Handle longer chains if needed
        if use_long_chains and (source_ranges is not None and target_ranges is not None):
            processed_matrix = extract_standard_chains_vectorized(matrix, source_ranges, target_ranges)
        else:
            processed_matrix = matrix
        
        # Perform SVD
        eigvals, eigvecs = perform_svd(processed_matrix, n_modes)
        
        # Calculate per-residue magnitudes
        magnitudes = map_eigenvectors_to_residues(eigvecs)
        
        return {
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs,
            'magnitudes': magnitudes
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_matrices_parallel(matrices_or_paths: List, 
                            n_modes: int = 10, 
                            use_long_chains: bool = False,
                            source_ranges: List[Tuple[int, int]] = None,
                            target_ranges: List[Tuple[int, int]] = None,
                            n_processes: int = None) -> Dict:
    """
    Process matrices in parallel for better performance.
    
    Args:
        matrices_or_paths: List of matrices or file paths to process
        n_modes: Number of modes to extract
        use_long_chains: Whether to extract from longer chains
        source_ranges: Source residue ranges if different from target
        target_ranges: Target residue ranges
        n_processes: Number of processes to use (default: CPU count)
        
    Returns:
        Dictionary with processing results including error statistics
    """
    if use_long_chains and (source_ranges is None or target_ranges is None):
        raise ValueError("For long chains, source_ranges and target_ranges must be specified")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(matrices_or_paths))
    
    # Check if we're processing paths or matrices
    processing_paths = isinstance(matrices_or_paths[0], str)
    
    if processing_paths:
        # Process file paths in parallel
        print(f"Processing {len(matrices_or_paths)} files using {n_processes} processes...")
        
        # Create a partial function with fixed arguments
        process_func = partial(
            process_individual_matrix,
            n_modes=n_modes,
            use_long_chains=use_long_chains,
            source_ranges=source_ranges,
            target_ranges=target_ranges
        )
        
        # Process in parallel
        with mp.Pool(processes=n_processes) as pool:
            individual_results = pool.map(process_func, matrices_or_paths)
    else:
        # Process matrices directly (serially for now)
        individual_results = []
        for matrix in matrices_or_paths:
            if use_long_chains:
                matrix = extract_standard_chains_vectorized(matrix, source_ranges, target_ranges)
            
            # Perform SVD
            eigvals, eigvecs = perform_svd(matrix, n_modes)
            
            # Calculate per-residue magnitudes
            magnitudes = map_eigenvectors_to_residues(eigvecs)
            
            individual_results.append({
                'eigenvalues': eigvals,
                'eigenvectors': eigvecs,
                'magnitudes': magnitudes
            })
    
    # Filter out None results
    individual_results = [r for r in individual_results if r is not None]
    
    if not individual_results:
        raise ValueError("No valid results after processing")
    
    # Combine the results to get averages and errors
    n_matrices = len(individual_results)
    
    # For eigenvalues
    all_eigenvalues = np.array([r['eigenvalues'] for r in individual_results])
    avg_eigenvalues = np.mean(all_eigenvalues, axis=0)
    std_eigenvalues = np.std(all_eigenvalues, axis=0)
    sem_eigenvalues = std_eigenvalues / np.sqrt(n_matrices)
    
    # For eigenvector magnitudes
    all_magnitudes = np.array([r['magnitudes'] for r in individual_results])
    avg_magnitudes = np.mean(all_magnitudes, axis=0)
    std_magnitudes = np.std(all_magnitudes, axis=0)
    sem_magnitudes = std_magnitudes / np.sqrt(n_matrices)
    
    # Process the average covariance matrix for consensus eigenvectors
    if processing_paths:
        # Read all matrices first
        matrices = []
        for path in matrices_or_paths:
            try:
                matrix = read_covariance_matrix(path)
                matrices.append(matrix)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    else:
        matrices = matrices_or_paths
    
    # Average the covariance matrices
    processed_matrices = []
    for matrix in matrices:
        if use_long_chains:
            matrix = extract_standard_chains_vectorized(matrix, source_ranges, target_ranges)
        processed_matrices.append(matrix)
    
    avg_cov = np.mean(processed_matrices, axis=0)
    consensus_eigvals, consensus_eigvecs = perform_svd(avg_cov, n_modes)
    
    # Create residue labels
    if use_long_chains:
        labels = create_residue_labels(target_ranges)
    else:
        # If not using long chains, estimate the ranges from the matrix size
        n_residues = consensus_eigvecs.shape[0] // 3
        # Assume equal split between chains
        chain_length = n_residues // len(source_ranges) if source_ranges else n_residues // 2
        
        std_ranges = []
        for i in range(len(source_ranges) if source_ranges else 2):
            std_ranges.append((2, 2 + chain_length - 1))
            
        labels = create_residue_labels(std_ranges)
    
    return {
        'eigenvalues': consensus_eigvals,
        'eigenvectors': consensus_eigvecs,
        'magnitudes': map_eigenvectors_to_residues(consensus_eigvecs),
        'labels': labels,
        'avg_eigenvalues': avg_eigenvalues,
        'std_eigenvalues': std_eigenvalues,
        'sem_eigenvalues': sem_eigenvalues,
        'avg_magnitudes': avg_magnitudes,
        'std_magnitudes': std_magnitudes,
        'sem_magnitudes': sem_magnitudes,
        'n_samples': n_matrices
    }
