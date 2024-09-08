import argparse
import numpy as np
import torch
import sys
import os

def load_and_save_c_order(file_path):
    try:
        array = np.load(file_path, allow_pickle=True)
        return file_path, array.astype(np.int32)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

def print_array_info(array, name):
    print(f"\n{name} Information:")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"Order: {'Fortran' if array.flags['F_CONTIGUOUS'] else 'C'}-contiguous")
    print(f"Min value: {np.min(array)}")
    print(f"Max value: {np.max(array)}")
    print(f"Mean value: {np.mean(array)}")

def compare_npy_files(file1, file2, tolerance):
    new_file1, array1 = load_and_save_c_order(file1)
    new_file2, array2 = load_and_save_c_order(file2)

    if array1 is None or array2 is None:
        return None

    print_array_info(array1, "Array 1")
    print_array_info(array2, "Array 2")

    if array1.shape != array2.shape:
        print(f"\nError: Arrays have different shapes: {array1.shape} vs {array2.shape}")
        return None

    # Convert numpy arrays to PyTorch tensors
    tensor1 = torch.from_numpy(array1.astype(np.int32))
    tensor2 = torch.from_numpy(array2)

    # Check if the tensors are close with the specified tolerance
    are_close = torch.allclose(tensor1, tensor2, rtol=0, atol=tolerance)

    # Calculate the absolute difference
    diff = torch.abs(tensor1 - tensor2)

    # Get statistics about the difference
    # max_diff = torch.max(diff).item()
    # mean_diff = torch.mean(diff).item()
    num_diff = torch.sum(diff > tolerance).item()

    return are_close, num_diff, tensor1.numel(), new_file1, new_file2

def main():
    parser = argparse.ArgumentParser(description='Compare two .npy files and show detailed differences')
    parser.add_argument('file1', type=str, help='Path to the first .npy file')
    parser.add_argument('file2', type=str, help='Path to the second .npy file')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Tolerance for comparison (default: 1e-3)')

    args = parser.parse_args()

    result = compare_npy_files(args.file1, args.file2, args.tolerance)

    if result is None:
        sys.exit(1)

    are_close, num_diff, total_elements, new_file1, new_file2 = result

    print(f"\nComparison results (tolerance: {args.tolerance}):")
    print(f"Arrays are close: {are_close}")
    print(f"Number of elements exceeding tolerance: {num_diff} out of {total_elements}")
    print(f"Percentage of differing elements: {(num_diff / total_elements) * 100:.2f}%")
    
    print(f"\nNew C-order files created:")
    print(f"File 1: {new_file1}")
    print(f"File 2: {new_file2}")
    print("\nPlease use these new files with Candle.rs")

if __name__ == "__main__":
    main()