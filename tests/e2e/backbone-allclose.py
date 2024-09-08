import argparse
import numpy as np
import torch
import sys
import os

def load_and_save_c_order(file_path):
    try:
        array = np.load(file_path, allow_pickle=True)
        array = np.ascontiguousarray(array)
        new_file_path = file_path.replace('.npy', '_c_order.npy')
        # np.save(new_file_path, array)
        # print(f"Saved C-order array to {new_file_path}")
        return new_file_path, array
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

def find_difference_locations(tensor1, tensor2, tolerance):
    diff = torch.abs(tensor1 - tensor2)
    diff_locations = torch.nonzero(diff > tolerance, as_tuple=False)
    return diff_locations

def summarize_differences(diff_locations, tensor_shape):
    if len(diff_locations) == 0:
        return "No differences found."

    summary = []
    for dim in range(len(tensor_shape)):
        dim_values = diff_locations[:, dim].tolist()
        unique_values = sorted(set(dim_values))
        if len(unique_values) == tensor_shape[dim]:
            summary.append(f"Dimension {dim}: All values")
        elif len(unique_values) <= 10:
            summary.append(f"Dimension {dim}: {unique_values}")
        else:
            summary.append(f"Dimension {dim}: Range [{min(unique_values)}, {max(unique_values)}]")

    return ", ".join(summary)

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

    tensor1 = torch.from_numpy(array1)
    tensor2 = torch.from_numpy(array2)

    are_close = torch.allclose(tensor1, tensor2, rtol=0, atol=tolerance)
    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    num_diff = torch.sum(diff > tolerance).item()

    diff_locations = find_difference_locations(tensor1, tensor2, tolerance)
    diff_summary = summarize_differences(diff_locations, tensor1.shape)

    return are_close, max_diff, mean_diff, num_diff, tensor1.numel(), new_file1, new_file2, diff_summary

def main():
    parser = argparse.ArgumentParser(description='Compare two .npy files and show detailed differences')
    parser.add_argument('file1', type=str, help='Path to the first .npy file')
    parser.add_argument('file2', type=str, help='Path to the second .npy file')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Tolerance for comparison (default: 1e-3)')

    args = parser.parse_args()

    result = compare_npy_files(args.file1, args.file2, args.tolerance)

    if result is None:
        sys.exit(1)

    are_close, max_diff, mean_diff, num_diff, total_elements, new_file1, new_file2, diff_summary = result

    print(f"\nComparison results (tolerance: {args.tolerance}):")
    print(f"Arrays are close: {are_close}")
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Number of elements exceeding tolerance: {num_diff} out of {total_elements}")
    print(f"Percentage of differing elements: {(num_diff / total_elements) * 100:.2f}%")
    print(f"\nDifference locations summary:")
    print(diff_summary)
    
    # print(f"\nNew C-order files created:")
    # print(f"File 1: {new_file1}")
    # print(f"File 2: {new_file2}")
    print("\nPlease use these new files with Candle.rs")

if __name__ == "__main__":
    main()