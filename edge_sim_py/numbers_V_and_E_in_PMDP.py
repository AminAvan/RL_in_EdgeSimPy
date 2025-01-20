import itertools
import math


def generate_binary_matrices(n_rows, n_cols):
    """
    Generates all possible binary matrices with n_rows and n_cols adhering to the following rules:
    1. Each row has at most one '1'.
    2. Each column has at most floor(n_rows / 2) '1's.
    3. Matrices are generated in order of increasing total number of '1's.

    ***PLEASE NOTE that for larger matrices (with more than 20 rows or columns), the code may become time-consuming.***
    """
    from itertools import combinations
    matrices = []
    column_limit = math.floor(n_rows / 2)
    max_sum = min(n_rows, n_cols * column_limit)

    order = 1
    for s in range(1, max_sum + 1):
        # Select s rows out of n_rows
        # print(f"n_rows:{n_rows}")
        # print(f"s:{s}")
        row_combinations = list(combinations(range(order), s))
        # print(f"row_combinations:{row_combinations}")
        # if 1 <= order <= len(row_combinations):
        #     row_combinations = [row_combinations[order - 1]]
        order += 1
        # print(f"row_combinations:{row_combinations}")
        # print()
        for rows_selected in row_combinations:
            # print(f"row_combinations:{row_combinations}")
            # Generate all possible column assignments for these rows
            assignments = []
            current_assignment = {}
            column_counts = [0] * n_cols

            def backtrack(index):
                if index == len(rows_selected):
                    # Complete assignment
                    assignments.append(current_assignment.copy())
                    return
                row = rows_selected[index]
                for col in range(n_cols):
                    if column_counts[col] < column_limit:
                        # Assign this column to the current row
                        current_assignment[row] = col
                        # print(f"current_assignment[row]:{current_assignment[row]}")
                        column_counts[col] += 1
                        # print(f"column_counts[col]:{column_counts[col]}")
                        backtrack(index + 1)
                        # Backtrack
                        column_counts[col] -= 1
                        del current_assignment[row]

            backtrack(0)

            ## [for drawing the actual matrix -- it face long time issue when the matrix get big
            # # For each assignment, create the binary matrix
            # for assign in assignments:
            #     matrix = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
            #     for row, col in assign.items():
            #         matrix[row][col] = 1
            #     matrices.append(matrix)
            ## for drawing the actual matrix -- it face long time issue when the matrix get big]

            # For each assignment, create the binary matrix
            for assign in assignments:
                matrices.append(1)


    ## [for drawing the actual matrix -- it face long time issue when the matrix get big
    # Optionally, sort the matrices by the number of '1's and lex order
    # matrices_sorted = sorted(matrices, key=lambda m: (sum(sum(row) for row in m), m))
    # return matrices_sorted
    ## for drawing the actual matrix -- it face long time issue when the matrix get big]
    return matrices


# Example usage:
if __name__ == "__main__":
    n_rows = int(input("Enter number of rows: "))
    n_cols = int(input("Enter number of columns: "))
    matrices = generate_binary_matrices(n_rows, n_cols)
    worst_vertices = (len(matrices) - 1)
    # len(matrices) contain the root as well as the best_vertices
    print(f"\nTotal vertices: {len(matrices) + worst_vertices}")
    print(f"Total edges: {(len(matrices) + worst_vertices) - 1}")
    ## [for drawing the actual matrix -- it face long time issue when the matrix get big (with more than 20 rows or columns)
    # for idx, mat in enumerate(matrices, 1):
    #     print(f"Matrix {idx}:")
    #     for row in mat:
    #         print(row)
    #     print()
    ## [for drawing the actual matrix -- it face long time issue when the matrix get big (with more than 20 rows or columns)

