package main

import (
	"fmt"
	// t "github.com/gotensor/gotensor"
	// tensor "gobasics.dev/tensor"
)

// dotProduct performs the dot product of two 2D tensors (matrices).
func dotProduct(A, B [][]float64) ([][]float64, error) {
	m := len(A)
	n := len(A[0])
	p := len(B[0])
	// fmt.Println(m, n, p)
	// Check if the matrices are compatible for dot product
	if len(B) != n {
		return nil, fmt.Errorf("incompatible dimensions for dot product")
	}

	// Initialize the result matrix C
	C := make([][]float64, m)
	for i := range C {
		C[i] = make([]float64, p)
	}

	// Perform the dot product
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += A[i][k] * B[k][j]
			}
			C[i][j] = sum
		}
	}

	return C, nil
}

// transpose takes a 2D slice of float64 and returns its transpose.
func transpose(matrix [][]float64) [][]float64 {
	if len(matrix) == 0 {
		return nil
	}

	rows := len(matrix)
	cols := len(matrix[0])

	// Create a new 2D slice with the dimensions swapped
	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
	}

	// Copy the elements into the transposed matrix
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}

	return transposed
}

type Shape struct {
	Rows    int
	Columns int
}

func shape(matrix [][]float64) Shape {
	rows := len(matrix)
	cols := 0
	if rows > 0 {
		cols = len(matrix[0])
	}
	return Shape{Rows: rows, Columns: cols}
}

// matrixMultiply performs matrix multiplication on two 2D slices.
// It returns the resulting matrix and an error if the matrices are not compatible.
func matrixMultiply(A, B [][]float64) ([][]float64, error) {
	// Check if matrices are compatible for multiplication
	if len(A) == 0 || len(B) == 0 || len(A[0]) != len(B) {
		return nil, fmt.Errorf("matrices are not compatible for multiplication")
	}

	m := len(A)
	n := len(B[0])
	p := len(A[0]) // Also equal to len(B)

	// Initialize the result matrix with zeros
	C := make([][]float64, m)
	for i := range C {
		C[i] = make([]float64, n)
	}

	// Perform the matrix multiplication
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < p; k++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}

	return C, nil
}

func addBiases(matrix [][]float64, biases [][]float64) ([][]float64, error) {
	if len(matrix) == 0 || len(biases) == 0 || len(matrix[0]) != len(biases[0]) {
		return nil, fmt.Errorf("matrix and biases are not compatible")
	}

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] += biases[0][j] // Assuming biases is a 1xN matrix
		}
	}
	return matrix, nil
}

func main() {
	// Example matrices A and B
	inputs := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{4, 5, 6},
	}
	weights := [][]float64{
		{7, 8, 9},
	}
	biases := [][]float64{
		{2.0, 3.0, 0.5},
	}

	weightsT := transpose(weights)

	// Perform the dot product
	outputs_DP, err := dotProduct(inputs, weightsT)
	if err != nil {
		fmt.Println("Error performing dot product:", err)
		return
	}

	outputs_DP_T := transpose(outputs_DP)
	// Add biases to the dot product result
	outputsWithBiases, err := addBiases(outputs_DP_T, biases)
	if err != nil {
		fmt.Println("Error adding biases:", err, shape(outputs_DP), shape(biases))
		return
	}

	// Print the result
	fmt.Println("Result with biases added:")
	for _, row := range outputsWithBiases {
		fmt.Println(row)
	}
	// for _, row := range A {
	// 	fmt.Println(row)
	// }
	// fmt.Println("\n")
	// for _, row := range B {
	// 	fmt.Println(row)
	// }
	// fmt.Println("\n")
	// for _, row := range C {
	// 	fmt.Println(row)
	// }
	// fmt.Println("\n")
	// for _, row := range D {
	// 	fmt.Println(row)
	// }
	// fmt.Println("\n")
	// if err != nil {
	// 	fmt.Println("Error:", err)
	// 	return
	// }

	// Print the result
	// fmt.Println("Result of dot product:", C, D)
}
