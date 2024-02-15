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
	fmt.Println(m, n, p)
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
func main() {
	// Example matrices A and B
	inputs := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	weights := [][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	}

	weightsT := transpose(weights)

	// Perform the dot product
	// C, err := dotProduct(inputs, weights)
	// fmt.Println(err)
	outputs, err := dotProduct(inputs, weightsT)
	fmt.Println(err, shape(inputs), shape(weightsT))
	// fmt.Println("Inputs:\n", inputs, shape(inputs), "\nWeights:\n ", weights, shape(weights), "\nDot Product inputs*B:\n ", C, shape(C), "\nTransposed Weights:\n ", weightsT, shape(weightsT))
	fmt.Println("I*W^T", outputs, shape(outputs))
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
