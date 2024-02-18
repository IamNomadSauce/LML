package main

import (
	"fmt"
	"go_nn/tensor"
	"math/rand"
	"time"
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

	X := [][]float64{
		{0., 0.},
		{0.10738789, 0.02852226},
		{0.09263825, -0.20199226},
		{-0.32224888, -0.08524539},
		{-0.3495118, 0.27454028},
		{-0.52100587, 0.19285966},
		{0.5045865, 0.43570277},
		{0.76882404, 0.11767714},
		{0.49269393, -0.73984873},
		{-0.70364994, -0.71054685},
		{-0., -0.},
		{-0.07394107, 0.08293611},
		{0.00808054, 0.22207525},
		{0.24548167, 0.22549914},
		{0.38364738, -0.22437814},
		{-0.00801609, -0.5554977},
		{-0.66060567, 0.08969161},
		{-0.7174548, 0.30032802},
		{0.17299275, 0.87189275},
		{0.66193414, 0.74956197},
	}

	X_tensor := tensor.NewTensor(X)

	// inputs := gen_data(100)
	for _, row := range X_tensor.Data {
		fmt.Println("Row", row)
	}
	fmt.Println("X_tensor", X_tensor.Rows, X_tensor.Cols)
	// fmt.Println("Inputs\n", inputs.Rows, inputs.Cols, "\n")

	layer := tensor.New_Dense_Layer(2, 3)
	layer2 := tensor.New_Dense_Layer(3, 1)
	// fmt.Println("Layer Weights:\n", layer.Weights.Rows, layer.Weights.Cols, "\n")
	// fmt.Println("Layer Biases:\n", layer.Biases.Rows, layer.Biases.Cols, "\n")

	layer.Forward(X_tensor)
	fmt.Println("Main: Layer1 Forward", layer.Outputs.Rows, layer.Outputs.Cols)

	layer2.Forward(layer.Outputs)

	fmt.Println("Main: Layer Forward", layer2.Outputs.Rows, layer2.Outputs.Cols)

}

// generateRandomTuples generates a slice of tuples, each with two random floats in the range of -10 to 10.
func gen_data(samples int) *tensor.Tensor { // Note the use of tensor.Tensor
	// Seed the random number generator using the current time
	rand.Seed(time.Now().UnixNano())

	// Create a 2D slice to hold the random floats
	data := make([][]float64, samples)
	for i := range data {
		data[i] = make([]float64, 2) // Each sample has 2 floats
		for j := range data[i] {
			// Generate a random number between 0.0 and 1.0
			randomFloat := rand.Float64()
			// Adjust the range to be between -10 and 10
			randomFloat = randomFloat*2 - 1
			data[i][j] = randomFloat
		}
	}

	// Create and return a new Tensor with the generated data
	return tensor.NewTensor(data) // Use the NewTensor function from the tensor package
}
