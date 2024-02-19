package tensor

import (
	"fmt"
)

// type Shape struct {
// 	Row  []int64
// 	cols []int64
// }

// Tensor represents a 2D tensor or matrix
type Tensor struct {
	Data [][]float64
	Rows int
	Cols int
}

// NewTensor creates a new Tensor given its data.
func NewTensor(data [][]float64) *Tensor {
	rows := len(data)
	cols := 0
	if rows > 0 {
		cols = len(data[0])
	}
	return &Tensor{Data: data, Rows: rows, Cols: cols}
}

// Shape returns the shape of the tensor as {rows, columns}
func (t *Tensor) Shape() (int, int) {
	fmt.Println("Tensor", t)
	return t.Rows, t.Cols
}

// Transpose returns a new tensor that is the transpose of the original
func (t *Tensor) Transpose() *Tensor {
	transposedData := make([][]float64, t.Cols)
	for i := range transposedData {
		transposedData[i] = make([]float64, t.Rows)
		for j := range transposedData[i] {
			transposedData[i][j] = t.Data[j][i]
		}
	}
	return NewTensor(transposedData)
}

// Add performs element-wise addition with another tensor and returns a new tensor
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	fmt.Println("\nAdd\nT:\n", t.Rows, t.Cols, "\nOther:\n", other.Rows, other.Cols)
	if t.Rows != other.Rows || t.Cols != other.Cols {
		fmt.Println("Tensors Have Different Shapes!\n", t.Rows, t.Cols, "\n", other.Rows, other.Cols, "\n")
		return nil, fmt.Errorf("tensors have different shapes")
	}
	resultData := make([][]float64, t.Rows)
	for i := range resultData {
		resultData[i] = make([]float64, t.Cols)
		for j := range resultData[i] {
			resultData[i][j] = t.Data[i][j] + other.Data[i][j]
		}
	}
	return NewTensor(resultData), nil
}

func Reshape(biases *Tensor, numRows int) *Tensor {
	replicatedBiasesData := make([][]float64, numRows)
	for i := range replicatedBiasesData {
		replicatedBiasesData[i] = make([]float64, len(biases.Data[0]))
		for j := range biases.Data[0] {
			replicatedBiasesData[i][j] = biases.Data[0][j]
		}
	}
	return NewTensor(replicatedBiasesData)
}

// MatrixMultiply performs matrix multiplication with another tensor and returns a new tensor
func (t *Tensor) MatrixMultiply(other *Tensor) (*Tensor, error) {
	fmt.Println("\nTensor-MatMul\n", t.Rows, t.Cols, "\n", other.Rows, other.Cols, "\n")

	if t.Cols != other.Rows {

		fmt.Println("incompatible shapes for matrix multiplication")
		return nil, fmt.Errorf("incompatible shapes for matrix multiplication")
	}
	resultData := make([][]float64, t.Rows)
	fmt.Println("Tensor-Matrix_Multiply")

	// TODO Implement gonum?
	for i := range resultData {
		resultData[i] = make([]float64, other.Cols)
		for j := range resultData[i] {
			for k := 0; k < t.Cols; k++ {
				resultData[i][j] += t.Data[i][k] * other.Data[k][j]

			}
		}
	}
	rs := NewTensor(resultData)
	fmt.Println("MatMul Results\n", rs.Rows, rs.Cols, "\n")
	return rs, nil
}

// DotProduct performs the dot product operation on two tensors.
// For 1D tensors, it calculates the sum of the products of corresponding elements.
// For 2D tensors, it performs matrix multiplication.
func (t *Tensor) DotProduct(other *Tensor) (*Tensor, error) {
	fmt.Println("\nTrensor DotProduct\n", t.Rows, t.Cols, "\n", other.Rows, other.Cols, "\n")
	// Check if both tensors are vectors (1D)
	if t.Cols == 1 && other.Cols == 1 && t.Rows == len(other.Data) {
		resultData := make([][]float64, 1)
		resultData[0] = make([]float64, 1)
		for i := 0; i < t.Rows; i++ {
			resultData[0][0] += t.Data[i][0] * other.Data[i][0]

		}
		return NewTensor(resultData), nil
	}

	// Otherwise, perform matrix multiplication (2D tensors)
	fmt.Println("Tensor- Dot_Product: Send to MatMul")
	return t.MatrixMultiply(other)
}

// func main() {

// 	// Example usage for 2D tensors (matrices)
// 	inputs := NewTensor([][]float64{
// 		{1, 2, 3, 2.5},
// 		{2, 5, -1, 2},
// 		{-1.5, 2.7, 3.3, -0.8},
// 	})
// 	// fmt.Println(inputs.Data)
// 	weights := NewTensor([][]float64{
// 		{0.2, 0.8, -0.5, 1.0},
// 		{0.5, -0.91, 0.26, -0.5},
// 		{-0.26, -0.27, 0.17, 0.87},
// 	})
// 	// fmt.Println(weights.Data)
// 	biases := NewTensor([][]float64{
// 		{2, 3, 0.5},
// 	})
// 	fmt.Println(biases.Data)

// 	dotProduct, err := inputs.DotProduct(weights.Transpose())
// 	if err != nil {
// 		fmt.Println("Error calculating dot product:", err)
// 		return
// 	}
// 	fmt.Println("Dot Product Shape:")
// 	fmt.Println(dotProduct.Shape())
// 	fmt.Println("Biases Shape:")
// 	fmt.Println(biases.Shape())

// 	// Loop through the rows of dotProduct
// 	for i, row := range dotProduct.Data {
// 		fmt.Printf("Row %d: %v\n", i, row)
// 	}
// 	fmt.Println("")

// 	// forward

// 	// Replicate biases to match the shape of dotProduct
// 	replicatedBiases := reshape(biases, len(dotProduct.Data))

// 	// Add biases to dotProduct
// 	dotProductWithBiases, err := dotProduct.Add(replicatedBiases)
// 	if err != nil {
// 		fmt.Println("Error adding biases:", err)
// 		return
// 	}

// 	// Print the new tensor with biases added
// 	fmt.Println("dotProduct with biases added:")
// 	for _, row := range dotProductWithBiases.Data {
// 		fmt.Println(row)
// 	}
// }
