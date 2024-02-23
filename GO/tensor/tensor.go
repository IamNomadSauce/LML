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

// Zero-filled Tensor
func NewZerosTensor(inputN, outputN int) *Tensor {
	// Iterate over each element in the outer slice
	sliceOfSlices := make([][]float64, inputN)
	for i := range sliceOfSlices {
		// Initialize each inner slice with length y, filled with zeros
		sliceOfSlices[i] = make([]float64, outputN)
	}

	return NewTensor(sliceOfSlices)
}

// Shape returns a new tensor representing the shape of the tensor as {rows, columns}.
func (t *Tensor) Shape() *Tensor {
	shapeData := [][]float64{{float64(t.Rows), float64(t.Cols)}}
	return NewTensor(shapeData)
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

// Add performs element-wise addition with another tensor and returns a new tensor.
// This version includes a simple form of broadcasting for bias addition.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if t.Rows == other.Rows && t.Cols == other.Cols {
		// Shapes match, perform regular element-wise addition
		resultData := make([][]float64, t.Rows)
		for i := range resultData {
			resultData[i] = make([]float64, t.Cols)
			for j := range resultData[i] {
				resultData[i][j] = t.Data[i][j] + other.Data[i][j]
			}
		}
		return NewTensor(resultData), nil
	} else if other.Rows == 1 && other.Cols == t.Cols {
		// Attempt to broadcast other tensor (bias) across the rows of t
		resultData := make([][]float64, t.Rows)
		for i := range resultData {
			resultData[i] = make([]float64, t.Cols)
			for j := range resultData[i] {
				resultData[i][j] = t.Data[i][j] + other.Data[0][j]
			}
		}
		return NewTensor(resultData), nil
	} else {
		return nil, fmt.Errorf("tensors have different shapes and cannot be broadcasted")
	}
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
	// fmt.Println("\nTensor-MatMul\n", t.Rows, t.Cols, "\n", other.Rows, other.Cols, "\n")

	if t.Cols != other.Rows {

		fmt.Println("incompatible shapes for matrix multiplication")
		return nil, fmt.Errorf("incompatible shapes for matrix multiplication")
	}
	resultData := make([][]float64, t.Rows)
	// fmt.Println("Tensor-Matrix_Multiply")

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
	// fmt.Println("MatMul Results\n", rs.Rows, rs.Cols, "\n")
	return rs, nil
}

// DotProduct performs the dot product operation on two tensors.
// For 1D tensors, it calculates the sum of the products of corresponding elements.
// For 2D tensors, it performs matrix multiplication.
func (t *Tensor) DotProduct(other *Tensor) (*Tensor, error) {
	// fmt.Println("\nTrensor DotProduct\n", t.Rows, t.Cols, "\n", other.Rows, other.Cols, "\n")
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
	// fmt.Println("Tensor- Dot_Product: Send to MatMul")
	return t.MatrixMultiply(other)
}

// Copy creates a copy of the Tensor
func (t *Tensor) Copy() *Tensor {
	dataCopy := make([][]float64, len(t.Data))
	for i := range t.Data {
		dataCopy[i] = make([]float64, len(t.Data[i]))
		copy(dataCopy[i], t.Data[i])
	}
	return NewTensor(dataCopy)
}

// Argmax returns a Tensor containing the indices of the max element in each row
func (t *Tensor) Argmax(axis int) *Tensor {
	if axis != 1 {
		panic("Argmax currently only supports axis=1")
	}
	indices := make([][]float64, t.Rows)
	for i, row := range t.Data {
		maxIdx := 0
		maxVal := row[0]
		for j, val := range row {
			if val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}
		indices[i] = []float64{float64(maxIdx)}
	}
	return NewTensor(indices)
}
