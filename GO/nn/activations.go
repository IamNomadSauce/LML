package nn

import (
	"fmt"
	"go_nn/tensor"
	_ "go_nn/tensor"
	"math"
	"math/rand"
	"time"
)

// ----------------------------------------------------------------
// ReLU Activation
// ----------------------------------------------------------------

// Activation_ReLU represents a ReLU activation layer
type Activation_ReLU struct {
	Inputs  *tensor.Tensor
	Output  *tensor.Tensor
	DInputs *tensor.Tensor
}

// New_ReLU_Activation creates a new instance of Activation_ReLU.
func New_ReLU_Activation() *Activation_ReLU {
	return &Activation_ReLU{}
}

// Forward performs the forward pass
func (a *Activation_ReLU) Forward(Inputs *tensor.Tensor, training bool) {
	// Remember input values
	a.Inputs = Inputs
	// Calculate Output values from Inputs
	OutputData := make([][]float64, Inputs.Rows)
	for i := range OutputData {
		OutputData[i] = make([]float64, Inputs.Cols)
		for j := range OutputData[i] {
			OutputData[i][j] = math.Max(0, Inputs.Data[i][j])
		}
	}
	// fmt.Println("\nActivation Relu", OutputData, "\n")
	a.Output = tensor.NewTensor(OutputData)
}

// Backward performs the backward pass
func (a *Activation_ReLU) Backward(dvalues *tensor.Tensor) {
	// Make a copy of values first
	DInputsData := make([][]float64, dvalues.Rows)
	for i := range DInputsData {
		DInputsData[i] = make([]float64, dvalues.Cols)
		copy(DInputsData[i], dvalues.Data[i])
	}
	a.DInputs = tensor.NewTensor(DInputsData)

	// Zero gradient where input values were negative
	for i := range a.Inputs.Data {
		for j := range a.Inputs.Data[i] {
			if a.Inputs.Data[i][j] <= 0 {
				a.DInputs.Data[i][j] = 0
			}
		}
	}
}

// ----------------------------------------------------------------
// Softmax
// ----------------------------------------------------------------

// ActivationSoftmax represents a Softmax activation layer
type ActivationSoftmax struct {
	Inputs               *tensor.Tensor
	Output               *tensor.Tensor
	DInputs              *tensor.Tensor
	GetBiasRegularizerL1 float64
	GetBiasRegularizerL2 float64
	Layer
}

// NewActivationSoftmax creates a new instance of ActivationSoftmax.
func New_Softmax_Activation() *ActivationSoftmax {
	return &ActivationSoftmax{}
}

// Forward performs the forward pass
func (a *ActivationSoftmax) Forward(Inputs *tensor.Tensor, training bool) {
	fmt.Println("Activation Softmax Forward")
	// Remember input values
	a.Inputs = Inputs

	// fmt.Println("Activation Inputs", Inputs)
	// Get unnormalized probabilities
	OutputData := make([][]float64, Inputs.Rows)
	for i, row := range Inputs.Data {
		// Find the max value in the row to prevent "exp" overflow
		rowMax := row[0]
		for _, val := range row {
			if val > rowMax {
				rowMax = val
			}
		}
		// Subtract the max and exponentiate
		expValues := make([]float64, len(row))
		expSum := 0.0
		for j, val := range row {
			expValues[j] = math.Exp(val - rowMax)
			expSum += expValues[j]
		}
		// Normalize the probabilities
		OutputData[i] = make([]float64, len(row))
		for j, expVal := range expValues {
			OutputData[i][j] = expVal / expSum
		}
	}
	a.Output = tensor.NewTensor(OutputData)
}

// Backward performs the backward pass
func (a *ActivationSoftmax) Backward(dvalues *tensor.Tensor) {
	// Create uninitialized array
	DInputsData := make([][]float64, dvalues.Rows)
	for i, singleOutput := range a.Output.Data {
		// Calculate Jacobian matrix of the Output
		jacobianMatrix := make([][]float64, len(singleOutput))
		for j := range jacobianMatrix {
			jacobianMatrix[j] = make([]float64, len(singleOutput))
			for k := range jacobianMatrix[j] {
				if j == k {
					jacobianMatrix[j][k] = singleOutput[j] * (1 - singleOutput[k])
				} else {
					jacobianMatrix[j][k] = -singleOutput[j] * singleOutput[k]
				}
			}
		}
		// Calculate sample-wise gradient and add it to the array of sample gradients
		DInputsData[i] = make([]float64, len(singleOutput))
		for j := range dvalues.Data[i] {
			for k := range jacobianMatrix {
				DInputsData[i][j] += jacobianMatrix[j][k] * dvalues.Data[i][k]
			}
		}
	}
	a.DInputs = tensor.NewTensor(DInputsData)
}

// ----------------------------------------------------------------
// Sigmoid
// ----------------------------------------------------------------

// ActivationSigmoid represents a Sigmoid activation layer
type ActivationSigmoid struct {
	Inputs *tensor.Tensor
	Output *tensor.Tensor
}

// NewSigmoidActivation creates a new instance of ActivationSigmoid.
func NewSigmoidActivation() *ActivationSigmoid {
	return &ActivationSigmoid{}
}

// Forward performs the forward pass
func (a *ActivationSigmoid) Forward(inputs *tensor.Tensor, training bool) {
	// Remember input values
	a.Inputs = inputs
	// Calculate Output values from Inputs
	outputData := make([][]float64, len(inputs.Data))
	for i, row := range inputs.Data {
		outputData[i] = make([]float64, len(row))
		for j, val := range row {
			outputData[i][j] = 1 / (1 + math.Exp(-val))
		}
	}
	a.Output = tensor.NewTensor(outputData)
}

type LinearLayer struct {
	Weights *tensor.Tensor
	Biases  *tensor.Tensor
}

// ----------------------------------------------------------------
// Linear
// ----------------------------------------------------------------

// NewLinearLayer creates a new linear layer with given input and output sizes
func NewLinearLayer(inputDim, outputDim int, inputs *tensor.Tensor) *LinearLayer {
	fmt.Println("\nNew Linear Layer", inputDim, outputDim)

	// Initialize weights with Xavier initialization
	weightsData := XavierInit(outputDim, inputDim)
	weights := tensor.NewTensor(weightsData)

	// Initialize biases with zeros
	biasesData := make([][]float64, outputDim)
	for i := range biasesData {
		biasesData[i] = make([]float64, 1) // Only one bias per output unit
	}
	biases := tensor.NewTensor(biasesData)

	fmt.Println("Weights:", weights.Data, "\n", weights.Rows, weights.Cols, "\nBiases", biases.Data, "\n", biases.Rows, biases.Cols)

	return &LinearLayer{Weights: weights, Biases: biases}
}

// Forward performs the forward pass
func (l *LinearLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	// Perform matrix multiplication between input and weights
	fmt.Println("\nLinear_Layer Forward", input.Rows, input.Cols)
	output, err := input.MatrixMultiply(l.Weights)
	// for _, row := range output.Data {
	// 	fmt.Println("Row", row)
	// }
	fmt.Println("LL-Forward-Outputs", output.Rows, output.Cols)
	if err != nil {
		fmt.Println("Error during forward pass matrix multiplication:", err)
		return nil
	}

	// Add biases to the result of the matrix multiplication
	output, err = output.Add(l.Biases)
	if err != nil {
		fmt.Println("Error during forward pass bias addition:", err)
		return nil
	}

	return output
}

// -------

// XavierInit initializes a slice of slices with Xavier/Glorot uniform distribution
// inputDim is the number of input units, outputDim is the number of output units
func XavierInit(inputDim, outputDim int) [][]float64 {
	rand.Seed(time.Now().UnixNano()) // Seed the random number generator

	lower := -math.Sqrt(6.0 / float64(inputDim+outputDim))
	upper := math.Sqrt(6.0 / float64(inputDim+outputDim))
	weights := make([][]float64, outputDim)

	for i := range weights {
		weights[i] = make([]float64, inputDim)
		for j := range weights[i] {
			weights[i][j] = lower + rand.Float64()*(upper-lower) // U(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
		}
	}

	return weights
}
