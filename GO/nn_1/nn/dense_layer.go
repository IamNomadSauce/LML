package nn

import (
	"fmt"
	"go_nn/tensor"
	_ "go_nn/tensor"
	"math/rand"
	"time"
)

// Layer defines the methods that all neural network layers should implement.
type Layer interface {
	Forward(inputs *tensor.Tensor, training bool)
	Backward(dvalues *tensor.Tensor)
	SetPrev(layer Layer)
	SetNext(layer Layer)
}

// TrainableLayer defines methods for layers with trainable parameters.
type TrainableLayer interface {
	Layer
	GetWeights() *tensor.Tensor
	GetBiases() *tensor.Tensor
	GetWeightRegularizerL1() float64
	GetWeightRegularizerL2() float64
	GetBiasRegularizerL1() float64
	GetBiasRegularizerL2() float64
}

type LayerDense struct {
	Weights               *tensor.Tensor
	Biases                *tensor.Tensor
	Outputs               *tensor.Tensor
	DInputs               *tensor.Tensor
	InputsN               int64
	OutputsN              int64
	DWeights              *tensor.Tensor
	DBiases               *tensor.Tensor
	Weight_Regularizer_L1 float64
	Weight_Regularizer_L2 float64
	Bias_Regularizer_L1   float64
	Bias_Regularizer_L2   float64
}

// LayerInput represents an input layer
type LayerInput struct {
	Output *tensor.Tensor
	// GetWeightRegularizerL1 float64
	// GetWeightRegularizerL2 float64
	// GetBiasRegularizerL1   float64
	// GetBiasRegularizerL2   float64
}

// NewLayerInput creates a new instance of LayerInput
func NewLayerInput() *LayerInput {
	return &LayerInput{}
}
func New_Dense_Layer(nInputs, nOutputs int64) LayerDense {
	// fmt.Println("New Dense Layer", nInputs, nOutputs)

	rand.Seed(time.Now().UnixNano())
	// weights := [][]float64{
	// 	{-0.0255299, 0.00653619, 0.00864436},
	// 	{-0.00742165, 0.02269755, -0.01454366},
	// }
	// TODO Random numbers generated here are different results than python
	weights := make([][]float64, nInputs)
	for i := range weights {
		weights[i] = make([]float64, nOutputs)
		for j := range weights[i] {
			weights[i][j] = 0.01 * rand.NormFloat64()
		}
	}

	biasesData := make([][]float64, 1)
	biasesData[0] = make([]float64, nOutputs) // Zeros intialization\

	weights_tensor := tensor.NewTensor(weights)
	biases_tensor := tensor.NewTensor(biasesData)

	return LayerDense{
		Weights: weights_tensor,
		Biases:  biases_tensor,
	}
}

// Forward Pass
func (d *LayerDense) Forward(inputs *tensor.Tensor) {
	// Calculate output values from inputs, weights and biases
	// fmt.Println("DL_Forward: Inputs\n", d.InputsN, d.OutputsN, inputs.Rows, inputs.Cols, "\n")
	// fmt.Println("DL_Forward: Weights\n", d.Weights.Rows, d.Weights.Cols, "\n")

	// Perform the dot_product on the inputs using weights
	dp, _ := inputs.DotProduct(d.Weights)
	// fmt.Println("DL-DP", dp.Rows, dp.Cols)
	// for _, row := range dp.Data {
	// 	fmt.Println("Row", row)
	// }
	// // Replicate/Reshape biases to match the shape of dotProduct
	replicatedBiases := tensor.Reshape(d.Biases, len(dp.Data))

	// Add biases to dot_products
	output, _ := dp.Add(replicatedBiases)
	d.Outputs = output
	// fmt.Println("DL: Dot_Product + Biases")
	// for _, row := range output.Data {
	// 	fmt.Println("Row", row)
	// }
	// fmt.Println("d.Outputs:", d.Outputs.Rows, d.Outputs.Cols, "\n")
}

// Backward Pass
// Backward performs the backward pass
func (l *LayerDense) Backward(dvalues *tensor.Tensor) *tensor.Tensor {
	// Gradients on parameters
	l.DWeights, _ = l.Outputs.Transpose().DotProduct(dvalues)
	fmt.Println("l.OutputsN", l.OutputsN)
	l.DBiases = tensor.NewZerosTensor(1, int(l.OutputsN)) // Initialize DBiases with zeros
	fmt.Println("Dense Layer Backward\nl.DWeights:", l.DWeights.Shape().Data, "\nl.DBiases: ", l.DBiases.Shape().Data)
	for i := 0; i < dvalues.Rows; i++ {
		for j := 0; j < dvalues.Cols; j++ {
			l.DBiases.Data[0][j] += dvalues.Data[i][j]
		}
	}

	// Gradients on regularization
	// L1 on weights
	if l.Weight_Regularizer_L1 > 0 {
		dL1 := make([][]float64, l.Weights.Rows)
		for i := range dL1 {
			dL1[i] = make([]float64, l.Weights.Cols)
			for j := range dL1[i] {
				if l.Weights.Data[i][j] < 0 {
					dL1[i][j] = -1
				} else {
					dL1[i][j] = 1
				}
			}
		}
		dL1Tensor := tensor.NewTensor(dL1)
		l.DWeights, _ = l.DWeights.Add(dL1Tensor)
	}
	// L2 on Weights
	if l.Weight_Regularizer_L2 > 0 {
		for i := range l.Weights.Data {
			for j := range l.Weights.Data[i] {
				l.DWeights.Data[i][j] += 2 * l.Weight_Regularizer_L2 * l.Weights.Data[i][j]
			}
		}
	}
	// L1 on biases
	if l.Bias_Regularizer_L1 > 0 {
		dL1 := make([][]float64, l.Biases.Rows)
		for i := range dL1 {
			dL1[i] = make([]float64, l.Biases.Cols)
			for j := range dL1[i] {
				if l.Biases.Data[i][j] < 0 {
					dL1[i][j] = -1
				} else {
					dL1[i][j] = 1
				}
			}
		}
		dL1Tensor := tensor.NewTensor(dL1)
		l.DBiases, _ = l.DBiases.Add(dL1Tensor)
	}
	// L2 on Biases
	if l.Bias_Regularizer_L2 > 0 {
		for i := range l.Biases.Data {
			for j := range l.Biases.Data[i] {
				l.DBiases.Data[i][j] += 2 * l.Bias_Regularizer_L2 * l.Biases.Data[i][j]
			}
		}
	}

	// Gradient on values
	l.DInputs, _ = dvalues.DotProduct(l.Weights.Transpose())
	return l.DInputs
}

// ------------
// DropoutLayer represents a dropout layer
type DropoutLayer struct {
	Output *tensor.Tensor
	Rate   float64        // The dropout rate, i.e., the probability of setting a neuron to zero
	Mask   *tensor.Tensor // A mask tensor that determines which units are dropped
}

// NewDropoutLayer creates a new instance of DropoutLayer
func NewDropoutLayer(rate float64) *DropoutLayer {
	return &DropoutLayer{
		Rate: rate,
	}
}

// Forward performs the forward pass with dropout
func (d *DropoutLayer) Forward(input *tensor.Tensor, training bool) *tensor.Tensor {
	if !training {
		// During inference, return the input as is
		return input
	}

	// During training, create a mask to randomly drop units
	maskData := make([][]float64, input.Rows)
	for i := range maskData {
		maskData[i] = make([]float64, input.Cols)
		for j := range maskData[i] {
			if rand.Float64() > d.Rate {
				maskData[i][j] = 1.0 / (1.0 - d.Rate) // Scale the activations to not reduce the expected value
			} else {
				maskData[i][j] = 0.0
			}
		}
	}
	d.Mask = tensor.NewTensor(maskData)

	// Apply the mask to the input
	output := tensor.ElementWiseMultiply(input, d.Mask)
	d.Output = output
	return output
}

// Backward performs the backward pass for the dropout layer
func (d *DropoutLayer) Backward(dvalues *tensor.Tensor) *tensor.Tensor {
	// Element-wise multiplication of the incoming gradient with the mask
	dInputs := tensor.ElementWiseMultiply(dvalues, d.Mask)
	return dInputs
}
