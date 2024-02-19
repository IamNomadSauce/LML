package tensor

import (
	"fmt"
	"math/rand"
	"time"
)

type LayerDense struct {
	Weights  *Tensor
	Biases   *Tensor
	Outputs  *Tensor
	DInputs  *Tensor
	InputsN  int64
	OutputsN int64
}

func New_Dense_Layer(nInputs, nOutputs int64) LayerDense {
	fmt.Println("New Dense Layer", nInputs, nOutputs)

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

	weights_tensor := NewTensor(weights)
	biases_tensor := NewTensor(biasesData)

	return LayerDense{
		Weights: weights_tensor,
		Biases:  biases_tensor,
	}
}

// Forward Pass
func (d *LayerDense) Forward(inputs *Tensor) {
	// Calculate output values from inputs, weights and biases
	fmt.Println("DL_Forward: Inputs\n", d.InputsN, d.OutputsN, inputs.Rows, inputs.Cols, "\n")
	fmt.Println("DL_Forward: Weights\n", d.Weights.Data, d.Weights.Rows, d.Weights.Cols, "\n")

	// Perform the dot_product on the inputs using weights
	dp, _ := inputs.DotProduct(d.Weights)
	// fmt.Println("DL-DP", dp.Rows, dp.Cols)
	// for _, row := range dp.Data {
	// 	fmt.Println("Row", row)
	// }
	// // Replicate/Reshape biases to match the shape of dotProduct
	replicatedBiases := reshape(d.Biases, len(dp.Data))

	// Add biases to dot_products
	output, _ := dp.Add(replicatedBiases)
	d.Outputs = output
	fmt.Println("DL: Dot_Product + Biases")
	for _, row := range output.Data {
		fmt.Println("Row", row)
	}
	fmt.Println("d.Outputs:", d.Outputs.Rows, d.Outputs.Cols, "\n")
}

// Backward Pass
