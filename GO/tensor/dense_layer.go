package tensor

import (
	"fmt"
	"math/rand"
)

type LayerDense struct {
	Weights *Tensor
	Biases  *Tensor
	Outputs *Tensor
}

func New_Dense_Layer(nInputs, nOutputs int64) LayerDense {

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
	fmt.Println("DL_Forward: Inputs\n", inputs.Data, inputs.Rows, inputs.Cols, "\n")
	fmt.Println("DL_Forward: Weights\n", d.Weights.Data, d.Weights.Rows, d.Weights.Cols, "\n")

	dp, _ := inputs.DotProduct(d.Weights)
	fmt.Println("DL-DP", dp.Rows, dp.Cols)
	for _, row := range dp.Data {
		fmt.Println("Row", row)
	}
	// fmt.Println("Dense_Layer Dot_Product\n", dp.Rows, dp.Cols, "\n", d.Biases.Rows, d.Biases.Cols, "\n")
	// output, _ := dp.Add(d.Biases)
	// fmt.Println("D:", d.Outputs, "\nOutput\n", output, "\n")
}
