package tensor

import (
	"fmt"
	"math/rand"
	"time"
)

type LayerDense struct {
	Weights *Tensor
	Biases  *Tensor
	Outputs *Tensor
}

func New_Dense_Layer(nInputs, nOutputs int64) LayerDense {

	rand.Seed(time.Now().UnixNano())
	weights := [][]float64{
		{-0.0255299, 0.00653619, 0.00864436},
		{-0.00742165, 0.02269755, -0.01454366},
	}
	// TODO Random numbers generated here are different results than python
	// weights := make([][]float64, nInputs)
	// for i := range weights {
	// 	weights[i] = make([]float64, nOutputs)
	// 	for j := range weights[i] {
	// 		weights[i][j] = 0.01 * rand.NormFloat64()
	// 	}
	// }

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

	// Perform the dot_product on the inputs using weights
	dp, _ := inputs.DotProduct(d.Weights)
	fmt.Println("DL-DP", dp.Rows, dp.Cols)
	for _, row := range dp.Data {
		fmt.Println("Row", row)
	}
	// // Replicate/Reshape biases to match the shape of dotProduct
	replicatedBiases := reshape(d.Biases, len(dp.Data))

	// Add biases to dot_products
	output, _ := dp.Add(replicatedBiases)
	fmt.Println("DL: Dot_Product + Biases")
	for _, row := range output.Data {
		fmt.Println("Row", row)
	}
	fmt.Println("D:", d.Outputs, "\nOutputs\n", "\n", output.Rows, output.Cols, "\n")

	// // Add biases to dotProduct
	// dotProductWithBiases, err := dotProduct.Add(replicatedBiases)
	// if err != nil {
	// 	fmt.Println("Error adding biases:", err)
	// 	return
	// }

	// // Print the new tensor with biases added
	// fmt.Println("dotProduct with biases added:")
	// for _, row := range dotProductWithBiases.Data {
	// 	fmt.Println(row)
	// }

	// // fmt.Println("Dense_Layer Dot_Product\n", dp.Rows, dp.Cols, "\n", d.Biases.Rows, d.Biases.Cols, "\n")
	// fmt.Println("D:", d.Outputs, "\nOutput\n", output, "\n")
}
