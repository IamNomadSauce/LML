package nn

import (
	"fmt"
	_ "fmt"
	"go_nn/tensor"
	_ "math"
)

type Model struct {
	N_Input       int
	N_Hidden      int
	N_Outputs     int
	HiddenLayer   LayerDense
	OutputLayer   LayerDense
	Learning_Rate float64
}

func NewModel(nInput, nHidden, nOutputs int, learningRate float64) *Model {
	// Initialize the hidden and output layers
	hiddenLayer := New_Dense_Layer(int64(nInput), int64(nHidden))
	outputLayer := New_Dense_Layer(int64(nHidden), int64(nOutputs))

	return &Model{
		N_Input:       nInput,
		N_Hidden:      nHidden,
		N_Outputs:     nOutputs,
		HiddenLayer:   hiddenLayer,
		OutputLayer:   outputLayer,
		Learning_Rate: learningRate,
	}
}

func (net *Model) Train(X, y, X_test, y_test *tensor.Tensor, epochs int) {
	fmt.Println("Training the Model")

	for epoch := 1; epoch <= epochs; epoch++ {
		// Forward pass through the hidden layer
		net.HiddenLayer.Forward(X)

		// Forward pass through the output layer
		net.OutputLayer.Forward(net.HiddenLayer.Outputs)

		// Here you would calculate the loss and perform a backward pass and update weights
		// This part is omitted for brevity

		// Example of accessing the output of the network
		output := net.OutputLayer.Outputs
		fmt.Printf("Epoch %d: Output\n", epoch, output.Data)
	}
}

// type Layer interface {
// 	Forward(inputs *tensor.Tensor, training bool)
// 	Backward(dvalues *tensor.Tensor)
// 	SetPrev(layer Layer)
// 	SetNext(layer Layer)
// 	// ... other methods required by Layer
// }
