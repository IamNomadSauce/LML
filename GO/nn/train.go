package nn

import (
	"fmt"
	_ "fmt"
	"go_nn/tensor"
	_ "math"
)

type Model struct {
	InputLayer      LayerDense
	InputActivation *Activation_ReLU
	// DropoutLayer        *LayerDropout // Placeholder for dropout layer.
	OutputLayer      LayerDense
	OutputActivation *ActivationSoftmax // Corrected type for softmax activation
	Learning_Rate    float64
	Loss             LossFunction
}

func NewModel(nInput, nHidden, nOutputs int, learningRate float64) *Model {
	inputLayer := New_Dense_Layer(int64(nInput), int64(nHidden))
	inputActivation := New_ReLU_Activation()
	// dropoutLayer := New_Dropout_Layer() // Placeholder, initialize with appropriate parameters
	outputLayer := New_Dense_Layer(int64(nHidden), int64(nOutputs))
	outputActivation := New_Softmax_Activation() // Corrected function name for softmax activation

	return &Model{
		InputLayer:      inputLayer,
		InputActivation: inputActivation,
		// DropoutLayer:        dropoutLayer,
		OutputLayer:      outputLayer,
		OutputActivation: outputActivation,
		Learning_Rate:    learningRate,
	}
}

func (m *Model) Train(X, y, X_test, y_test *tensor.Tensor, epochs int) {
	fmt.Println("Training the Model")
	for epoch := 1; epoch <= epochs; epoch++ {
		// Forward pass through each layer and activation
		m.InputLayer.Forward(X)
		m.InputActivation.Forward(m.InputLayer.Outputs, true)
		// Apply dropout here
		// m.DropoutLayer.Forward(m.InputActivation.Output, true) // Placeholder for dropout forward pass
		// m.OutputLayer.Forward(m.DropoutLayer.Output) // Assuming dropout layer has an Output field
		m.OutputActivation.Forward(m.InputActivation.Output, true) // Assuming softmax activation has a Forward method
		fmt.Println("Output Activation", m.OutputActivation.Output)
		// Calculate loss (not shown here)
		// Backward pass and optimization (not shown here)

		fmt.Printf("Epoch %d: Training...\n", epoch)
	}
}

// type Layer interface {
// 	Forward(inputs *tensor.Tensor, training bool)
// 	Backward(dvalues *tensor.Tensor)
// 	SetPrev(layer Layer)
// 	SetNext(layer Layer)
// 	// ... other methods required by Layer
// }
