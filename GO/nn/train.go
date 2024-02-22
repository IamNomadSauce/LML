package nn

import (
	"fmt"
	_ "fmt"
	"go_nn/tensor"
	_ "math"
)

type Model struct {
	InputLayer       LayerDense
	InputActivation  *Activation_ReLU
	OutputLayer      *LinearLayer
	OutputActivation *ActivationSoftmax
	Learning_Rate    float64
	Loss             LossFunction
}

func NewModel(nInput, nHidden, nOutputs int, learningRate float64) *Model {
	// Dense Layer
	inputLayer := New_Dense_Layer(int64(nInput), int64(nHidden))
	// Relu Activation
	inputActivation := New_ReLU_Activation()
	// Linear output
	linearLayer := NewLinearLayer(nHidden, nOutputs, inputActivation.Output) // Create the linear layer
	fmt.Println(linearLayer)

	// BCEwLogits
	// loss_function :=

	return &Model{
		InputLayer:      inputLayer,
		InputActivation: inputActivation,
		OutputLayer:     linearLayer,
		Learning_Rate:   learningRate,
		// Loss:            LossFunction,
	}
}

func (m *Model) Train(X, y, X_test, y_test *tensor.Tensor, epochs int) {
	fmt.Println("\nTraining the Model\n")
	for epoch := 1; epoch <= epochs; epoch++ {

		// fmt.Println("FORWARD-Inputs Layer", X.Rows, X.Cols)
		// Forward pass through input layer and activation
		m.InputLayer.Forward(X)
		// fmt.Println("FORWARD-DenseLayer", m.InputLayer.Outputs.Rows, m.InputLayer.Outputs.Cols)

		// Forward pass through ReLU activation layer
		m.InputActivation.Forward(m.InputLayer.Outputs, true)
		// fmt.Println("FORWARD ReLU Forward", m.InputActivation.Output.Rows, m.InputActivation.Output.Cols)

		// Forward pass through output layer (linear layer)
		output := m.OutputLayer.Forward(m.InputLayer.Outputs)
		// fmt.Println("Model Train Linear Layer Forward", output.Rows, output.Cols)
		// for _, row := range output.Data {
		// 	fmt.Println("Row", row)
		// }

		// Loss
		loss, err := BCEWithLogitsLoss(output, y)

		if err != nil {
			fmt.Println("BCEwL ")
		}

		fmt.Println("Outputs:", output.Data, "\ny_train", y.Data, "\n", "LOSS", loss, "\n")

		// Apply softmax activation
		// m.OutputActivation.Forward(output, true)

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
