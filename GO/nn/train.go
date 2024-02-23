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
	OutputLayer     *LinearLayer
	Learning_Rate   float64
	Loss            *BCEWithLogitsLoss // Add the Loss field
	Optimizer       *AdamOptimizer
}

func NewModel(nInput, nHidden, nOutputs int, learningRate float64) *Model {
	inputLayer := New_Dense_Layer(int64(nInput), int64(nHidden))
	inputActivation := New_ReLU_Activation()
	linearLayer := NewLinearLayer(nHidden, nOutputs)

	// Initialize the BCEWithLogitsLoss for the model
	loss := &BCEWithLogitsLoss{} // Assuming predictions and targets will be set during training

	// Initialize the AdamOptimizer
	optimizer := NewAdamOptimizer(learningRate, 0.9, 0.999, 1e-8)

	return &Model{
		InputLayer:      inputLayer,
		InputActivation: inputActivation,
		OutputLayer:     linearLayer,
		Learning_Rate:   learningRate,
		Loss:            loss,
		Optimizer:       optimizer,
	}
}

func (m *Model) Train(X, y, X_test, y_test *tensor.Tensor, epochs int) {
	fmt.Println("\nTraining the Model", X.Shape().Data)

	for epoch := 1; epoch <= epochs; epoch++ {
		fmt.Println("\nEpoch", epoch)

		m.Optimizer.ZeroGrad()
		// Forward pass through layers
		m.InputLayer.Forward(X)
		m.InputActivation.Forward(m.InputLayer.Outputs, true)
		output := m.OutputLayer.Forward(m.InputLayer.Outputs)

		// Compute loss
		loss, err := m.Loss.Forward(output, y) // Use the Forward method of BCEWithLogitsLoss
		if err != nil {
			fmt.Println("Error computing loss:", err)
			return
		}

		// Compute gradients
		dLoss := m.Loss.Backward(output, y)
		// Check if dLoss is not nil before calling Update
		if dLoss != nil {
			fmt.Println("Proper Tensor Gradient")
			m.Optimizer.Update(m.OutputLayer.Weights, dLoss)
			// Note: You should also calculate and pass the correct gradients for biases
			// m.Optimizer.Update(m.OutputLayer.Biases, dBiases)
		} else {
			fmt.Println("Error: dLoss is nil")
			return
		}
		// dLoss, err := lossGradient(output, y)
		// m.OutputLayer.Backward(dLoss)
		// if err != nil {
		// 	fmt.Println("Error computing gradients:", err)
		// 	return
		// }

		// Update weights and biases using the optimizer
		m.Optimizer.Update(m.OutputLayer.Weights, m.OutputLayer.DWeights)
		m.Optimizer.Update(m.OutputLayer.Biases, m.OutputLayer.DBiases)

		fmt.Printf("Epoch %d: Training... Loss: %f\n", epoch, loss, dLoss)
	}
}
