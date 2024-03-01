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
	Dropout         *DropoutLayer // Add the Dropout field
	Learning_Rate   float64
	Loss            *BCEWithLogitsLoss // Add the Loss field
	Optimizer       *AdamOptimizer
	LossValues      []float64
}

// Parameters returns a slice of all parameter tensors in the model
func (m *Model) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{
		m.InputLayer.Weights,
		m.InputLayer.Biases,
		m.OutputLayer.Weights,
		m.OutputLayer.Biases,
	}
	return params
}

// GradientForParameter returns the gradient tensor corresponding to a given parameter tensor
func (m *Model) GradientForParameter(param *tensor.Tensor) *tensor.Tensor {
	var grad *tensor.Tensor
	switch param {
	case m.InputLayer.Weights:
		grad = m.InputLayer.DWeights
	case m.InputLayer.Biases:
		grad = m.InputLayer.DBiases
	case m.OutputLayer.Weights:
		grad = m.OutputLayer.DWeights
	case m.OutputLayer.Biases:
		grad = m.OutputLayer.DBiases
	}
	return grad
}

func NewModel(nInput, nHidden, nOutputs int, learningRate float64) *Model {
	inputLayer := New_Dense_Layer(int64(nInput), int64(nHidden))
	inputActivation := New_ReLU_Activation()
	linearLayer := NewLinearLayer(nHidden, nOutputs)

	// Initialize the BCEWithLogitsLoss for the model
	loss := &BCEWithLogitsLoss{} // Assuming predictions and targets will be set during training

	// Initialize the AdamOptimizer
	optimizer := NewAdamOptimizer(0.0001, 0.9, 0.999, 1e-8, 1e-5, 1e-5)

	dropout := NewDropoutLayer(0.5)
	return &Model{
		InputLayer:      inputLayer,
		InputActivation: inputActivation,
		OutputLayer:     linearLayer,
		Dropout:         dropout,
		Learning_Rate:   learningRate,
		Loss:            loss,
		Optimizer:       optimizer,
	}
}

func (m *Model) Train(X, y, X_test, y_test *tensor.Tensor, epochs int) {
	fmt.Println("\nTraining the Model", X.Shape().Data)

	m.LossValues = make([]float64, epochs)
	for epoch := 1; epoch <= epochs; epoch++ {

		m.Optimizer.ZeroGrad()

		// Forward pass through layers
		// fmt.Println("Do input-forward")
		m.InputLayer.Forward(X)
		// fmt.Println("Do input-activation-forward")
		m.InputActivation.Forward(m.InputLayer.Outputs, true)

		// Apply dropout after the activation
		// fmt.Println("Do Dropout-forward")
		m.Dropout.Forward(m.InputActivation.Output, true)
		output := m.OutputLayer.Forward(m.Dropout.Output) // Use the output from dropout layer

		// fmt.Println("Predictions", output.Data)

		// Compute loss
		loss, err := m.Loss.Forward(output, y)
		if err != nil {
			fmt.Println("Error computing loss:", err)
			return
		}
		fmt.Println("Compute Loss", loss)

		// Backward Pass
		dLoss := m.Loss.Backward(output, y)
		dOutput := m.OutputLayer.Backward(dLoss)
		dDropout := m.Dropout.Backward(dOutput)
		dInputActivation := m.InputActivation.Backward(dDropout)
		m.InputLayer.Backward(dInputActivation)

		m.Optimizer.Step(m)

		m.LossValues[epoch-1] = loss

		// fmt.Println()
		// // Compute gradients
		// dLoss := m.Loss.Backward(output, y)

		// // Backpropagate through the output layer to compute gradients for weights and biases
		// dOutput := m.OutputLayer.Backward(dLoss) // Backpropagate through the output layer

		// // Backpropagate through dropout
		// dDropout := m.Dropout.Backward(dOutput) // Backpropagate through the dropout layer

		// // Backpropagate through the input activation layer
		// dInputActivation := m.InputActivation.Backward(dDropout) // Backpropagate through the input activation layer

		// // Backpropagate through the input layer to compute gradients for weights and biases
		// m.InputLayer.Backward(dInputActivation) // Backpropagate through the input layer

		// // Update weights and biases using the optimizer
		// m.Optimizer.Step(m) // Perform the optimization step to update all weights and biases

		// m.LossValues[epoch-1] = loss // Store the loss value for this epoch

		// if epoch%100 == 0 {
		// 	fmt.Printf("Epoch %d: Training... Loss: %f\n", epoch, loss)
		// }
	}
}
