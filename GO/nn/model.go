package nn

// import (
// 	"fmt"
// 	"go_nn/tensor"
// )

// // Forward performs the forward pass
// func (l *LayerInput) Forward(inputs *tensor.Tensor, training bool) {
// 	l.Output = inputs
// }

// // Backward performs the backward pass (dummy implementation for LayerInput)
// func (l *LayerInput) Backward(dvalues *tensor.Tensor) {
// 	// Input layer does not have a backward pass, so this can be a no-op
// }

// // SetPrev sets the previous layer (dummy implementation for LayerInput)
// func (l *LayerInput) SetPrev(layer Layer) {
// 	// Input layer does not have a previous layer, so this can be a no-op
// }

// // SetNext sets the next layer (dummy implementation for LayerInput)
// func (l *LayerInput) SetNext(layer Layer) {
// 	// Input layer does not directly set the next layer, so this can be a no-op
// }

// // Model represents a neural network model
// type Model struct {
// 	Layers                  []Layer
// 	SoftmaxClassifierOutput *ActivationSoftmaxLossCategoricalCrossentropy
// 	Loss                    LossFunction
// 	Optimizer               Optimizer
// 	Accuracy                AccuracyCalculator
// 	InputLayer              *LayerInput
// 	TrainableLayers         []Layer
// 	OutputLayerActivation   Layer
// }

// // NewModel creates a new instance of Model
// func NewModel() *Model {
// 	return &Model{}
// }

// // Add adds a layer to the model
// func (m *Model) Add(layer Layer) {
// 	m.Layers = append(m.Layers, layer)
// }

// // Set sets the loss, optimizer, and accuracy for the model
// func (m *Model) Set(loss LossFunction, optimizer Optimizer, accuracy AccuracyCalculator) {
// 	m.Loss = loss
// 	m.Optimizer = optimizer
// 	m.Accuracy = accuracy
// }

// // Finalize finalizes the model, preparing it for training
// func (m *Model) Finalize() {
// 	// Create and set the input layer.
// 	m.InputLayer = NewLayerInput()

// 	// Count all the objects.
// 	layerCount := len(m.Layers)

// 	// Initialize a list containing trainable layers.
// 	m.TrainableLayers = []Layer{} // Use Layer interface for the slice

// 	// Iterate the objects.
// 	for i := 0; i < layerCount; i++ {
// 		// Link layers.
// 		if i == 0 {
// 			m.Layers[i].SetPrev(m.InputLayer)
// 			if layerCount > 1 {
// 				m.Layers[i].SetNext(m.Layers[i+1])
// 			}
// 		} else if i < layerCount-1 {
// 			m.Layers[i].SetPrev(m.Layers[i-1])
// 			m.Layers[i].SetNext(m.Layers[i+1])
// 		} else {
// 			m.Layers[i].SetPrev(m.Layers[i-1])
// 			// The last layer's next object is not set to m.Loss since it's not a Layer.
// 			// Instead, directly reference the last layer as the output layer activation.
// 			m.OutputLayerActivation = m.Layers[i]
// 		}

// 		// If layer is trainable, add it to the list of trainable layers.
// 		if layer, ok := m.Layers[i].(TrainableLayer); ok {
// 			m.TrainableLayers = append(m.TrainableLayers, layer) // Append as Layer type
// 		}
// 	}

// 	// Correctly perform type assertions for the last layer and loss function.
// 	_, lastLayerIsSoftmax := m.Layers[layerCount-1].(*ActivationSoftmax)
// 	_, lossIsCategoricalCrossentropy := m.Loss.(*LossCategoricalCrossentropy)

// 	// If the last layer is a softmax classifier and the loss function is categorical cross-entropy,
// 	// create a combined softmax classifier output.
// 	if lastLayerIsSoftmax && lossIsCategoricalCrossentropy {
// 		m.SoftmaxClassifierOutput = NewActivationSoftmaxLossCategoricalCrossentropy()
// 	}
// }

// // Train trains the model on the provided data
// func (m *Model) Train(X, y *tensor.Tensor, epochs int, printEvery int, validationData *ValidationData) {
// 	fmt.Println("TRAIN MODEL", X, y)

// 	// Initialize accuracy object
// 	m.Accuracy.Init(y)

// 	// Main training loop
// 	for epoch := 1; epoch <= epochs; epoch++ {
// 		// Perform the forward pass
// 		output := m.Forward(X, true)

// 		// Calculate loss
// 		dataLoss := m.Loss.Calculate(output, y)
// 		regularizationLoss := CalculateRegularizationLoss(m.TrainableLayers)
// 		loss := dataLoss + regularizationLoss

// 		// Get predictions and calculate an accuracy
// 		predictions := m.OutputLayerActivation.Predictions(output)
// 		accuracy := m.Accuracy.Calculate(predictions, y)

// 		// Perform backward pass
// 		m.Backward(output, y)

// 		// Optimize (update parameters)
// 		m.Optimizer.PreUpdateParams()
// 		for _, layer := range m.TrainableLayers {
// 			m.Optimizer.UpdateParams(layer)
// 		}
// 		m.Optimizer.PostUpdateParams()

// 		// Print a summary
// 		if epoch%printEvery == 0 {
// 			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f (data_loss: %.3f, reg_loss: %.3f), lr: %.3f\n",
// 				epoch, accuracy, loss, dataLoss, regularizationLoss, m.Optimizer.GetCurrentLearningRate())
// 		}
// 	}

// 	// If there is validation data
// 	if validationData != nil {
// 		// Perform the forward pass
// 		output := m.Forward(validationData.X, false)

// 		// Calculate the loss
// 		loss := m.Loss.Calculate(output, validationData.y)

// 		// Get predictions and calculate an accuracy
// 		predictions := m.OutputLayerActivation.Predictions(output)
// 		accuracy := m.Accuracy.Calculate(predictions, validationData.y)

// 		// Print a summary
// 		fmt.Printf("validation, acc: %.3f, loss: %.3f\n", accuracy, loss)
// 	}
// }

// // Forward performs the forward pass
// func (m *Model) Forward(X *tensor.Tensor, training bool) *tensor.Tensor {
// 	// Call forward method on the input layer
// 	// this will set the output property that
// 	// the first layer in "prev" object is expecting
// 	m.InputLayer.Forward(X, training)

// 	// Call forward method of every object in a chain
// 	// Pass output of the previous object as a parameter
// 	var output *tensor.Tensor
// 	for _, layer := range m.Layers {
// 		layer.Forward(output, training)
// 		output = layer.Da() // Assuming GetOutput method exists
// 	}

// 	// "output" is now the last object from the list,
// 	// return its output
// 	return output
// }

// // Backward performs the backward pass
// func (m *Model) Backward(dvalues, yTrue *tensor.Tensor) {
// 	// If softmax classifier and categorical cross-entropy loss are combined
// 	if m.SoftmaxClassifierOutput != nil {
// 		// Call backward method on the combined activation/loss
// 		m.SoftmaxClassifierOutput.Backward(dvalues, yTrue)

// 		// Set dinputs in the last layer to the dinputs of the combined activation/loss
// 		lastLayer := m.Layers[len(m.Layers)-1].(TrainableLayer)
// 		lastLayer.SetDInputs(m.SoftmaxClassifierOutput.GetDInputs())

// 		// Call backward method going through all the objects but last
// 		// in reversed order passing dinputs as a parameter
// 		for i := len(m.Layers) - 2; i >= 0; i-- {
// 			layer := m.Layers[i].(TrainableLayer)
// 			nextLayer := m.Layers[i+1].(TrainableLayer)
// 			layer.Backward(nextLayer.GetDInputs())
// 		}
// 		return
// 	}

// 	// First call backward method on the loss
// 	// this will set dinputs property that the last layer will try to access shortly
// 	m.Loss.Backward(dvalues, yTrue)

// 	// Call backward method going through all the objects in reversed order
// 	// passing dinputs as a parameter
// 	for i := len(m.Layers) - 1; i >= 0; i-- {
// 		layer := m.Layers[i].(TrainableLayer)
// 		var nextDInputs *tensor.Tensor
// 		if i == len(m.Layers)-1 {
// 			nextDInputs = m.Loss.GetDInputs()
// 		} else {
// 			nextLayer := m.Layers[i+1].(TrainableLayer)
// 			nextDInputs = nextLayer.GetDInputs()
// 		}
// 		layer.Backward(nextDInputs)
// 	}
// }

// // ValidationData represents validation data
// type ValidationData struct {
// 	X *tensor.Tensor
// 	y *tensor.Tensor
// }

// // -------------------
// // TODO:::

// // AccuracyCalculator represents an accuracy calculator
// type AccuracyCalculator interface {
// 	Calculate(predictions, y []int) float64
// }
