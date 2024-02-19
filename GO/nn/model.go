package nn

import "go_nn/tensor"

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

// Forward performs the forward pass
func (l *LayerInput) Forward(inputs *tensor.Tensor, training bool) {
	l.Output = inputs
}

// Backward performs the backward pass (dummy implementation for LayerInput)
func (l *LayerInput) Backward(dvalues *tensor.Tensor) {
	// Input layer does not have a backward pass, so this can be a no-op
}

// SetPrev sets the previous layer (dummy implementation for LayerInput)
func (l *LayerInput) SetPrev(layer Layer) {
	// Input layer does not have a previous layer, so this can be a no-op
}

// SetNext sets the next layer (dummy implementation for LayerInput)
func (l *LayerInput) SetNext(layer Layer) {
	// Input layer does not directly set the next layer, so this can be a no-op
}

// Model represents a neural network model
type Model struct {
	Layers                  []Layer
	SoftmaxClassifierOutput *ActivationSoftmaxLossCategoricalCrossentropy
	Loss                    LossFunction
	Optimizer               Optimizer
	Accuracy                AccuracyCalculator
	InputLayer              *LayerInput
	TrainableLayers         []Layer
	OutputLayerActivation   Layer
}

// NewModel creates a new instance of Model
func NewModel() *Model {
	return &Model{}
}

// Add adds a layer to the model
func (m *Model) Add(layer Layer) {
	m.Layers = append(m.Layers, layer)
}

// Set sets the loss, optimizer, and accuracy for the model
func (m *Model) Set(loss LossFunction, optimizer Optimizer, accuracy AccuracyCalculator) {
	m.Loss = loss
	m.Optimizer = optimizer
	m.Accuracy = accuracy
}

// Finalize finalizes the model, preparing it for training
func (m *Model) Finalize() {
	// Create and set the input layer.
	m.InputLayer = NewLayerInput()

	// Count all the objects.
	layerCount := len(m.Layers)

	// Initialize a list containing trainable layers.
	m.TrainableLayers = []Layer{} // Use Layer interface for the slice

	// Iterate the objects.
	for i := 0; i < layerCount; i++ {
		// Link layers.
		if i == 0 {
			m.Layers[i].SetPrev(m.InputLayer)
			if layerCount > 1 {
				m.Layers[i].SetNext(m.Layers[i+1])
			}
		} else if i < layerCount-1 {
			m.Layers[i].SetPrev(m.Layers[i-1])
			m.Layers[i].SetNext(m.Layers[i+1])
		} else {
			m.Layers[i].SetPrev(m.Layers[i-1])
			// The last layer's next object is not set to m.Loss since it's not a Layer.
			// Instead, directly reference the last layer as the output layer activation.
			m.OutputLayerActivation = m.Layers[i]
		}

		// If layer is trainable, add it to the list of trainable layers.
		if layer, ok := m.Layers[i].(TrainableLayer); ok {
			m.TrainableLayers = append(m.TrainableLayers, layer) // Append as Layer type
		}
	}

	// Correctly perform type assertions for the last layer and loss function.
	_, lastLayerIsSoftmax := m.Layers[layerCount-1].(*ActivationSoftmax)
	_, lossIsCategoricalCrossentropy := m.Loss.(*LossCategoricalCrossentropy)

	// If the last layer is a softmax classifier and the loss function is categorical cross-entropy,
	// create a combined softmax classifier output.
	if lastLayerIsSoftmax && lossIsCategoricalCrossentropy {
		m.SoftmaxClassifierOutput = NewActivationSoftmaxLossCategoricalCrossentropy()
	}
}

// -------------------
// TODO:::

// AccuracyCalculator represents an accuracy calculator
type AccuracyCalculator interface {
	Calculate(predictions, y []int) float64
}
