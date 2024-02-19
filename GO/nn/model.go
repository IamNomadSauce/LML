package nn

import "go_nn/tensor"

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
	// Create and set the input layer
	m.InputLayer = NewLayerInput()

	// Count all the objects
	layerCount := len(m.Layers)

	// Initialize a list containing trainable layers
	m.TrainableLayers = []Layer{}

	// Iterate the objects
	for i := 0; i < layerCount; i++ {
		// If it's the first layer, the previous layer object is the input layer
		if i == 0 {
			m.Layers[i].SetPrev(m.InputLayer)
			m.Layers[i].SetNext(m.Layers[i+1])
		} else if i < layerCount-1 { // All layers except for the first and the last
			m.Layers[i].SetPrev(m.Layers[i-1])
			m.Layers[i].SetNext(m.Layers[i+1])
		} else { // The last layer - the next object is the loss
			m.Layers[i].SetPrev(m.Layers[i-1])
			m.Layers[i].SetNext(m.Loss)
			m.OutputLayerActivation = m.Layers[i]
		}

		// If layer is trainable, add it to the list of trainable layers
		if layer, ok := m.Layers[i].(TrainableLayer); ok {
			m.TrainableLayers = append(m.TrainableLayers, layer)
		}
	}

	// Update loss object with trainable layers
	m.Loss.RememberTrainableLayers(m.TrainableLayers)

	// If output activation is Softmax and loss function is Categorical Cross-Entropy
	// create an object of combined activation and loss function containing faster gradient calculation
	if _, ok := m.OutputLayerActivation.(*ActivationSoftmax); ok {
		if _, ok := m.Loss.(*LossCategoricalCrossentropy); ok {
			m.SoftmaxClassifierOutput = NewActivationSoftmaxLossCategoricalCrossentropy()
		}
	}
}

type TrainableLayer interface {
	Layer
	GetWeights() *tensor.Tensor
	GetBiases() *tensor.Tensor
	GetWeightRegularizerL1() float64
	GetWeightRegularizerL2() float64
	GetBiasRegularizerL1() float64
	GetBiasRegularizerL2() float64
}

// -------------------
// TODO:::

// AccuracyCalculator represents an accuracy calculator
type AccuracyCalculator interface {
	Calculate(predictions, y []int) float64
}
