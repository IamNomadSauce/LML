package nn

import (
	_ "fmt"
	_ "go_nn/tensor"
	"math"
)

// Loss represents the common loss struct
type Loss struct {
	trainableLayers []Layer
}

// NewLoss creates a new Loss instance
func NewLoss() *Loss {
	return &Loss{}
}

// RememberTrainableLayers sets the trainable layers for the loss calculation
func (l *Loss) RememberTrainableLayers(trainableLayers []Layer) {
	l.trainableLayers = trainableLayers
}

// RegularizationLoss calculates the regularization loss
func (l *Loss) RegularizationLoss() float64 {
	regularizationLoss := 0.0

	// Calculate regularization loss
	for _, layer := range l.trainableLayers {
		// L1 regularization - weights
		if layer.GetWeightRegularizerL1() > 0 {
			for _, row := range layer.GetWeights().Data {
				for _, weight := range row {
					regularizationLoss += layer.GetWeightRegularizerL1() * math.Abs(weight)
				}
			}
		}

		// L2 regularization - weights
		if layer.GetWeightRegularizerL2() > 0 {
			for _, row := range layer.GetWeights().Data {
				for _, weight := range row {
					regularizationLoss += layer.GetWeightRegularizerL2() * weight * weight
				}
			}
		}

		// L1 regularization - biases
		if layer.GetBiasRegularizerL1() > 0 {
			for _, bias := range layer.GetBiases().Data[0] {
				regularizationLoss += layer.GetBiasRegularizerL1() * math.Abs(bias)
			}
		}

		// L2 regularization - biases
		if layer.GetBiasRegularizerL2() > 0 {
			for _, bias := range layer.GetBiases().Data[0] {
				regularizationLoss += layer.GetBiasRegularizerL2() * bias * bias
			}
		}
	}

	return regularizationLoss
}

// Calculate calculates the data and regularization losses
// func (l *Loss) Calculate(output, y *tensor.Tensor, includeRegularization bool) (float64, float64) {
// 	// Calculate sample losses
// 	sampleLosses := l.Forward(output, y) // Assuming Forward is implemented elsewhere

// 	// Calculate mean loss
// 	dataLoss := mean(sampleLosses) // Assuming mean is implemented elsewhere

// 	// If just data loss - return it
// 	if !includeRegularization {
// 		return dataLoss, 0
// 	}

// 	// Return the data and regularization losses
// 	return dataLoss, l.RegularizationLoss()
// }
