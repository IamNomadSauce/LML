package nn

import (
	_ "fmt"
	"go_nn/tensor"
	"math"
)

// Loss represents the common loss struct
type Loss struct {
	trainableLayers []Layer
}

// LossFunction represents a loss function
// TODO May need to add 'forward'
type LossFunction interface {
	Calculate(output, y *tensor.Tensor, includeRegularization bool) (float64, float64)
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

// LossCategoricalCrossentropy represents categorical cross-entropy loss
type LossCategoricalCrossentropy struct {
	dinputs *tensor.Tensor
}

// Forward performs the forward pass
func (l *LossCategoricalCrossentropy) Forward(yPred, yTrue *tensor.Tensor) []float64 {
	samples := yPred.Rows
	negativeLogLikelihoods := make([]float64, samples)

	for i := 0; i < samples; i++ {
		// Clip data to prevent division by 0
		for j := range yPred.Data[i] {
			yPred.Data[i][j] = math.Max(1e-7, math.Min(1-1e-7, yPred.Data[i][j]))
		}

		// Probabilities for target values
		if yTrue.Cols == 1 {
			correctConfidences := yPred.Data[i][int(yTrue.Data[i][0])]
			negativeLogLikelihoods[i] = -math.Log(correctConfidences)
		} else {
			correctConfidences := 0.0
			for j := range yPred.Data[i] {
				correctConfidences += yPred.Data[i][j] * yTrue.Data[i][j]
			}
			negativeLogLikelihoods[i] = -math.Log(correctConfidences)
		}
	}

	return negativeLogLikelihoods
}

// Backward performs the backward pass
func (l *LossCategoricalCrossentropy) Backward(dvalues, yTrue *tensor.Tensor) {
	samples := len(dvalues.Data)
	labels := len(dvalues.Data[0])

	if yTrue.Cols == 1 {
		yTrueOneHot := make([][]float64, samples)
		for i := range yTrueOneHot {
			yTrueOneHot[i] = make([]float64, labels)
			yTrueOneHot[i][int(yTrue.Data[i][0])] = 1
		}
		yTrue.Data = yTrueOneHot
	}

	l.dinputs = tensor.NewTensor(make([][]float64, samples))
	for i := range dvalues.Data {
		for j := range dvalues.Data[i] {
			l.dinputs.Data[i][j] = -yTrue.Data[i][j] / dvalues.Data[i][j]
			l.dinputs.Data[i][j] = l.dinputs.Data[i][j] / float64(samples)
		}
	}
}

// -----------------
// ActivationSoftmaxLossCategoricalCrossentropy represents a combined softmax activation
// and categorical cross-entropy loss for faster backward step
type ActivationSoftmaxLossCategoricalCrossentropy struct {
	dinputs *tensor.Tensor
}

// NewActivationSoftmaxLossCategoricalCrossentropy creates a new instance
func NewActivationSoftmaxLossCategoricalCrossentropy() *ActivationSoftmaxLossCategoricalCrossentropy {
	return &ActivationSoftmaxLossCategoricalCrossentropy{}
}

// Backward performs the backward pass
func (a *ActivationSoftmaxLossCategoricalCrossentropy) Backward(dvalues, yTrue *tensor.Tensor) {
	// Number of samples
	samples := len(dvalues.Data)
	// If labels are one-hot encoded, turn them into discrete values
	if yTrue.Cols > 1 {
		yTrue = yTrue.Argmax(1)
	}

	// Copy so we can safely modify
	a.dinputs = dvalues.Copy()
	// Calculate gradient
	for i, label := range yTrue.Data {
		labelIndex := int(label[0]) // Convert label[0] from float64 to int
		a.dinputs.Data[i][labelIndex] -= 1
	}
	// Normalize gradient
	for i := range a.dinputs.Data {
		for j := range a.dinputs.Data[i] {
			a.dinputs.Data[i][j] /= float64(samples)
		}
	}
}

// Calculate calculates the data and regularization losses
// func (l *Loss) Calculate(output, y *tensor.Tensor, includeRegularization bool, calculateDataLoss func(*tensor.Tensor, *tensor.Tensor) []float64) (float64, float64) {
// 	// Calculate sample losses using the provided function
// 	sampleLosses := calculateDataLoss(output, y)

// 	// Calculate mean loss
// 	dataLoss := mean(sampleLosses) // Assuming mean is implemented elsewhere

// 	// If just data loss - return it
// 	if !includeRegularization {
// 		return dataLoss, 0
// 	}

// 	// Return the data and regularization losses
// 	return dataLoss, l.RegularizationLoss()
// }
