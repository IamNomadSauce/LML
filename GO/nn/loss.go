package nn

import (
	_ "fmt"
	"go_nn/tensor"
	"math"
)

// LossFunction represents a loss function
type LossFunction interface {
	Calculate(output, y *tensor.Tensor) float64
}

// LossCategoricalCrossentropy represents categorical cross-entropy loss
type LossCategoricalCrossentropy struct{}

// Forward performs the forward pass
func (l *LossCategoricalCrossentropy) Forward(yPred, yTrue *tensor.Tensor) []float64 {
	samples := yPred.Rows
	negativeLogLikelihoods := make([]float64, samples)

	for i := 0; i < samples; i++ {
		// Clip data to prevent division by 0
		for j := range yPred.Data[i] {
			yPred.Data[i][j] = math.Max(1e-7, math.Min(1-1e-7, yPred.Data[i][j]))
		}

		var correctConfidences float64
		if yTrue.Cols == 1 {
			correctConfidences = yPred.Data[i][int(yTrue.Data[i][0])]
		} else {
			for j := range yPred.Data[i] {
				correctConfidences += yPred.Data[i][j] * yTrue.Data[i][j]
			}
		}

		negativeLogLikelihoods[i] = -math.Log(correctConfidences)
	}

	return negativeLogLikelihoods
}

// Backward performs the backward pass
func (l *LossCategoricalCrossentropy) Backward(dvalues, yTrue *tensor.Tensor) *tensor.Tensor {
	samples := dvalues.Rows
	labels := dvalues.Cols

	if yTrue.Cols == 1 {
		yTrueOneHot := tensor.NewTensor(make([][]float64, samples))
		for i := range yTrueOneHot.Data {
			yTrueOneHot.Data[i] = make([]float64, labels)
			yTrueOneHot.Data[i][int(yTrue.Data[i][0])] = 1
		}
		yTrue = yTrueOneHot
	}

	dinputs := tensor.NewTensor(make([][]float64, samples))
	for i := range dvalues.Data {
		for j := range dvalues.Data[i] {
			dinputs.Data[i][j] = -yTrue.Data[i][j] / dvalues.Data[i][j]
			dinputs.Data[i][j] /= float64(samples)
		}
	}

	return dinputs
}

// Calculate calculates the data loss
func (l *LossCategoricalCrossentropy) Calculate(output, y *tensor.Tensor) float64 {
	// Implementation of categorical cross-entropy loss calculation
	// Placeholder implementation - replace with actual logic
	return 0.0
}

// -------------------------------

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
	samples := len(dvalues.Data)

	// If labels are one-hot encoded, turn them into discrete values
	if yTrue.Cols > 1 {
		yTrue = yTrue.Argmax(1)
	}

	// Copy so we can safely modify
	a.dinputs = dvalues.Copy()
	// Calculate gradient
	for i, label := range yTrue.Data {
		a.dinputs.Data[i][int(label[0])] -= 1
	}

	// Normalize gradient
	for i := range a.dinputs.Data {
		for j := range a.dinputs.Data[i] {
			a.dinputs.Data[i][j] /= float64(samples)
		}
	}
}

// ---------------------------------------------------------

// CalculateRegularizationLoss calculates the regularization loss for all trainable layers
func CalculateRegularizationLoss(trainableLayers []TrainableLayer) float64 {
	regularizationLoss := 0.0
	for _, layer := range trainableLayers {
		// L1 regularization - weights
		if reg := layer.GetWeightRegularizerL1(); reg > 0 {
			regularizationLoss += reg * sumAbs(layer.GetWeights())
		}
		// L2 regularization - weights
		if reg := layer.GetWeightRegularizerL2(); reg > 0 {
			regularizationLoss += reg * sumSquares(layer.GetWeights())
		}
		// L1 regularization - biases
		if reg := layer.GetBiasRegularizerL1(); reg > 0 {
			regularizationLoss += reg * sumAbs(layer.GetBiases())
		}
		// L2 regularization - biases
		if reg := layer.GetBiasRegularizerL2(); reg > 0 {
			regularizationLoss += reg * sumSquares(layer.GetBiases())
		}
	}
	return regularizationLoss
}

// sumAbs calculates the sum of absolute values in a tensor
func sumAbs(t *tensor.Tensor) float64 {
	sum := 0.0
	for _, row := range t.Data {
		for _, val := range row {
			sum += math.Abs(val)
		}
	}
	return sum
}

// sumSquares calculates the sum of squares in a tensor
func sumSquares(t *tensor.Tensor) float64 {
	sum := 0.0
	for _, row := range t.Data {
		for _, val := range row {
			sum += val * val
		}
	}
	return sum
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
