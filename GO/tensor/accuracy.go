package tensor

import (
	"math"
)

// Accuracy represents the common accuracy struct
type Accuracy struct{}

// Compare compares predictions to the ground truth values for regression data
func (a *AccuracyRegression) Compare(predictions []float64, y []float64) []bool {
	comparisons := make([]bool, len(predictions))
	for i, prediction := range predictions {
		comparisons[i] = math.Abs(prediction-y[i]) < a.Precision
	}
	return comparisons
}

// Calculate calculates the accuracy given predictions and ground truth values
// func (a *Accuracy) Calculate(predictions, y []int) float64 {
// 	comparisons := a.Compare(predictions, y)
// 	sum := 0
// 	for _, match := range comparisons {
// 		if match {
// 			sum++
// 		}
// 	}
// 	accuracy := float64(sum) / float64(len(comparisons))
// 	return accuracy
// }

// Compare should be implemented by the specific accuracy struct that embeds Accuracy
// It should compare predictions to the ground truth values and return a slice of bool

// AccuracyCategorical represents accuracy calculation for classification model
type AccuracyCategorical struct {
	Accuracy
	Binary bool
}

// NewAccuracyCategorical creates a new instance of AccuracyCategorical
func NewAccuracyCategorical(binary bool) *AccuracyCategorical {
	return &AccuracyCategorical{Binary: binary}
}

// Compare compares predictions to the ground truth values for categorical data
func (a *AccuracyCategorical) Compare(predictions, y []int) []bool {
	comparisons := make([]bool, len(predictions))
	for i, prediction := range predictions {
		comparisons[i] = prediction == y[i]
	}
	return comparisons
}

// AccuracyRegression represents accuracy calculation for regression model
type AccuracyRegression struct {
	Accuracy
	Precision float64
}

// NewAccuracyRegression creates a new instance of AccuracyRegression
func NewAccuracyRegression() *AccuracyRegression {
	return &AccuracyRegression{}
}

// Init calculates precision value based on passed-in ground truth values
func (a *AccuracyRegression) Init(y []float64, reinit bool) {
	if a.Precision == 0 || reinit {
		var sum float64
		for _, value := range y {
			sum += value
		}
		mean := sum / float64(len(y))
		var varianceSum float64
		for _, value := range y {
			varianceSum += (value - mean) * (value - mean)
		}
		a.Precision = math.Sqrt(varianceSum/float64(len(y))) / 250
	}
}
