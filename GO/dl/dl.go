package dl

import (
	"math"
	"math/rand"
	"time"
)

// meanNormalize performs mean normalization on a slice of data.
func meanNormalize(data [][]float64, meanVals, rangeVals []float64) [][]float64 {
	normalized := make([][]float64, len(data))
	for i, v := range data {
		normalized[i] = make([]float64, len(v))
		for j, val := range v {
			normalized[i][j] = (val - meanVals[j]) / rangeVals[j]
		}
	}
	return normalized
}

// calculateMean calculates the mean of a slice of float64.
func calculateMean(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// calculateRange calculates the range (max - min) of a slice of float64.
func calculateRange(data []float64) float64 {
	minVal := data[0]
	maxVal := data[0]
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal - minVal
}

// definePivot translates the define_pivot function from Python to Go.
// This function needs to be implemented based on the logic provided in the Python code.
func definePivot(a, b, c float64) []float64 {
	// Implement the logic for defining the pivot here.
	// This is a placeholder function.
	return []float64{}
}

// generatePointLabels generates random points and labels for them.
func generatePointLabels(num int) ([][]float64, [][]float64, [][]float64, [][]float64) {
	rand.Seed(time.Now().UnixNano())
	inputs := make([][]float64, num)
	outputs := make([][]float64, num)
	for i := 0; i < num; i++ {
		a := float64(rand.Intn(100000) + 1)
		b := float64(rand.Intn(100000) + 1)
		c := float64(rand.Intn(100000) + 1)
		if b < a {
			c = b + rand.Float64()*(100000-b)
		} else {
			c = rand.Float64() * b
		}
		d := definePivot(a, b, c)
		inputs[i] = []float64{a, b, c}
		outputs[i] = d
	}

	// Calculate mean and range for normalization
	meanVals := make([]float64, 3)
	rangeVals := make([]float64, 3)
	for i := 0; i < 3; i++ {
		col := make([]float64, num)
		for j := 0; j < num; j++ {
			col[j] = inputs[j][i]
		}
		meanVals[i] = calculateMean(col)
		rangeVals[i] = calculateRange(col)
	}

	inputsNormalized := meanNormalize(inputs, meanVals, rangeVals)

	// Split the data into training and test sets
	splitIndex := int(math.Floor(float64(len(inputsNormalized)) * 0.8))
	XTraining := inputsNormalized[:splitIndex]
	XTest := inputsNormalized[splitIndex:]
	yTraining := outputs[:splitIndex]
	yTest := outputs[splitIndex:]

	return XTraining, yTraining, XTest, yTest
}

// generateUpdownData generates data points based on whether one number is less than another, normalizes the data, and splits it into training and test sets.
func GenerateUpdownData(num int) ([][]float64, [][]float64, [][]float64, [][]float64) {
	rand.Seed(time.Now().UnixNano())
	inputs := make([][]float64, num)
	outputs := make([][]float64, num)
	for i := 0; i < num; i++ {
		a := float64(rand.Intn(100000) + 1)
		b := float64(rand.Intn(100000) + 1)
		c := 0.0
		if a < b {
			c = 1.0
		}
		inputs[i] = []float64{a, b, c}
		outputs[i] = []float64{c}
	}

	// Calculate mean and range for normalization
	meanVals := make([]float64, 3)
	rangeVals := make([]float64, 3)
	for i := 0; i < 3; i++ {
		col := make([]float64, num)
		for j := 0; j < num; j++ {
			col[j] = inputs[j][i]
		}
		meanVals[i] = calculateMean(col)
		rangeVals[i] = calculateRange(col)
	}

	inputsNormalized := meanNormalize(inputs, meanVals, rangeVals)

	// Split the data into training and test sets
	splitIndex := int(math.Floor(float64(len(inputsNormalized)) * 0.8))
	XTraining := inputsNormalized[:splitIndex]
	XTest := inputsNormalized[splitIndex:]
	yTraining := outputs[:splitIndex]
	yTest := outputs[splitIndex:]

	return XTraining, yTraining, XTest, yTest
}
