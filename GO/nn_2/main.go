package main

import (
	"math"

	_ "gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NN struct {
	config  NN_Config
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

type NN_Config struct {
	inputN       int
	outputN      int
	hiddenN      int
	epochs       int
	learningRate float64
}

func New_NN(config NN_Config) *NN {
	return &NN{config: config}
}

// Sigmoid Activation
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}
func main() {

}
