// nn_2 is a go implementation of Andrej Karpathy's 'Building Micrograd'
package main

import (
	"fmt"
	"math/rand"
	"time"
)

type Neuron struct {
	InputsN int
	Weights []float64
	Biases  []float64
}

func NewNeuron(size int) *Neuron {
	rand.Seed(time.Now().UnixNano())

	weights := make([]float64, size)

	biases := make([]float64, size)

	for i := range weights {
		weights[i] = rand.Float64()*2 - 1
		biases[i] = rand.Float64()*2 - 1
	}
	neur := &Neuron{
		InputsN: size,
		Weights: weights,
		Biases:  biases,
	}
	return neur
}
func main() {

	neuron := NewNeuron(2)
	fmt.Println(neuron)
}
