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
	Output  float64
}

func NewNeuron(inputs []float64) *Neuron {
	rand.Seed(time.Now().UnixNano())

	weights := make([]float64, len(inputs))

	biases := make([]float64, len(inputs))

	for i := range weights {
		w := weights[i]
		b := biases[i]
		x := inputs[i]

		w = rand.Float64()*2 - 1
		b = rand.Float64()*2 - 1
		fmt.Println("w*x+b: ", (w*x)+b)
	}
	neur := &Neuron{
		InputsN: len(inputs),
		Weights: weights,
		Biases:  biases,
	}
	return neur
}

type Layer struct {
	in  int
	out int
}

func main() {

	inputs := []float64{
		2.0,
		4.1,
		2.1,
	}

	neuron := NewNeuron(inputs)
	fmt.Println(neuron)
}
