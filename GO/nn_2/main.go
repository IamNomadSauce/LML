// nn_2 is a go implementation of Andrej Karpathy's 'Building Micrograd'
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Neuron struct {
	InputsN int
	Weights []float64
	Biases  float64
	Output  float64
}

func NewNeuron(nInputs int, inputs []float64) *Neuron {
	fmt.Println("New Neuron", nInputs, "inputs\n")
	rand.Seed(time.Now().UnixNano())

	weights := make([]float64, nInputs)

	bias := rand.Float64()*2 - 1
	out := 0.0
	sum := 0.0
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1

		// sum(w * x) (Dot Product)
		wxb := weights[i] * inputs[i]
		sum += wxb
		// out := (weights[i] * inputs[i]) + biases[i]
		// fmt.Println("w*x+b: \n", weights, "\n", biases)
	}
	// w.x + b
	out = sum + bias
	act := TanH(out)
	neur := &Neuron{
		InputsN: nInputs,
		Weights: weights,
		Biases:  bias,
		Output:  act.Output,
	}
	return neur
}

type Activation struct {
	input  float64
	Output float64
}

// TanH activation
func TanH(x float64) *Activation {
	out := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	return &Activation{input: x, Output: out}
}

type Layer struct {
	in     int
	out    int
	output []float64
}

func NewLayer(nIn, nOut int, inputs []float64) *Layer {

	neurons := make([]float64, nOut)
	for i := range neurons {
		neurons[i] = NewNeuron(nIn, inputs).Output
	}
	fmt.Println("New Layer\n", neurons)

	for i := range neurons {
		fmt.Println("Output", i, neurons[i])
	}
	return &Layer{in: nIn, out: nOut, output: neurons}
}

// func NewLayer(nIn, nOut int) {
// 	neurons := NewNeuron(2)
// 	fmt.Println(neurons)
// }
func main() {

	inputs := []float64{
		2.0,
		3.0,
		// 2.1,
	}

	layer := NewLayer(len(inputs), 3, inputs)
	fmt.Println("Layer:", layer)
}
