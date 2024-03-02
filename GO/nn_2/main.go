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
	DotP    float64 // w * x + b
	Output  float64
}

func NewNeuron(nInputs int, inputs []float64) *Neuron {
	// fmt.Println("New Neuron", nInputs, "inputs\n")
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
		DotP:    out,
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
	in      int
	out     int
	Neurons []Neuron
	output  []float64
}

func NewLayer(nIn, nOut int, inputs []float64) *Layer {
	// fmt.Println("New Layyer!", nIn, nOut)

	neur := make([]Neuron, nOut)
	neurons := make([]float64, nOut)
	for i := range neurons {
		neurons[i] = NewNeuron(nIn, inputs).Output
		neur[i] = *NewNeuron(nIn, inputs)
	}
	// fmt.Println("New Layer\n", len(neurons))

	// for i := range neurons {
	// 	fmt.Println("Output", i, neurons[i])
	// }
	return &Layer{in: nIn, out: nOut, Neurons: neur, output: neurons}
}

func (l *Layer) Parameters() []float64 {
	params := []float64{}
	for _, neuron := range l.Neurons {
		fmt.Println("Parameters:Neurons", neuron)
		// params = append(params, neuron.DotP)
		params = append(params, neuron.DotP)
	}
	return params
}

type MLP struct {
	In     int
	nOuts  int
	Layers []Layer
	Output float64
}

func NewMLP(nIn int, layers []int, inputs []float64) *MLP {

	sz := append([]int{nIn}, layers...)
	fmt.Println("NewMLP", sz)
	nInputs := inputs
	lyrs := []Layer{}
	for i := 0; i < len(sz)-1; i++ {
		layer := NewLayer(sz[i], sz[i+1], nInputs)
		lyrs = append(lyrs, *layer)
		nInputs = layer.output
		// fmt.Println("Layer", layer.in, layer.out, layer.output)
	}
	// for i := range layers {
	output := lyrs[len(layers)-1].output[0]
	// fmt.Println("Output", output)
	// 	fmt.Println("Layer", layers[i], nIn)

	// 	output := NewLayer(layers[i], layers[i+1], inputs)
	// 	fmt.Println("Output", output)
	// }
	return &MLP{
		In:     nIn,
		nOuts:  len(nInputs),
		Layers: lyrs,
		Output: output,
	}
}

func (m *MLP) Parameters() []float64 {
	var params []float64
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func main() {

	// inputs := []float64{
	// 	2.0,
	// 	3.0,
	// 	-1.0,
	// }

	// layer := NewLayer(len(inputs), 3, inputs)
	// fmt.Println("Layer:", layer)
	layers := []int{
		4,
		4,
		1,
	}
	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	for i := range xs {
		fmt.Println(i)
		mlp := NewMLP(len(xs[0]), layers, xs[i])
		fmt.Println("MLP", mlp.Output)
		params := mlp.Parameters()
		fmt.Println("Parameters", params)
	}

}
