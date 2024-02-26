package main

import (
	"fmt"
	"go_nn/dl"
	_ "go_nn/dl"
	"go_nn/nn"
	_ "go_nn/nn"
	"go_nn/tensor"
	"math/rand"
	"time"

	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {

	X, y, X_test, y_test := dl.GenerateUpdownData(1000)
	X_tensor, y_tensor := tensor.NewTensor(X), tensor.NewTensor(y)
	X_test_tensor, y_test_tensor := tensor.NewTensor(X_test), tensor.NewTensor(y_test)

	// fmt.Println(X_tensor.Data, y_tensor.Data, X_test_tensor.Data, y_test_tensor.Data)

	model := nn.NewModel(2, 10, 1, 5e-10)
	fmt.Println("OutputsN:", model.InputLayer.Outputs)

	model.Train(X_tensor, y_tensor, X_test_tensor, y_test_tensor, 10001)

	// Create a new plot
	p := plot.New()
	// if err != nil {
	// 	log.Fatalf("Could not create plot: %v", err)
	// }

	p.Title.Text = "Training Loss Over Time"
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Loss"

	// Create a line plotter with data
	pts := make(plotter.XYs, len(model.LossValues))
	for i := range pts {
		pts[i].X = float64(i + 1) // Epochs start at 1
		pts[i].Y = model.LossValues[i]
	}

	line, err := plotter.NewLine(pts)
	if err != nil {
		log.Fatalf("Could not create line plotter: %v", err)
	}
	p.Add(line)

	// Save the plot to a PNG file
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "loss_plot.png"); err != nil {
		log.Fatalf("Could not save plot: %v", err)
	}

}

// generateRandomTuples generates a slice of tuples, each with two random floats in the range of -10 to 10.
func gen_data(samples int) *tensor.Tensor { // Note the use of tensor.Tensor
	// Seed the random number generator using the current time
	rand.Seed(time.Now().UnixNano())

	// Create a 2D slice to hold the random floats
	data := make([][]float64, samples)
	for i := range data {
		data[i] = make([]float64, 2) // Each sample has 2 floats
		for j := range data[i] {
			// Generate a random number between 0.0 and 1.0
			randomFloat := rand.Float64()
			// Adjust the range to be between -10 and 10
			randomFloat = randomFloat*2 - 1
			data[i][j] = randomFloat
		}
	}

	// Create and return a new Tensor with the generated data
	return tensor.NewTensor(data) // Use the NewTensor function from the tensor package
}

// X := [][]float64{
// 	{0., 0.},
// 	{0.10738789, 0.02852226},
// 	{0.09263825, -0.20199226},
// 	{-0.32224888, -0.08524539},
// 	{-0.3495118, 0.27454028},
// 	{-0.52100587, 0.19285966},
// 	{0.5045865, 0.43570277},
// 	{0.76882404, 0.11767714},
// 	{0.49269393, -0.73984873},
// 	{-0.70364994, -0.71054685},
// 	{-0., -0.},
// 	{-0.07394107, 0.08293611},
// 	{0.00808054, 0.22207525},
// 	{0.24548167, 0.22549914},
// 	{0.38364738, -0.22437814},
// 	{-0.00801609, -0.5554977},
// 	{-0.66060567, 0.08969161},
// 	{-0.7174548, 0.30032802},
// 	{0.17299275, 0.87189275},
// 	{0.66193414, 0.74956197},
// }
