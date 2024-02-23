package nn

import (
	"fmt"
	"go_nn/tensor"
	"math"
)

// AdamOptimizer represents the Adam optimization algorithm
type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	M            *tensor.Tensor
	V            *tensor.Tensor
	T            int
}

// NewAdamOptimizer creates a new instance of AdamOptimizer
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		M:            nil, // Initialized later based on parameter size
		V:            nil, // Initialized later based on parameter size
		T:            0,
	}
}

// Update updates the parameters based on gradients
func (opt *AdamOptimizer) Update(params, grads *tensor.Tensor) {
	fmt.Println("AdamUpdate\n", params.Data, "\n", grads.Data)
	if opt.M == nil || opt.V == nil {
		fmt.Println("Nil Nil")
		// Initialize M and V tensors with the same shape as params but filled with zeros
		m := tensor.NewZerosTensor(params.Rows, params.Cols)
		v := tensor.NewZerosTensor(params.Rows, params.Cols)
		fmt.Println("m", m.Shape().Data, "\nv", v.Shape().Data)
		opt.M = m
		opt.V = v
	}

	opt.T++ // Increment timestep

	for i := range params.Data {
		for j := range params.Data[i] {
			g := grads.Data[i][j]

			// Update biased first moment estimate
			opt.M.Data[i][j] = opt.Beta1*opt.M.Data[i][j] + (1-opt.Beta1)*g
			// Update biased second raw moment estimate
			opt.V.Data[i][j] = opt.Beta2*opt.V.Data[i][j] + (1-opt.Beta2)*(g*g)

			// Compute bias-corrected first moment estimate
			mHat := opt.M.Data[i][j] / (1 - math.Pow(opt.Beta1, float64(opt.T)))
			// Compute bias-corrected second raw moment estimate
			vHat := opt.V.Data[i][j] / (1 - math.Pow(opt.Beta2, float64(opt.T)))

			// Update parameters
			params.Data[i][j] -= opt.LearningRate * mHat / (math.Sqrt(vHat) + opt.Epsilon)
		}
	}
}

// ZeroGrad resets the gradients to zero
func (opt *AdamOptimizer) ZeroGrad() {
	if opt.M != nil && opt.V != nil {
		// Iterate over M and V tensors and set all values to zero
		for i := range opt.M.Data {
			for j := range opt.M.Data[i] {
				opt.M.Data[i][j] = 0
				opt.V.Data[i][j] = 0
			}
		}
	}
}
