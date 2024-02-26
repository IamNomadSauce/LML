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
	LambdaL1     float64 // L1 regularization strength
	LambdaL2     float64 // L2 regularization strength
}

// NewAdamOptimizer creates a new instance of AdamOptimizer
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon, lambdaL1, lambdaL2 float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		LambdaL1:     lambdaL1,
		LambdaL2:     lambdaL2,
		M:            nil, // Initialized later based on parameter size
		V:            nil, // Initialized later based on parameter size
		T:            0,
	}
}

// Update updates the parameters based on gradients
func (opt *AdamOptimizer) Update(params, grads *tensor.Tensor) {
	if opt.M == nil {
		opt.M = tensor.NewZerosTensor(params.Rows, params.Cols)
		if opt.M == nil {
			fmt.Println("Failed to initialize M tensor")
			return
		}
	}
	if opt.V == nil {
		opt.V = tensor.NewZerosTensor(params.Rows, params.Cols)
		if opt.V == nil {
			fmt.Println("Failed to initialize V tensor")
			return
		}
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

			// Apply L1 regularization
			if opt.LambdaL1 > 0 {
				g += opt.LambdaL1 * Sign(params.Data[i][j])
			}

			// Apply L2 regularization
			if opt.LambdaL2 > 0 {
				g += opt.LambdaL2 * params.Data[i][j]
			}

			// Update parameters
			params.Data[i][j] -= opt.LearningRate * mHat / (math.Sqrt(vHat) + opt.Epsilon)
		}
	}
}

func (opt *AdamOptimizer) Step(model *Model) {
	fmt.Println("STEP")
	// Iterate through each parameter in the model
	for _, param := range model.Parameters() {
		// Retrieve the gradient for this parameter
		grad := model.GradientForParameter(param)

		// Use the Update method to apply the Adam optimization step
		opt.Update(param, grad)
	}
}

func Sign(x float64) float64 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
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
