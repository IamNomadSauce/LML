package nn

import (
	"fmt"
	_ "fmt"
	"go_nn/tensor"
	_ "math"
)

type Model struct {
	N_Input        int
	N_Hidden       int
	N_Outputs      int
	Hidden_Weights *tensor.Tensor
	Output_Weights *tensor.Tensor
	Learning_Rate  float64
}

func NewModel() *Model {
	return &Model{}
}

// c CreateModel

func (net *Model) Train(X, y, X_test, y_test *tensor.Tensor, epochs int) {
	fmt.Println("Train the Model", X, y)
}
