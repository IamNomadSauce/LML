package nn

// Optimizer represents an optimizer
type Optimizer interface {
	UpdateParams(layer Layer)
}
