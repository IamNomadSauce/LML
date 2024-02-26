package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// Node represents a node in the graph with a unique identifier and a list of child nodes.
type Node struct {
	ID       int     `json:"id"`
	Children []*Node `json:"children,omitempty"` // Use pointers to Node
}

// Graph represents a directed graph.
type Graph struct {
	Nodes map[int]*Node
}

// NewGraph creates a new Graph instance.
func NewGraph() *Graph {
	return &Graph{
		Nodes: make(map[int]*Node),
	}
}

// AddNode adds a new node to the graph with the given id.
func (g *Graph) AddNode(id int) {
	if _, exists := g.Nodes[id]; !exists {
		g.Nodes[id] = &Node{ID: id}
	}
}

// AddEdge adds a directed edge from the node with id `from` to the node with id `to`.
func (g *Graph) AddEdge(from, to int) {
	fromNode, fromExists := g.Nodes[from]
	toNode, toExists := g.Nodes[to]

	if !fromExists || !toExists {
		fmt.Printf("One or both nodes not found in the graph: from=%d, to=%d\n", from, to)
		return
	}

	fromNode.Children = append(fromNode.Children, toNode) // Add pointer to the child node
}

// FindRootNode finds the root node of the graph (node with no incoming edges).
func (g *Graph) FindRootNode() *Node {
	// Assuming that the root node is the one with no incoming edges
	incomingEdges := make(map[int]bool)
	for _, node := range g.Nodes {
		for _, child := range node.Children {
			incomingEdges[child.ID] = true
		}
	}

	for id, node := range g.Nodes {
		if !incomingEdges[id] {
			return node // Found the root node
		}
	}
	return nil // No root node found
}

// enableCors sets the necessary headers to enable CORS.
func enableCors(w *http.ResponseWriter) {
	(*w).Header().Set("Access-Control-Allow-Origin", "*")
	(*w).Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	(*w).Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
}

// graphHandler responds with the graph data in JSON format.
func graphHandler(w http.ResponseWriter, r *http.Request) {
	enableCors(&w) // Enable CORS

	if r.Method == "OPTIONS" {
		return // Handle preflight request
	}

	graph := NewGraph()
	// Initialize your graph here

	graph.AddNode(5)
	graph.AddNode(6)
	graph.AddNode(13)
	graph.AddNode(15)
	graph.AddNode(18)
	graph.AddNode(20)

	graph.AddNode(1)
	graph.AddNode(2)
	graph.AddNode(3)

	graph.AddEdge(5, 6)
	graph.AddEdge(5, 13)

	graph.AddEdge(5, 15)
	graph.AddEdge(5, 18)

	graph.AddEdge(13, 1)
	graph.AddEdge(13, 2)
	graph.AddEdge(13, 3)

	graph.AddEdge(15, 2)
	graph.AddEdge(15, 1)
	graph.AddEdge(18, 1)
	graph.AddEdge(3, 20)
	graph.AddEdge(2, 20)

	rootNode := graph.FindRootNode()
	if rootNode == nil {
		http.Error(w, "Root node not found", http.StatusInternalServerError)
		return
	}
	graphJSON, err := json.Marshal(rootNode)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Set the content type to application/json
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(graphJSON)
}

func main() {
	// Set up the HTTP server
	http.HandleFunc("/graph", graphHandler)
	fmt.Println("Server is running on port 8069...")
	http.ListenAndServe(":8069", nil)
}
