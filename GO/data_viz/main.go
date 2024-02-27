package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
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
	fmt.Println("Add Node", id)
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
func decision_tree(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Decision_Tree")
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

// graphHandler responds with the graph data in JSON format.
func node_graphHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Node_Graph")

	enableCors(&w) // Enable CORS

	if r.Method == "OPTIONS" {
		return // Handle preflight request
	}

	// Convert the graph to JSON
	graphJSON, err := json.Marshal(graph.Nodes)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Set the content type to application/json
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(graphJSON)
}

func api_add_node(w http.ResponseWriter, r *http.Request) {
	enableCors(&w) // Enable CORS for all responses

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var newNode Node
	err = json.Unmarshal(body, &newNode)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var parentNode Node
	parentNode = newNode

	// Find the maximum existing node ID
	maxID := -1
	for id := range graph.Nodes {
		if id > maxID {
			maxID = id
		}
	}

	// Set the newNode.ID to the maximum ID found plus one
	newNode.ID = maxID + 1

	// Add the new node to the graph
	graph.AddNode(newNode.ID)
	graph.AddEdge(parentNode.ID, newNode.ID)

	// for _, child := range newNode.Children {
	// 	graph.AddEdge(newNode.ID, child.ID)
	// }

	fmt.Printf("Received node: %+v\n", newNode)

	graphJSON, err := json.Marshal(graph.Nodes)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(graphJSON)
}

var graph = NewGraph()

func main() {
	// Set up the HTTP server
	graph.AddNode(0)
	graph.AddNode(1)
	graph.AddNode(2)
	graph.AddNode(3)

	graph.AddEdge(0, 1)
	graph.AddEdge(0, 2)

	// graph.AddNode(5)
	// graph.AddNode(6)
	// graph.AddNode(13)
	// graph.AddNode(15)

	// // graph.AddEdge(1, 2)
	// // graph.AddEdge(1, 3)

	// graph.AddEdge(5, 6)
	// graph.AddEdge(5, 13)

	// graph.AddEdge(5, 15)
	// graph.AddEdge(5, 18)

	// graph.AddEdge(13, 2)
	// graph.AddEdge(13, 3)

	// graph.AddNode(25)
	// graph.AddNode(30)

	// graph.AddEdge(0, 25)
	// graph.AddEdge(25, 6)
	// graph.AddEdge(25, 30)
	http.HandleFunc("/node_graph", node_graphHandler)
	http.HandleFunc("/add_node", api_add_node)
	http.HandleFunc("/decision_tree", decision_tree)
	fmt.Println("Server is running on port 8069...")
	http.ListenAndServe(":8069", nil)
}
