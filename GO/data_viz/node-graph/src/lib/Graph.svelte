<script>
    import { onMount } from 'svelte';
  
    export let graphDataArray = [];
    const initialX = 100; // Initial X position for the first node
    const paddingX = 100; // Horizontal padding value between parent and child nodes
    const paddingY = 100; // Vertical padding value between child nodes
    let nodes = [];
    let edges = [];
  
    // Function to calculate initial positions for nodes
    // Child nodes are positioned to the right of their parent node
    function calculateInitialPositions(graphData) {
      let currentPositionX = initialX;
      let baseY = 300; // Base Y position for the first node
  
      // Assign x and y positions to parent nodes
      graphData.forEach(node => {
        node.x = currentPositionX; // Set x for parent node
        node.y = baseY; // Set y for parent node
  
        // Increment x position for the next node
        currentPositionX += paddingX;
      });
  
      // Assign x and y positions to child nodes
      graphData.forEach(parentNode => {
        if (parentNode.children && parentNode.children.length > 0) {
          // Calculate starting y position for the first child
          let startY = parentNode.y - parentNode.children.length * paddingY / 2 + paddingY / 2;
          parentNode.children.forEach((childId, index) => {
            let childNode = graphData.find(n => n.id === childId);
            if (childNode) {
              childNode.x = parentNode.x + paddingX; // Position child to the right of the parent
              childNode.y = startY + index * paddingY; // Calculate y position for each child
            }
          });
        }
      });
  
      return graphData;
    }
  
    // Parse graph data and apply layout
    function parseGraph(graph) {
      if (graph.length === 0) {
        return { point: [], links: [] };
      }
      let positionedNodes = calculateInitialPositions(graph);
      let links = graph.flatMap(node =>
        (node.children || []).map(childId => ({ source: node.id, target: childId }))
      );
      return { point: positionedNodes, links };
    }
  
    $: if (graphDataArray.length > 0) {
      let data = parseGraph(graphDataArray);
      nodes = data.point;
      edges = data.links;
    }
  </script>
  
  <h1>Graph</h1>
  <svg class="graph" width="800" height="600">
    {#each nodes as node}
      <circle on:mouseover={() => console.log("node", node, nodes)}
              cx={node.x} cy={node.y} r="10" fill="blue" />
    {/each}
    {#each edges as edge}
      <line x1={nodes.find(n => n.id === edge.source).x} 
            y1={nodes.find(n => n.id === edge.source).y} 
            x2={nodes.find(n => n.id === edge.target).x} 
            y2={nodes.find(n => n.id === edge.target).y} 
            stroke="black" />
    {/each}
  </svg>
  
  <style>
    .graph {
      background-color: #212529 !important;
    }
  </style>