<script>
    import { onMount } from 'svelte';
  
    export let graphDataArray = [];
    let nodes = [];
    let edges = [];
  
    // Dimensions of the SVG canvas
    const width = 800;
    const height = 600;
  
    // Center of the SVG canvas
    const centerX = width / 2;
    const centerY = height / 2;
  
    // Radius of the circle on which nodes will be placed
    const radius = Math.min(width, height) / 3; // Adjust as needed
  
    // Parse graph data
    function parseGraph(graph) {
      if (graph.length === 0) {
        return { nodes: [], edges: [] };
      }
      // Calculate angle step for each node based on the total number
      const angleStep = (2 * Math.PI) / graph.length;
  
      let nodes = graph.map((d, i) => ({
        id: d.id,
        // Calculate x and y using circle formula
        x: centerX + radius * Math.cos(i * angleStep),
        y: centerY + radius * Math.sin(i * angleStep)
      }));
      let edges = graph.flatMap(d =>
        (d.children || []).map(childId => ({
          source: d.id,
          target: childId
        }))
      );
      return { nodes, edges };
    }
  
    onMount(() => {
      if (graphDataArray.length > 0) {
        let data = parseGraph(graphDataArray);
        nodes = data.nodes;
        edges = data.edges;
      }
    });
  
    $: if (graphDataArray.length > 0) {
        
        let data = parseGraph(graphDataArray);
        nodes = data.nodes;
        edges = data.edges;
      }
  </script>
  
  <h1>Graph</h1>
  <svg class="graph" width={width} height={height}>
    {#each edges as edge}
      <line x1={edge.source.x} y1={edge.source.y}
            x2={edge.target.x} y2={edge.target.y}
            stroke="black" />
    {/each}
    {#each nodes as node}
      <circle cx={node.x} cy={node.y} r="10" fill="blue" />
    {/each}
  </svg>
  
  <style>
    .graph {
      background-color: #212529 !important;
      display: block; /* Centers the SVG horizontally */
      margin: auto; /* Centers the SVG block in the page */
    }
  </style>