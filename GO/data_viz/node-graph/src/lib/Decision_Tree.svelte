<script>
    import { onMount } from 'svelte';
    import * as d3 from 'd3';
  
    export let graphDataObject = {}; // This will be passed from the parent component
  
    let nodes = [];
    let links = [];
  
    // Dimensions of the SVG canvas
    const width = 800;
    const height = 600;
  
    // Parse graph data and apply tree layout
    function applyTreeLayout(graph) {
      // Create a d3 hierarchy from the graph
      const root = d3.hierarchy(graph);
  
      // Create a tree layout
      const treeLayout = d3.tree().size([height, width - 160]); // Adjust size to leave some padding
  
      // Apply the tree layout to the root hierarchy
      const treeData = treeLayout(root);
  
      // Extract nodes and links from the tree data
      nodes = treeData.descendants();
      links = treeData.links();
    }
  
    let translate = { x: 0, y: 0 };
  
    onMount(() => {
      if (Object.keys(graphDataObject).length > 0) {
        applyTreeLayout(graphDataObject); // Pass the hierarchical data to the layout function
        // Calculate the extent of the tree and translate the group to center it
        const xExtent = d3.extent(nodes, d => d.x);
        const yExtent = d3.extent(nodes, d => d.y);
        translate = {
          x: (width - (yExtent[1] - yExtent[0])) / 2 - yExtent[0],
          y: (height - (xExtent[1] - xExtent[0])) / 2 - xExtent[0]
        };
      }
    });

    $: if (Object.keys(graphDataObject).length > 0) {
        applyTreeLayout(graphDataObject); // Pass the hierarchical data to the layout function
        // Calculate the extent of the tree and translate the group to center it
        const xExtent = d3.extent(nodes, d => d.x);
        const yExtent = d3.extent(nodes, d => d.y);
        translate = {
          x: (width - (yExtent[1] - yExtent[0])) / 2 - yExtent[0],
          y: (height - (xExtent[1] - xExtent[0])) / 2 - xExtent[0]
        };
      }
  </script>
  
  <h1>Decision Tree</h1>
  <svg class="graph" width={width} height={height}>
    <g transform={`translate(${translate.x},${translate.y})`}>
      {#each links as link}
        <line x1={link.source.y} y1={link.source.x}
              x2={link.target.y} y2={link.target.x}
              stroke="black" />
      {/each}
      {#each nodes as node}
        <circle cx={node.y} cy={node.x} r="10" fill="blue" />
        <text x={node.y + 12} y={node.x + 3} style="font: 10px sans-serif;">
          {node.data.id}
        </text>
      {/each}
    </g>
  </svg>
  
  <style>
    .graph {
      background-color: #212529 !important;
      display: block;
      margin: auto;
    }
  </style>