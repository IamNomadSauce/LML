<script>
  import { onMount } from "svelte";

  export let graphDataArray = [];
//   $: console.log("DATA",graphDataArray)
  $: if (Object.keys(graphDataArray).length > 0) {
    graphDataArray = Object.values(graphDataArray)
  }
  const initialX = 100; // Initial X position for the first node
  const paddingX = 100; // Horizontal padding value between parent and child nodes
  const paddingY = 100; // Vertical padding value between child nodes
  let nodes = [];
  let edges = [];
  let outerWidth =0
  let innerWidth =0
  let outerHeight =0
  let innerHeight =0


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
        // Calculate the offset to start placing children symmetrically around the parent's baseY
        let offset = (parentNode.children.length - 1) * paddingY / 2;

        // Determine the X position for all children of this parent
        let childrenX = parentNode.x + paddingX;

        parentNode.children.forEach((childId, index) => {
            let childNode = graphData.find(n => n.id === childId.id);
            if (childNode) {
            childNode.x = childrenX; // Assign the same X value to all children
            // Calculate y position for each child to be symmetrical around the parent's baseY
            childNode.y = parentNode.y + (index * paddingY) - offset;
            }
        });
        }
    });

    return graphData;
    }
  
    // Parse graph data and apply layout
    function parseGraph(graph) {
        // console.log("Parse Graph")
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

    // $: console.log("Nodes\n", nodes, "\nEdges\n", edges)

    async function addNode(nodeData) {
        console.log("Add Node", nodeData)
    try {
      const response = await fetch('http://localhost:8069/add_node', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(nodeData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Node added:', result);
      graphDataArray = result
    } catch (error) {
      console.error('Error adding node:', error);
    }
  }
</script>

<h1>Graph</h1>
<svelte:window bind:outerWidth bind:outerHeight bind:innerWidth bind:innerHeight />

<svg class="graph" width={innerWidth} height={innerHeight}>
  
  {#each edges as edge}
    <!-- {#if nodes.find((n) => n.id === edge.source) && nodes.find((n) => n.id === edge.target)} -->
        <line
        x1={nodes.find((n) => n.id === edge.source).x}
        y1={nodes.find((n) => n.id === edge.source).y}
        x2={nodes.find((n) => n.id === edge.target.id).x}
        y2={nodes.find((n) => n.id === edge.target.id).y}
        stroke="black"
        />
    <!-- {/if} -->
    {/each}
    {#each nodes as node, i}
    <circle
      on:mouseover={() => console.log("node", i, node, nodes)}
      on:click={() => addNode(node)}

      cx={node.x}
      cy={node.y}
      r="10"
      fill="blue"
    />
  {/each}
</svg>

<style>
  .graph {
    background-color: #212529 !important;
  }
</style>
