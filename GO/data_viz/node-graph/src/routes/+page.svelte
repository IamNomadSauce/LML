<script>
    import { onMount } from 'svelte';
    import Graph from '$lib/Graph.svelte';
  
    let nodes = [];
    let links = [];
  
    onMount(async () => {
      const response = await fetch('http://localhost:8069/graph');
      const graphDataObject = await response.json();
      console.log(graphDataObject, "\n", typeof(graphDataObject));
  
      // Transform the object into an array of objects
      const graphDataArray = Object.values(graphDataObject);
      console.log(graphDataArray, "\n", typeof(graphDataArray));
  
      // Now you can use .map and .flatMap on the array
      nodes = graphDataArray.map(node => ({ id: node.id }));
      links = graphDataArray.flatMap(node => 
        (node.children || []).map(childId => ({ source: node.id, target: childId }))
      );
    });
  </script>
  
  <Graph {nodes} {links} />