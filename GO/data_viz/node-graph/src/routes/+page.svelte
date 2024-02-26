<script>
    import { onMount } from 'svelte';
    import Graph from '$lib/Graph.svelte';
    import Force from '$lib/Force_Graph.svelte';
    import ForceGraph from '../lib/Force_Graph.svelte';

  
    let nodes = [];
    let links = [];
    let graphDataObject = []
    let graphDataArray= []

  
    onMount(async () => {
      let response = await fetch('http://localhost:8069/graph');
      graphDataObject = await response.json();
      console.log(graphDataObject, "\n", typeof(graphDataObject));
  
      // Transform the object into an array of objects
      graphDataArray = Object.values(graphDataObject);
    //   console.log(graphDataArray, "\n", typeof(graphDataArray));
  
      
    });
  </script>
  
  <ForceGraph {graphDataArray} />
  <!-- <Graph {graphDataArray} /> -->
