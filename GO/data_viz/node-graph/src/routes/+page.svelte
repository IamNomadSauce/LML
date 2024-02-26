<script>
    import { onMount } from 'svelte';
    import DecisionTree from '../lib/Decision_Tree.svelte';
  
    let graphDataObject = {};
  
    onMount(async () => {
      let response = await fetch('http://localhost:8069/graph');
      console.log(response)
      graphDataObject = await response.json();
      console.log(graphDataObject, "\n", typeof(graphDataObject));
    });
  </script>
  
  <DecisionTree {graphDataObject} />