<script>
    import { onMount } from 'svelte';
    import Graph from '../lib/Graph.svelte';
    import DecisionTree from '../lib/Decision_Tree.svelte';
  
    let graphDataArray = {};
    let decision_tree_data = {};

  
    onMount(async () => {
      let response = await fetch('http://localhost:8069/node_graph');
      graphDataArray = await response.json();
      console.log("Node Graph\n",graphDataArray, "\n", typeof(graphDataArray));

      let response_2 = await fetch('http://localhost:8069/decision_tree');
      decision_tree_data = await response_2.json();
      console.log("Decision Tree\n",decision_tree_data, "\n", typeof(decision_tree_data), decision_tree_data.length);
    });
  </script>
  


  <Graph {graphDataArray} />
  <!-- <DecisionTree {decision_tree_data} /> -->