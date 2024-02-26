<script>
    export let nodes = [];
    export let edges = [];
</script>
  

<h1>
Graph
</h1>
<svg class="graph" width="800" height="600">
    {#each nodes as node}
        <circle on:mouseover={()=>console.log("node", node, nodes)}
        cx={node.x} cy={node.y} r="10" fill="blue" />
    {/each}
    {#each edges as edge}
        <line x1={edge.from.x} y1={edge.from.y} x2={edge.to.x} y2={edge.to.y} stroke="black" />
    {/each}
</svg>

<style>
    .graph {
        background-color: black !important;
    }
</style>