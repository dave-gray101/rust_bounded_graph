use std::{cmp, ops::{Index, IndexMut}};

use fixedbitset::FixedBitSet;
use petgraph::{ csr::IndexType, data::{Build, Create, DataMap, DataMapMut}, graph::{DefaultIx, Edge, EdgeIndex, EdgeIndices, EdgeReference, EdgeReferences, EdgeWeightsMut, Edges, EdgesConnecting, Externals, Frozen, GraphIndex, Neighbors, Node, NodeIndex, NodeIndices, NodeReferences, NodeWeightsMut}, graph6::ToGraph6, visit::{Data, EdgeCount, EdgeIndexable, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges, IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, IntoNodeReferences, NodeCompactIndexable, NodeCount, NodeIndexable, Visitable}, Directed, Direction, EdgeType, Graph, IntoWeightedEdge, Undirected};

#[cfg(feature = "serde-1")]
use serde::{Serialize, Deserialize};


pub trait BoundedNode<Ix: IndexType = DefaultIx> {
    fn can_add_edge(&self, existing_edges: usize, dir: Direction) -> bool;
}

pub trait EdgeNumberBoundedNode<Ix: IndexType = DefaultIx> {
    fn max_incoming_edges(&self) -> Ix;
    fn max_outgoing_edges(&self) -> Ix;
}

impl<Ix: IndexType, T: EdgeNumberBoundedNode<Ix>> BoundedNode<Ix> for T {
    fn can_add_edge(&self, existing_edges: usize, dir: Direction) -> bool {
        match dir {
            Direction::Incoming => {
                Ix::new(existing_edges) < self.max_incoming_edges()
            }
            Direction::Outgoing => {
                Ix::new(existing_edges) < self.max_outgoing_edges()
            }
        }
    
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(Serialize, Deserialize))]
pub struct BoundedGraph<N: BoundedNode<Ix>, E, Ty: EdgeType = Directed, Ix: IndexType = DefaultIx> {
    graph: Graph<N, E, Ty, Ix>,
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> BoundedGraph<N, E, Ty, Ix> {
    pub fn new() -> Self {
        Self {
            graph: Graph::<N, E, Ty, Ix>::default()
        }
    }

    pub fn can_add_edge(&self, node: NodeIndex<Ix>, dir: Direction) -> bool {
        let incoming = self.graph.edges_directed(node, dir);
        let weight = self.graph.node_weight(node);
        match weight {
            Some(weight) => {
                weight.can_add_edge(incoming.count(), dir)
            }
            None => false
        }
    }

    pub fn as_graph(&self) -> &Graph<N, E, Ty, Ix> {
        &self.graph
    }

    pub fn node_weight(&self, node: NodeIndex<Ix>) -> Option<&N> {
        self.graph.node_weight(node)
    }

    pub fn edge_weight(&self, edge: EdgeIndex<Ix>) -> Option<&E> {
        self.graph.edge_weight(edge)
    }

    pub fn node_weight_mut(&mut self, node: NodeIndex<Ix>) -> Option<&mut N> {
        self.graph.node_weight_mut(node)
    }

    pub fn edge_weight_mut(&mut self, edge: EdgeIndex<Ix>) -> Option<&mut E> {
        self.graph.edge_weight_mut(edge)
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    pub fn add_node(&mut self, node: N) -> NodeIndex<Ix> {
        self.graph.add_node(node)
    }

    pub fn add_edge(&mut self, source: NodeIndex<Ix>, target: NodeIndex<Ix>, edge: E) -> Result<EdgeIndex<Ix>, &'static str> {
        if ! self.can_add_edge(source, Direction::Outgoing) {
            return Err("Cannot add additional outgoing edge to source node");
        }

        if ! self.can_add_edge(target, Direction::Incoming) {
            return Err("Cannot add additional incoming edge to target node");
        }
        
        Ok(self.graph.add_edge(source, target, edge))
    }

    pub fn update_edge(&mut self, a: NodeIndex<Ix>, b: NodeIndex<Ix>, weight: E) -> Result<EdgeIndex<Ix>, &'static str> {
        match self.graph.find_edge(a, b) {
            Some(edge) => {
                self.graph[edge] = weight;
                Ok(edge)
            },
            None => self.add_edge(a, b, weight)
        }
    }

    pub fn edge_endpoints(&self, edge: EdgeIndex<Ix>) -> Option<(NodeIndex<Ix>, NodeIndex<Ix>)> {
        self.graph.edge_endpoints(edge)
    }

    pub fn remove_node(&mut self, node: NodeIndex<Ix>) -> Option<N> {
        self.graph.remove_node(node)
    }

    pub fn remove_edge(&mut self, edge: EdgeIndex<Ix>) -> Option<E> {
        self.graph.remove_edge(edge)
    }

    pub fn neighbors(&self, node: NodeIndex<Ix>) -> Neighbors<'_, E, Ix> {
        self.graph.neighbors(node)
    }

    pub fn neighbors_directed(&self, node: NodeIndex<Ix>, dir: Direction) -> Neighbors<'_, E, Ix> {
        self.graph.neighbors_directed(node, dir)
    }

    pub fn neighbors_undirected(&self, node: NodeIndex<Ix>) -> Neighbors<'_, E, Ix> {
        self.graph.neighbors_undirected(node)
    }

    pub fn edges(&self, node: NodeIndex<Ix>) -> Edges<'_, E, Ty, Ix> {
        self.graph.edges(node)
    }

    pub fn edges_directed(&self, node: NodeIndex<Ix>, dir: Direction) -> Edges<'_, E, Ty, Ix> {
        self.graph.edges_directed(node, dir)
    }

    pub fn edges_connecting(&self, a: NodeIndex<Ix>, b: NodeIndex<Ix>) -> EdgesConnecting<E, Ty, Ix> {
        self.graph.edges_connecting(a, b)
    }

    pub fn contains_edge(&self, a: NodeIndex<Ix>, b: NodeIndex<Ix>) -> bool {
        self.graph.contains_edge(a, b)
    }

    pub fn find_edge(&self, a: NodeIndex<Ix>, b: NodeIndex<Ix>) -> Option<EdgeIndex<Ix>> {
        self.graph.find_edge(a, b)
    }

    pub fn find_edge_undirected(&self, a: NodeIndex<Ix>, b: NodeIndex<Ix>) -> Option<(EdgeIndex<Ix>, Direction)> {
        self.graph.find_edge_undirected(a, b)
    }

    pub fn externals(&self, dir: Direction) -> Externals<'_, N, Ty, Ix> {
        self.graph.externals(dir)
    }

    pub fn node_indices(&self) -> NodeIndices<Ix> {
        self.graph.node_indices()
    }

    pub fn node_weights_mut(&mut self) -> NodeWeightsMut<'_, N, Ix> {
        self.graph.node_weights_mut()
    }

    pub fn node_weights(&self) -> impl Iterator<Item = &N> + '_ {
        self.graph.node_weights()
    }

    pub fn edge_indices(&self) -> EdgeIndices<Ix> {
        self.graph.edge_indices()
    }
    
    pub fn edge_references(&self) -> EdgeReferences<'_, E, Ix>{
        self.graph.edge_references()
    }

    pub fn edge_weights(&self) -> impl Iterator<Item = &E> + '_ {
        self.graph.edge_weights()
    }

    pub fn edge_weights_mut(&mut self) -> EdgeWeightsMut<E, Ix> {
        self.graph.edge_weights_mut()
    }

    pub fn into_nodes_edges(self) -> (Vec<Node<N, Ix>>, Vec<Edge<E, Ix>>) {
        self.graph.into_nodes_edges()
    }

    pub fn index_twice_mut<T, U>(
        &mut self,
        i: T,
        j: U,
    ) -> (
        &mut <Graph<N, E, Ty, Ix> as Index<T>>::Output,
        &mut <Graph<N, E, Ty, Ix> as Index<U>>::Output,
    )
    where
        Graph<N, E, Ty, Ix>: Index<T> + Index<U>,
        Graph<N, E, Ty, Ix>: IndexMut<T> + IndexMut<U>,
        T: GraphIndex,
        U: GraphIndex,
    {
        self.graph.index_twice_mut(i, j)
    }


    // TODO: this one is more complex. Leave out for v0.1
    // pub fn reverse(&mut self) {
    //     self.graph.reverse();
    // }

    pub fn clear(&mut self) {
        self.graph.clear();
    }

    pub fn clear_edges(&mut self) {
        self.graph.clear_edges();
    }

    pub fn capacity(&self) -> (usize, usize) {
        self.graph.capacity()
    }

    pub fn reserve_nodes(&mut self, additional: usize) {
        self.graph.reserve_nodes(additional)
    }

    pub fn reserve_edges(&mut self, additional: usize) {
        self.graph.reserve_edges(additional)
    }

    pub fn reserve_exact_nodes(&mut self, additional: usize) {
        self.graph.reserve_exact_nodes(additional)
    }

    pub fn reserve_exact_edges(&mut self, additional: usize) {
        self.graph.reserve_exact_edges(additional)
    }

    pub fn shrink_to_fit_nodes(&mut self) {
        self.graph.shrink_to_fit_nodes()
    }

    pub fn shrink_to_fit_edges(&mut self) {
        self.graph.shrink_to_fit_edges()
    }

    pub fn shrink_to_fit(&mut self) {
        self.graph.shrink_to_fit()
    }

    pub fn retain_nodes<F>(&mut self, visit: F)
    where
        F: FnMut(Frozen<Graph<N, E, Ty, Ix>>, NodeIndex<Ix>) -> bool,
    {
        self.graph.retain_nodes(visit)
    }

    pub fn retain_edges<F>(&mut self, visit: F)
    where
        F: FnMut(Frozen<Graph<N, E, Ty, Ix>>, EdgeIndex<Ix>) -> bool,
    {
        self.graph.retain_edges(visit)
    }

    // This one needs to be copy pasted in in order to invoke the BoundedGraph add_edge impl
    pub fn extend_with_edges<I>(&mut self, iterable: I)
    where
        I: IntoIterator,
        I::Item: IntoWeightedEdge<E>,
        <I::Item as IntoWeightedEdge<E>>::NodeId: Into<NodeIndex<Ix>>,
        N: Default,
    {
        let iter = iterable.into_iter();
        let (low, _) = iter.size_hint();
        self.graph.reserve_edges(low);

        for elt in iter {
            let (source, target, weight) = elt.into_weighted_edge();
            let (source, target) = (source.into(), target.into());
            let nx = cmp::max(source, target);
            while nx.index() >= self.node_count() {
                self.add_node(N::default());
            }
            let _ = self.add_edge(source, target, weight);  // Continue on failures
        }
    }

    // Same as above
    pub fn from_edges<I>(iterable: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoWeightedEdge<E>,
        <I::Item as IntoWeightedEdge<E>>::NodeId: Into<NodeIndex<Ix>>,
        N: Default,
    {
        let mut g = Self::with_capacity(0, 0);
        g.extend_with_edges(iterable);
        g
    }

    pub fn map<'a, F, G, N2, E2>(
        &'a self,
        node_map: F,
        edge_map: G,
    ) -> BoundedGraph<N2, E2, Ty, Ix>
    where
        F: FnMut(NodeIndex<Ix>, &'a N) -> N2,
        G: FnMut(EdgeIndex<Ix>, &'a E) -> E2,
        N2: BoundedNode<Ix>,
    {
        BoundedGraph::<N2, E2, Ty, Ix> {
            graph: self.graph.map(node_map, edge_map)
        }
    }

    pub fn filter_map<'a, F, G, N2, E2>(
        &'a self,
        node_map: F,
        edge_map: G,
    ) -> BoundedGraph<N2, E2, Ty, Ix>
    where
        F: FnMut(NodeIndex<Ix>, &'a N) -> Option<N2>,
        G: FnMut(EdgeIndex<Ix>, &'a E) -> Option<E2>,
        N2: BoundedNode<Ix>,
    {
        BoundedGraph::<N2, E2, Ty, Ix> {
            graph: self.graph.filter_map(node_map, edge_map)
        }
    }

    pub fn into_edge_type<NewTy: EdgeType>(self) -> BoundedGraph<N, E, NewTy, Ix> {
        BoundedGraph::<N, E, NewTy, Ix> {
            graph: self.graph.into_edge_type()
        }
    }

}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> GraphBase for BoundedGraph<N, E, Ty, Ix> {
    type NodeId = NodeIndex<Ix>;
    type EdgeId = EdgeIndex<Ix>;
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Data for BoundedGraph<N, E, Ty, Ix> {
    type NodeWeight = N;
    type EdgeWeight = E;
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> GraphProp for BoundedGraph<N, E, Ty, Ix> {
    fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }
    
    type EdgeType = Ty;
}

impl <N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> DataMap for BoundedGraph<N, E, Ty, Ix> {
    fn node_weight(&self, node: NodeIndex<Ix>) -> Option<&N> {
        self.graph.node_weight(node)
    }

    fn edge_weight(&self, edge: EdgeIndex<Ix>) -> Option<&E> {
        self.graph.edge_weight(edge)
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> DataMapMut for BoundedGraph<N, E, Ty, Ix> {
    fn node_weight_mut(&mut self, node: NodeIndex<Ix>) -> Option<&mut N> {
        self.graph.node_weight_mut(node)
    }

    fn edge_weight_mut(&mut self, edge: EdgeIndex<Ix>) -> Option<&mut E> {
        self.graph.edge_weight_mut(edge)
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> EdgeCount for BoundedGraph<N, E, Ty, Ix> {
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> EdgeIndexable for BoundedGraph<N, E, Ty, Ix> {

    fn edge_bound(self: &Self) -> usize {
        self.graph.edge_bound()
    }

    fn to_index(self: &Self, a: Self::EdgeId) -> usize {
        <Graph<N, E, Ty, Ix> as EdgeIndexable>::to_index(&self.graph, a)
    }

    fn from_index(self: &Self, i:usize) -> Self::EdgeId {
        <Graph<N, E, Ty, Ix> as EdgeIndexable>::from_index(&self.graph, i)
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> NodeIndexable for BoundedGraph<N,E, Ty, Ix> {

    fn node_bound(self: &Self) -> usize {
        self.graph.node_bound()
    }

    fn to_index(self: &Self, a:Self::NodeId) -> usize {
        <Graph<N, E, Ty, Ix> as NodeIndexable>::to_index(&self.graph, a)
    }

    
    fn from_index(self: &Self, i:usize) -> Self::NodeId {
        <Graph<N, E, Ty, Ix> as NodeIndexable>::from_index(&self.graph, i)
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Index<EdgeIndex<Ix>> for BoundedGraph<N, E, Ty, Ix> {
    type Output = <Graph<N, E, Ty, Ix> as Index<EdgeIndex<Ix>>>::Output;

    fn index(&self, index: EdgeIndex<Ix>) -> &Self::Output {
        &self.graph[index]
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Index<NodeIndex<Ix>> for BoundedGraph<N, E, Ty, Ix> {
    type Output = <Graph<N, E, Ty, Ix> as Index<NodeIndex<Ix>>>::Output;

    fn index(&self, index: NodeIndex<Ix>) -> &Self::Output {
        &self.graph[index]
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> IndexMut<EdgeIndex<Ix>> for BoundedGraph<N, E, Ty, Ix> {
    fn index_mut(&mut self, index: EdgeIndex<Ix>) -> &mut Self::Output {
        &mut self.graph[index]
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> IndexMut<NodeIndex<Ix>> for BoundedGraph<N, E, Ty, Ix> {
    fn index_mut(&mut self, index: NodeIndex<Ix>) -> &mut Self::Output {
        &mut self.graph[index]
    }
}


impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoEdgeReferences for &'a BoundedGraph<N, E, Ty, Ix> {
    type EdgeRef = EdgeReference<'a, E, Ix>;

    type EdgeReferences = EdgeReferences<'a, E, Ix>;

    fn edge_references(self) -> Self::EdgeReferences {
        self.graph.edge_references()
    }
}



impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoEdges for &'a BoundedGraph<N, E, Ty, Ix> {
    type Edges = Edges<'a, E, Ty, Ix>;

    fn edges(self, a:Self::NodeId) -> Self::Edges {
        self.graph.edges(a)
    }
}

impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoEdgesDirected for &'a BoundedGraph<N, E, Ty, Ix> {
    type EdgesDirected = Edges<'a, E, Ty, Ix>;

    fn edges_directed(self, a: Self::NodeId, dir: Direction) -> Self::EdgesDirected {
        self.graph.edges_directed(a, dir)
    }
}

impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoNeighbors for &'a BoundedGraph<N, E, Ty, Ix> {
    type Neighbors = Neighbors<'a, E, Ix>;


    fn neighbors(self,a:Self::NodeId) -> Self::Neighbors {
        self.graph.neighbors(a)
    }
}

impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoNeighborsDirected for &'a BoundedGraph<N, E, Ty, Ix> {
    type NeighborsDirected = Neighbors<'a, E, Ix>;


    fn neighbors_directed(self, a: Self::NodeId, dir: Direction) -> Self::Neighbors {
        self.graph.neighbors_directed(a, dir)
    }
}

impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoNodeReferences for &'a BoundedGraph<N, E, Ty, Ix> {
    type NodeRef = (NodeIndex<Ix>, &'a N);

    type NodeReferences = NodeReferences<'a, N, Ix>;

    fn node_references(self) -> Self::NodeReferences {
        self.graph.node_references()
    }
}

impl<'a, N: 'a + BoundedNode<Ix>, E: 'a, Ty: EdgeType, Ix: IndexType> IntoNodeIdentifiers for &'a BoundedGraph<N, E, Ty, Ix> {
    type NodeIdentifiers = NodeIndices<Ix>;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.graph.node_identifiers()
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> NodeCount for BoundedGraph<N, E, Ty, Ix> {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

impl<N: BoundedNode<Ix>, E, Ix: IndexType> ToGraph6 for BoundedGraph<N, E, Undirected, Ix> {
    fn graph6_string(&self) -> String {
        self.graph.graph6_string()
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Visitable for BoundedGraph<N, E, Ty, Ix> {

    type Map = FixedBitSet;


    fn visit_map(self: &Self) -> Self::Map {
        self.graph.visit_map()
    }

    fn reset_map(self: &Self, map: &mut Self::Map) {
        self.graph.reset_map(map)
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> NodeCompactIndexable for BoundedGraph<N, E, Ty, Ix> {}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Create for BoundedGraph<N, E, Ty, Ix> {
    fn with_capacity(nodes: usize, edges: usize) -> Self {
        BoundedGraph { graph: Graph::with_capacity(nodes, edges) }
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Default for BoundedGraph<N, E, Ty, Ix> {
    fn default() -> Self {
        BoundedGraph { graph: Graph::default() }
    }
}

impl<N: BoundedNode<Ix>, E, Ty: EdgeType, Ix: IndexType> Build for BoundedGraph<N, E, Ty, Ix> {
    fn add_node(&mut self, weight: Self::NodeWeight) -> Self::NodeId {
        self.add_node(weight)
    }

    // TODO: This panics. Should it?
    fn update_edge(
        &mut self,
        a: Self::NodeId,
        b: Self::NodeId,
        weight: Self::EdgeWeight,
    ) -> Self::EdgeId {
        self.update_edge(a, b, weight).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use petgraph::dot::{Config, Dot};
    use Direction::Incoming;
    use super::*;

    #[test]
    fn petgraph_basics() {

        #[derive(Debug, Default)]
        pub struct BoundedNodeIndex {
            name: String,
            inputs: usize,
            outputs: usize,
        }

        impl BoundedNodeIndex {
            pub fn new(name: String, inputs: usize, outputs: usize) -> Self {
                Self {
                    name,
                    inputs,
                    outputs,
                }
            }
        }
        impl<Ix: IndexType> EdgeNumberBoundedNode<Ix> for BoundedNodeIndex {
            fn max_incoming_edges(&self) -> Ix {
                Ix::new(self.inputs)
            }

            fn max_outgoing_edges(&self) -> Ix {
                Ix::new(self.outputs)
            }
        }

        let mut deps = BoundedGraph::<BoundedNodeIndex, &str>::new();
        let pg = deps.add_node(BoundedNodeIndex::new("petgraph".to_string(), 0usize, 3usize));
        let fb = deps.add_node(BoundedNodeIndex::new("fixedbitset".to_string(), 4usize, 0usize));
        let qc = deps.add_node(BoundedNodeIndex::new("quickcheck".to_string(), 1usize, 2usize));
        let rand = deps.add_node(BoundedNodeIndex::new("rand".to_string(), 1usize, 2usize));
        let libc = deps.add_node(BoundedNodeIndex::new("libc".to_string(), 1usize, 2usize));
        deps.extend_with_edges(&[
            (pg, fb), (pg, qc),
            (qc, rand), (rand, libc), (qc, libc),
        ]);
        assert_eq!(deps[pg].name, "petgraph");
        assert_eq!(deps.edge_count(), 4);   // One should be missing.
        assert_eq!(deps.edges_directed(libc, Incoming).count(), 1); // It should be missing from here.

        println!("{:?}", Dot::with_config(deps.as_graph(), &[Config::EdgeNoLabel]));

        deps.update_edge(qc, fb, "").unwrap();
        assert_eq!(deps.edge_count(), 5);
    }
}
