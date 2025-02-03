## `rust_bounded_graph`
_A thin newtype wrapper for [`petgraph`](https://github.com/petgraph/petgraph) to assist in the creation of graphs with restrictions on their edges_


This crate is a simple wrapper around petgraph's Graph type. It exists to make it simpler to enforce restrictions at the time of edge creation, to ensure that the graph is never in a state with an "invalid" edge between two nodes.

---

In order to do so, your Node type should implement the following trait:
```rust
pub trait BoundedNode<Ix: IndexType = DefaultIx> {
    fn can_add_edge(&self, dir: Direction, existing_edge_count: usize, other_node: &Self) -> bool;
}
```

Alternatively, for the common and simple situation of a Node with an associated limit on incoming and outgoing edges, one can alternatively implement the following trait:
```rust
pub trait EdgeNumberBoundedNode<Ix: IndexType = DefaultIx> {
    fn max_incoming_edges(&self) -> Ix;
    fn max_outgoing_edges(&self) -> Ix;
}
```
This will provide a function 
```rust
has_edge_space(&self, dir: Direction, existing_edge_count: usize) -> bool
```
If you have no other requirements for your node type, implement the marker trait `SimpleEdgeNumberBoundedNode` to automatically use `has_edge_space` as `can_add_edge`

---

Most methods and traits of [Graph](https://docs.rs/petgraph/latest/petgraph/graph/struct.Graph.html) have been added directly to the BoundedGraph struct provided by this crate, although updating and adding edges is now a failable operation. You may obtain a Graph from a bounded Graph by calling `as_graph()`.

Currently, the following methods are known to be unimplemented:

* update_edge on the Build trait can panic if the edge is new, and invalid.
* `raw_nodes()`, `raw_edges()`, `first_edge()` and `next_edge()` are not implemented on BoundedGraph, as they are low level functions and I currently do not need them. Please file an issue if this is a problem for you.
* `reverse()` is skipped for the first version as its especially likely to break constraits... and I don't need it yet.
* `Arbitrary` trait is unlikely to be implemented at this time.


Please let me know if anything else is missing or incorrect!