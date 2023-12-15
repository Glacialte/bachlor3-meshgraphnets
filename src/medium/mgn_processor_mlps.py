# Note that the node and edge encoders both have the same hidden dimension
# size. This means that the input of the edge processor will always be
# three times the specified hidden dimension
# (input: adjacent node embeddings and self embeddings)
self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                           ReLU(),
                           Linear( out_channels, out_channels),
                           LayerNorm(out_channels))

self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                           ReLU(),
                           Linear( out_channels, out_channels),
                           LayerNorm(out_channels))