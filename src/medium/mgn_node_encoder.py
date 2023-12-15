self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                               ReLU(),
                               Linear(hidden_dim, hidden_dim),
                               LayerNorm(hidden_dim))