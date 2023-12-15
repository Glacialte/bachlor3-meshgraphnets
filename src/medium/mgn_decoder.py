self.decoder = Sequential(Linear(hidden_dim , hidden_dim),
                                 ReLU(),
                                 Linear( hidden_dim, output_dim)
                                 )