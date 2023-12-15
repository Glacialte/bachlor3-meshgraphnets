''
For each node i in the graph with neighbor node j:
x - (the node features) contains the ground truth 2D velocities concatenated with the 9D node type one-hot vector for all nodes [num_nodes x 11]
edge_index - contains the connectivity of the graph in COO format. [2 x num_edges]
edge_attr - contains the 2D position vector, u_ij, between connecting nodes (u_ij=u_i-u_j) concatenated with the 2-norm of the position vector. [num_edges x 3]
y - (the node outputs) contains the fluid acceleration between the current graph and the graph in the next time step. These are the features used for training: y=(v_t_next-v_t_curr)/dt [num_nodes x 2]
p - pressure scalar, used for validation [num_nodes x 1]
cells and mesh_pos: these attributes contain no new information. They are included for ease of visualization of the results and explained in detail in the Colab.
'''
Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, p=p,
                                  cells=cells, mesh_pos=mesh_pos)