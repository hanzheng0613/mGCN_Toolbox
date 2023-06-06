#!/usr/bin/env python

# Here is an example of visualization of multilayer network

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

import scipy.io as sio
import scipy.sparse as sp


import holoviews as hv

from mGCN_Toolbox.visualization.process import *








class LayeredNetworkGraph(object):

    def __init__(self, graphs, graphs_attribute,layout=nx.random_layout):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        self.graphs = graphs
        
        self.graphs_attribute = graphs_attribute
        self.layout = layout
        self.node_rwoc = self.random_walk_occupation_centrality(self.graphs_attribute)
        self.get_nodes_and_edges_layout()
        self.get_edges_between_layers()
        # create internal representation of nodes and edges
        self.draw_nodes(self.nodes_positions)
        self.draw_edges(self.edges_positions)
        self.draw_edges_between_layers(self.edges_between_layers)
        self.add_interaction(self.graphs_attribute,self.node_rwoc)
        
        self.draw()
        
        

        
        




    def get_nodes_and_edges_layout(self, *args, **kwargs):

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        self.pos = self.layout(composition)

        self.node_dict = dict()
        for z, g in enumerate(self.graphs):
            self.node_dict.update({(node, z) : (*self.pos[node], z) for node in g.nodes()})
            
        self.nodes_positions= []
        self.edges_positions= []
        for x, y in self.node_dict.items():
            self.nodes_positions.append(y)
            
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])
        
        self.edges_positions = [(self.node_dict[source], self.node_dict[target]) for source, target in self.edges_within_layers]
        

    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((*self.pos[node], z1), (*self.pos[node], z2)) for node in shared_nodes])
        
    
    def draw_nodes(self,nodes_positions,*args, **kwargs):
        node_x = []
        node_y = []
        node_z = []
        self.node_trace = None
        for j in range(len(self.nodes_positions)):
            x,y,z= self.nodes_positions[j]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        self.node_trace = go.Scatter3d(
            x=node_x, y=node_y,z = node_z,
            mode='markers',
            hoverinfo='text',
            marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=5,
            colorbar=dict(
            thickness=15,
            title='Degree',
            xanchor='left',
            titleside='right'
            ),
            line_width=2))
        
    
    def draw_edges(self, edges_positions,*args, **kwargs):
        edge_x = []
        edge_y = []
        edge_z = []
        self.edge_trace = None
        self.edges_position = self.edges_positions[1500:]
        for i in range(len(self.edges_position)):
            x0, y0, z0= self.edges_position[i][0]
            x1, y1, z1= self.edges_position[i][1]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

        self.edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z= edge_z,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
    def draw_edges_between_layers(self, edges_between_layers,*args, **kwargs):
        edge_between_layers_x = []
        edge_between_layers_y = []
        edge_between_layers_z = []
        self.edge_between_layers_trace = None
        for i in range(len(self.edges_between_layers)):
            x0, y0, z0= self.edges_between_layers[i][0]
            x1, y1, z1= self.edges_between_layers[i][1]
            edge_between_layers_x.append(x0)
            edge_between_layers_x.append(x1)
            edge_between_layers_x.append(None)
            edge_between_layers_y.append(y0)
            edge_between_layers_y.append(y1)
            edge_between_layers_y.append(None)
            edge_between_layers_z.append(z0)
            edge_between_layers_z.append(z1)
            edge_between_layers_z.append(None)

        self.edge_between_layers_trace = go.Scatter3d(
            x=edge_between_layers_x, y=edge_between_layers_y, z= edge_between_layers_z,
            line=dict(width=0.5, color='#888',dash='dot'),
            hoverinfo='none',
            mode='lines',
            )
        
    def random_walk_occupation_centrality(self,graphs_attribute):
        weight=self.graphs_attribute.sum()
        size = self.graphs_attribute.shape[0]
        rWOC=np.zeros(size)
        for i in range (size):
            rWOC[i]=self.graphs_attribute[i,:].sum()/weight
        
        return rWOC
    
    def add_interaction(self,graphs_attribute,node_rwoc):
        node_text = []
        node_adjacencies = []

        for i in self.graphs:

            
            for node, adjacencies in enumerate(i.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                attri_num = "{:.4f}".format(self.graphs_attribute[node].max())
                node_text.append('Node index: '+ str(node)+'<br>'+'Attribute: ' + '[' + str(attri_num)+','+ str(attri_num)+']'
                                +'<br>'+'Degree: '+str(len(adjacencies[1]))+'<br>'+'Random Walk Occupation centrality: '
                                + str(node_rwoc[node]))
        
        self.node_trace.marker.color = node_adjacencies

        self.node_trace.text = node_text
    
    

    def draw(self):

        fig = go.Figure(data=[self.edge_trace, self.node_trace,self.edge_between_layers_trace],
             layout=go.Layout(
                title='<br>Multilayer network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text=" ",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

        fig.show()
        
    
        
        

if __name__ == '__main__':
    
    adj_list, truefeatures, label, idx_train, idx_val, idx_test = load_acm_mat(3)

    # Generating sample data

    a1 = adj_list[0]
    a2 = adj_list[1]
    a1_mini = a1[0:100,0:100]
    a2_mini = a2[0:100,0:100]
    
    features = truefeatures[0:100,0:100]
    attribute = preprocess_features(features)


    # define graphs
    G = nx.from_scipy_sparse_array(a1_mini)

    I = nx.from_scipy_sparse_array(a2_mini)
    # initialise figure and plot
    
    LayeredNetworkGraph([G,I],graphs_attribute=attribute,layout=nx.random_layout)
    # Make interactive nodes here