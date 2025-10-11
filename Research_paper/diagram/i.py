from graphviz import Digraph

# Initialize diagram
dot = Digraph('Dimensional_Projection_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Inputs
dot.node('V', 'Visual Embedding\nV ∈ ℝ^{d_v} (e.g., 2048)', shape='ellipse', style='filled', fillcolor='white')
dot.node('T', 'Text Embedding\nT ∈ ℝ^{d_t} (e.g., 768)', shape='ellipse', style='filled', fillcolor='white')

# Linear Projections
dot.node('ProjV', 'Linear Projection\nV\' = W_v V + b_v', shape='box', style='rounded,filled', fillcolor='white')
dot.node('ProjT', 'Linear Projection\nT\' = W_t T + b_t', shape='box', style='rounded,filled', fillcolor='white')

# Shared Latent Space
dot.node('Latent', 'Shared Latent Space\nDimensionality: d_s', shape='ellipse', style='filled', fillcolor='white')

# Connections
dot.edge('V', 'ProjV')
dot.edge('T', 'ProjT')
dot.edge('ProjV', 'Latent')
dot.edge('ProjT', 'Latent')

# Render
dot.render('Dimensional_Projection_Module', view=True)
