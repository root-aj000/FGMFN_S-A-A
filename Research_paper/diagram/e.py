from graphviz import Digraph

dot = Digraph('Dimensional_Projection_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Inputs
dot.node('V', 'Visual Embedding\nV ∈ ℝ^{d_v}', shape='rect', style='filled', fillcolor='white')
dot.node('E', 'Text Embedding\nE ∈ ℝ^{d_t}', shape='rect', style='filled', fillcolor='white')

# Linear Projection layers
dot.node('ProjV', 'Linear Projection (W_v, b_v)\nTransforms Visual Features', shape='box', style='rounded,filled', fillcolor='white')
dot.node('ProjE', 'Linear Projection (W_t, b_t)\nTransforms Textual Features', shape='box', style='rounded,filled', fillcolor='white')

# Shared latent space
dot.node('Shared', 'Shared Latent Space ℝ^{d_s}\nAligned Multimodal Embedding Space', shape='ellipse', style='filled', fillcolor='white')

# Outputs
dot.node('Vdash', "Projected Visual Embedding\nV' = W_v V + b_v", shape='rect', style='filled', fillcolor='white')
dot.node('Edash', "Projected Text Embedding\nE' = W_t E + b_t", shape='rect', style='filled', fillcolor='white')

# Connections
dot.edge('V', 'ProjV')
dot.edge('E', 'ProjE')
dot.edge('ProjV', 'Shared')
dot.edge('ProjE', 'Shared')
dot.edge('Shared', 'Vdash')
dot.edge('Shared', 'Edash')

# Render diagram
dot.render('Dimensional_Projection_Module', view=True)
