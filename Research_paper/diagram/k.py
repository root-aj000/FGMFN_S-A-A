from graphviz import Digraph

# Initialize diagram
dot = Digraph('Cross_Modal_Fusion_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Inputs: Projected embeddings
dot.node('Vs', 'Visual Embedding\nV_s ∈ ℝ^{d_s}', shape='ellipse', style='filled', fillcolor='white')
dot.node('Ts', 'Text Embedding\nT_s ∈ ℝ^{d_s}', shape='ellipse', style='filled', fillcolor='white')

# Cross-Attention
dot.node('Text2Img', 'Text-to-Image Attention\nE_attn = softmax(Q_t K_v^T)V_s', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Img2Text', 'Image-to-Text Attention\nV_attn = softmax(Q_v K_t^T)T_s', shape='box', style='rounded,filled', fillcolor='white')

# Joint Fusion
dot.node('Fusion', 'Joint Fusion\nM = concat(E_attn, V_attn) + Residual', shape='box', style='rounded,filled', fillcolor='white')

# Connections
dot.edge('Vs', 'Text2Img')
dot.edge('Ts', 'Text2Img')
dot.edge('Vs', 'Img2Text')
dot.edge('Ts', 'Img2Text')
dot.edge('Text2Img', 'Fusion')
dot.edge('Img2Text', 'Fusion')

# Render
dot.render('Cross_Modal_Fusion_Module', view=True)
