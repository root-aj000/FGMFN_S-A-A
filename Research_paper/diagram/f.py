from graphviz import Digraph

# Initialize diagram
dot = Digraph('Cross_Modal_Fusion_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Inputs
dot.node('Vdash', "Projected Visual Embedding\nV' ∈ ℝ^{d_s}", shape='rect', style='filled', fillcolor='white')
dot.node('Edash', "Projected Text Embedding\nE' ∈ ℝ^{d_s}", shape='rect', style='filled', fillcolor='white')

# Attention mechanisms
dot.node('T2I', 'Text-to-Image Attention\nE\' → (V\')\nHighlights visual regions relevant to text',
         shape='box', style='rounded,filled', fillcolor='white')
dot.node('I2T', 'Image-to-Text Attention\nV\' → (E\')\nHighlights textual tokens relevant to image',
         shape='box', style='rounded,filled', fillcolor='white')

# Fusion block
dot.node('Fusion', 'Joint Fusion\n(Concatenation + Residual Addition)\nCombines attended outputs',
         shape='box', style='rounded,filled', fillcolor='white')

# Output
dot.node('M', 'Joint Multimodal Embedding\nM ∈ ℝ^{d_s}\nEncodes aligned image-text features',
         shape='ellipse', style='filled', fillcolor='white')

# Connections
dot.edge('Vdash', 'I2T')
dot.edge('Edash', 'T2I')
dot.edge('I2T', 'Fusion')
dot.edge('T2I', 'Fusion')
dot.edge('Fusion', 'M')

# Render
dot.render('Cross_Modal_Fusion_Module', view=True)
