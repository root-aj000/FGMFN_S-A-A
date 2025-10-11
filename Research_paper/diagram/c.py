from graphviz import Digraph

dot = Digraph('Visual_Embedding_Module_ResNet18', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Input
dot.node('Input', 'Preprocessed Image Tensor\nI ∈ R^{H×W×3}', shape='rect', style='filled', fillcolor='white')

# Conv Layer
dot.node('Conv1', 'Conv1: 7×7 Conv, Stride 2\nDetects edges & simple textures', shape='box', style='rounded,filled', fillcolor='white')

# Max Pooling
dot.node('MaxPool', 'Max Pool: 3×3\nReduces spatial size, keeps strong features', shape='box', style='rounded,filled', fillcolor='white')

# Residual Blocks (ResNet18 Core)
dot.node('Res1', 'Residual Block 1\nLow-level features (edges, corners)\n2×3×3 Conv + Skip Connection', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Res2', 'Residual Block 2\nMid-level features (logos, shapes)\n2×3×3 Conv + Skip Connection', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Res3', 'Residual Block 3\nHigh-level patterns (layout, structure)\n2×3×3 Conv + Skip Connection', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Res4', 'Residual Block 4\nSemantic-level features (relationships)\n2×3×3 Conv + Skip Connection', shape='box', style='rounded,filled', fillcolor='white')

# Multiscale Feature Fusion
dot.node('Upsample', 'Multiscale Alignment\n(upsample/pool to uniform size)', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Concat', 'Feature Concatenation\nCombine all scales [Res1–Res4]', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Fuse', 'Projection Layer\nFused Embedding V ∈ R^{d_v}', shape='rect', style='filled', fillcolor='white')

# Connections
dot.edge('Input', 'Conv1')
dot.edge('Conv1', 'MaxPool')
dot.edge('MaxPool', 'Res1')
dot.edge('Res1', 'Res2')
dot.edge('Res2', 'Res3')
dot.edge('Res3', 'Res4')

# Multiscale fusion connections
dot.edge('Res1', 'Upsample', style='dashed', label='Low-level features')
dot.edge('Res2', 'Upsample', style='dashed', label='Mid-level features')
dot.edge('Res3', 'Upsample', style='dashed', label='High-level features')
dot.edge('Res4', 'Upsample', style='dashed', label='Semantic features')

dot.edge('Upsample', 'Concat')
dot.edge('Concat', 'Fuse')

# Output
dot.node('Output', 'Visual Embedding\nV ∈ R^{d_v}', shape='rect', style='filled', fillcolor='white')
dot.edge('Fuse', 'Output')

# Layout
dot.attr(concentrate='true')

# Render Diagram
dot.render('Visual_Embedding_Module_ResNet18', view=True)
