from graphviz import Digraph

# Initialize diagram
dot = Digraph('Textual_Embedding_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Input
dot.node('OCR', 'OCR Extracted Text\nT = [t1, t2, ..., tn]', shape='ellipse', style='filled', fillcolor='white')

# Preprocessing
dot.node('Clean', 'Text Cleaning\n- Remove special chars\n- Tokenization\n- Padding/Truncation', shape='box', style='rounded,filled', fillcolor='white')

# Embedding Layer
dot.node('Embed', 'Token Embedding Layer\nConvert tokens → 768-d vectors', shape='box', style='rounded,filled', fillcolor='white')

# Positional Encoding
dot.node('Pos', 'Add Positional Encoding\nRetain word order', shape='box', style='rounded,filled', fillcolor='white')

# Transformer Layers
with dot.subgraph(name='cluster_transformer') as t:
    t.attr(label='DistilBERT Transformer Blocks (6 Layers)', style='dashed', color='black')
    t.node('SelfAttn', 'Self-Attention\nEach token attends to all others')
    t.node('FFN', 'Feed-Forward Network\nCapture higher-level representations')
    t.node('LayerNorm', 'Layer Norm + Residual\nStabilize training')

# Pooling
dot.node('Pool', 'Pooling\nCLS token or mean pooling → Sentence Embedding', shape='box', style='rounded,filled', fillcolor='white')

# Output
dot.node('E', 'Textual Embedding\nE ∈ ℝ^{d_t}', shape='ellipse', style='filled', fillcolor='white')

# Connections
dot.edge('OCR', 'Clean')
dot.edge('Clean', 'Embed')
dot.edge('Embed', 'Pos')
dot.edge('Pos', 'SelfAttn')
dot.edge('SelfAttn', 'FFN')
dot.edge('FFN', 'LayerNorm')
dot.edge('LayerNorm', 'Pool')
dot.edge('Pool', 'E')

# Render
dot.render('Textual_Embedding_Module', view=True)
