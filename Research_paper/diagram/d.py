from graphviz import Digraph

dot = Digraph('Textual_Embedding_Module_DistilBERT', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Input
dot.node('Input', 'Tokenized Text Sequence\nT = [t₁, t₂, ..., tₙ]', shape='rect', style='filled', fillcolor='white')

# Embedding Layer
dot.node('Embed', 'Token Embedding Layer\n(WordPiece → 768-d vectors)', shape='box', style='rounded,filled', fillcolor='white')

# Positional Encoding
dot.node('PosEnc', 'Positional Encoding\nAdds sequential order info', shape='box', style='rounded,filled', fillcolor='white')

# Transformer Layers
dot.node('TF1', 'Transformer Block 1\n(Self-Attention + FFN + Residual)', shape='box', style='rounded,filled', fillcolor='white')
dot.node('TF2', 'Transformer Block 2\nCaptures deeper context', shape='box', style='rounded,filled', fillcolor='white')
dot.node('TF3', 'Transformer Block 3\nEnhances semantic relations', shape='box', style='rounded,filled', fillcolor='white')
dot.node('TF4', 'Transformer Block 4\nModels long-range dependencies', shape='box', style='rounded,filled', fillcolor='white')
dot.node('TF5', 'Transformer Block 5\nRefines contextual meaning', shape='box', style='rounded,filled', fillcolor='white')
dot.node('TF6', 'Transformer Block 6\nFinal contextual representation', shape='box', style='rounded,filled', fillcolor='white')

# Pooling Layer
dot.node('Pool', 'Pooling Layer\n(CLS token / Mean pooling)', shape='box', style='rounded,filled', fillcolor='white')

# Output
dot.node('Output', 'Text Embedding\nE ∈ R^{d_t}\n(Semantic, Syntactic, Sentiment Features)', shape='rect', style='filled', fillcolor='white')

# Edges
dot.edge('Input', 'Embed')
dot.edge('Embed', 'PosEnc')
dot.edge('PosEnc', 'TF1')
dot.edge('TF1', 'TF2')
dot.edge('TF2', 'TF3')
dot.edge('TF3', 'TF4')
dot.edge('TF4', 'TF5')
dot.edge('TF5', 'TF6')
dot.edge('TF6', 'Pool')
dot.edge('Pool', 'Output')

# Connections for clarity
dot.attr(concentrate='true')

# Render Diagram
dot.render('Textual_Embedding_Module_DistilBERT', view=True)
