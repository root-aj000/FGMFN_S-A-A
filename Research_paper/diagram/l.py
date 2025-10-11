from graphviz import Digraph

# Initialize diagram
dot = Digraph('Multi_Task_Learning_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Input: Joint multimodal embedding
dot.node('Mms', 'Joint Multimodal Embedding\nM_{ms} ∈ ℝ^{d_s}', shape='ellipse', style='filled', fillcolor='white')

# Multi-Task Heads
dot.node('Sentiment', 'Sentiment Classification\nSoftmax', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Emotion', 'Emotion Detection\nSoftmax', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Theme', 'Theme Categorization\nSoftmax', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Engagement', 'Engagement Prediction\nRegression (Likes, Shares, CTR)', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Trust', 'Trustworthiness Assessment\nBinary / Probabilistic', shape='box', style='rounded,filled', fillcolor='white')

# Output
# dot.node('Output', 'Predictions\nY = {y_{sentiment}, y_{emotion}, y_{theme}, y_{engagement}, y_{trust}}', shape='ellipse', style='filled', fillcolor='white')

# Connections
dot.edge('Mms', 'Sentiment')
dot.edge('Mms', 'Emotion')
dot.edge('Mms', 'Theme')
dot.edge('Mms', 'Engagement')
dot.edge('Mms', 'Trust')

# dot.edge('Sentiment', 'Output')
# dot.edge('Emotion', 'Output')
# dot.edge('Theme', 'Output')
# dot.edge('Engagement', 'Output')
# dot.edge('Trust', 'Output')

# Render
dot.render('Multi_Task_Learning_Module', view=True)
