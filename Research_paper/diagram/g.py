from graphviz import Digraph

# Initialize diagram
dot = Digraph('Multi_Task_Learning_Module', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Input
dot.node('M', 'Joint Multimodal Embedding\nM ∈ ℝ^{d_s}', shape='ellipse', style='filled', fillcolor='white')

# Task heads
dot.node('Sentiment', 'Sentiment Classification\nSoftmax: Positive/Negative/Neutral', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Emotion', 'Emotion Prediction\nTrust, Joy, Excitement, etc.', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Theme', 'Theme Categorization\nFashion, Electronics, Promotion', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Engagement', 'Engagement Prediction\nRegression: Likes, Shares, CTR', shape='box', style='rounded,filled', fillcolor='white')
dot.node('Trust', 'Trust Assessment\nBinary / Probabilistic', shape='box', style='rounded,filled', fillcolor='white')

# Output
# dot.node('Y', 'Multi-Task Predictions\nY = {y_sentiment, y_emotion, y_theme, y_engagement, y_trust}', shape='ellipse', style='filled', fillcolor='white')

# Connections
dot.edge('M', 'Sentiment')
dot.edge('M', 'Emotion')
dot.edge('M', 'Theme')
dot.edge('M', 'Engagement')
dot.edge('M', 'Trust')

# dot.edge('Sentiment', 'Y')
# dot.edge('Emotion', 'Y')
# dot.edge('Theme', 'Y')
# dot.edge('Engagement', 'Y')
# dot.edge('Trust', 'Y')

# Render
dot.render('Multi_Task_Learning_Module', view=True)
