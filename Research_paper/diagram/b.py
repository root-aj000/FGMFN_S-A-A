from graphviz import Digraph

dot = Digraph('Advertisement_Input_Module_Detailed', format='png')
dot.attr(rankdir='TB', bgcolor='white', fontname='Helvetica', fontsize='12')

# Top: Raw Input
dot.node('Raw', 'Raw Advertisement Image\n(contains embedded text & graphics)', shape='rect', style='filled', fillcolor='white')

# Image Processing cluster
with dot.subgraph(name='cluster_image') as img:
    img.attr(label='Image Processing', style='dashed', color='black', fontsize='11')
    img.node('Resize', 'Resize to Standard Resolution\n(224x224x3)', shape='box')
    img.node('Normalize', 'Pixel Normalization\n(subtract mean / divide std)', shape='box')
    img.node('Augment', 'Data Augmentation \nrotation, flip, color jitter', shape='box')
    # order
    img.edge('Resize', 'Normalize')
    img.edge('Normalize', 'Augment')

# OCR Extraction cluster
with dot.subgraph(name='cluster_ocr') as ocr:
    ocr.attr(label='OCR Extraction', style='dashed', color='black', fontsize='11')
    ocr.node('OCRTool', 'OCR (PaddleOCR)\nExtract embedded text from image', shape='box')

# Text Processing cluster
with dot.subgraph(name='cluster_text') as txt:
    txt.attr(label='Text Processing', style='dashed', color='black', fontsize='11')
    txt.node('Clean', 'Clean Text\n(remove special chars, fix artifacts)', shape='box')
    txt.node('Tokenize', 'Tokenization\n(split into subwords / tokens)', shape='box')
    txt.node('Pad', 'Padding / Truncation\n(uniform sequence length)', shape='box')
    txt.edge('Clean', 'Tokenize')
    txt.edge('Tokenize', 'Pad')

# Outputs
dot.node('ImgOut', 'Preprocessed Image Tensor\nI ∈ R^{H×W×3}', shape='rect', style='filled', fillcolor='white')
dot.node('TxtOut', 'Tokenized Text Sequence\nT = [t₁, t₂, ..., tₙ]', shape='rect', style='filled', fillcolor='white')

# Edges: Raw -> Image processing and OCR
dot.edge('Raw', 'Resize', label='visual path')
dot.edge('Raw', 'OCRTool', label='text extraction')

# OCR -> Text processing
dot.edge('OCRTool', 'Clean')
# Connect image processing final step to image output
dot.edge('Augment', 'ImgOut')
# Connect text processing final step to text output
dot.edge('Pad', 'TxtOut')

# Optional: show that OCR also feeds into image preprocessing for text-aware augmentation (commentary-style edge)
dot.edge('OCRTool', 'Augment', style='dotted', label='( text-aware augmentation)')

# Layout tweaks
dot.attr(concentrate='true')

# Render (file saved as Advertisement_Input_Module_Detailed.png)
dot.render('Advertisement_Input_Module_Detailed', view=True)
