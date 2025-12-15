import os
from graphviz import Digraph

def generate_meta_model_diagram():
    """
    Generates a Graphviz diagram to visualize the concept of the meta-model
    as an adaptive learning layer on top of a base splice predictor.
    """
    dot = Digraph('MetaModelConcept', comment='MetaSpliceAI Meta-Model Architecture')
    dot.attr(rankdir='TB', splines='ortho', newrank='true', compound='true')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Title
    dot.attr(label='MetaSpliceAI: Meta-Model as an Adaptive Learning Layer\n\n', labelloc='t', fontsize='20', fontname='Helvetica-Bold')

    # --- Input Data ---
    dot.node('input_dna', 'DNA Sequence', shape='folder', fillcolor='whitesmoke')

    # --- Base Model Layer ---
    with dot.subgraph(name='cluster_base') as c:
        c.attr(label='Base Predictor Layer (Static)', style='filled', color='lightblue', fillcolor='azure', fontname='Helvetica', fontsize='14')
        c.node('base_model', 'Base Splice Predictor\n(e.g., SpliceAI)', shape='cylinder', fillcolor='aliceblue')
        c.node('base_preds', 'Initial Predictions\n(Raw Scores)', shape='note', fillcolor='lightyellow')

    # --- Contextual Features ---
    dot.node('context_features', 'Rich Contextual Features\n(Genomic, Transcriptomic, Regulatory)', shape='folder', fillcolor='whitesmoke')

    # --- Meta-Model Layer ---
    with dot.subgraph(name='cluster_meta') as c:
        c.attr(label='Adaptive Meta-Model Layer', style='filled', color='mediumseagreen', fillcolor='honeydew', fontname='Helvetica', fontsize='14')
        c.node('meta_input', 'Enriched Input', shape='note', fillcolor='lightyellow')
        c.node('meta_model', 'Meta-Model\n(e.g., XGBoost)', shape='cylinder', fillcolor='mediumaquamarine')
        c.node('final_preds', 'Recalibrated & Adapted Predictions', shape='ellipse', fillcolor='palegreen')

    # --- Final Output/Benefits ---
    dot.node('benefits', 'Improved Biological Insights\n- Enhanced Accuracy\n- Handles Complex Splicing\n- Context-Aware', shape='star', fillcolor='gold')

    # --- Define the data flow ---
    dot.edge('input_dna', 'base_model')
    dot.edge('base_model', 'base_preds', label='generates')
    
    dot.edge('base_preds', 'meta_input', ltail='cluster_base', lhead='cluster_meta')
    dot.edge('context_features', 'meta_input', lhead='cluster_meta', label='enriches')
    
    dot.edge('meta_input', 'meta_model')
    dot.edge('meta_model', 'final_preds', label='refines to')
    dot.edge('final_preds', 'benefits', label='leads to')

    # --- Save and render the diagram ---
    output_filename = 'meta_model_concept'
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    dot.render(output_path, format='png', view=False, cleanup=True)
    print(f"Diagram saved as {output_path}.png")

if __name__ == '__main__':
    generate_meta_model_diagram()
