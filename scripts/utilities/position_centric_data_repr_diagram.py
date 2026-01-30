from graphviz import Digraph
import os

def generate_position_centric_diagram():
    """
    Generates a Graphviz diagram to visualize the position-centric data representation
    used in MetaSpliceAI's error modeling and meta-model stages.
    """
    dot = Digraph('PositionCentricDataRepresentation', comment='MetaSpliceAI Data Model')
    dot.attr(rankdir='TB', splines='ortho', newrank='true')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # --- Central Data Representation ---
    with dot.subgraph(name='cluster_core') as c:
        c.attr(label='Core Position-Centric Representation', style='filled', color='lightgrey', fillcolor='whitesmoke', fontname='Helvetica', fontsize='14')
        c.node('position', 'Genomic Position\n(e.g., chr1:12345)', shape='diamond', fillcolor='cornflowerblue', fontcolor='white')
        c.node('sequence', 'Flanking DNA Sequence\n(~500nt upstream + ~500nt downstream)', shape='record', fillcolor='lightblue')
        c.edge('position', 'sequence', label='is centered within')

    # --- Error Model Branch ---
    with dot.subgraph(name='cluster_error_model') as c:
        c.attr(label='Error Modeling Stage', style='filled', color='lightcoral', fillcolor='seashell', fontname='Helvetica', fontsize='14')
        c.node('error_input', 'Input:\nSequence Context', shape='note', fillcolor='lightyellow')
        c.node('error_model', 'Error Model\n(e.g., Transformer)', shape='cylinder', fillcolor='coral')
        c.node('error_goal', 'Goal: Diagnostic Analysis\n(Identify sequence motifs causing errors)', shape='ellipse', fillcolor='mistyrose')
        c.edge('error_input', 'error_model')
        c.edge('error_model', 'error_goal', label='to achieve')

    # --- Meta-Model Branch ---
    with dot.subgraph(name='cluster_meta_model') as c:
        c.attr(label='Meta-Model Stage', style='filled', color='mediumseagreen', fillcolor='honeydew', fontname='Helvetica', fontsize='14')
        c.node('meta_input', 'Input:\nSequence + Probability Vectors', shape='note', fillcolor='lightyellow')
        c.node('meta_model', 'Meta-Model\n(e.g., XGBoost)', shape='cylinder', fillcolor='mediumaquamarine')
        c.node('meta_goal', 'Goal: Predictive Recalibration\n(Refine base model scores)', shape='ellipse', fillcolor='palegreen')
        c.edge('meta_input', 'meta_model')
        c.edge('meta_model', 'meta_goal', label='to achieve')

    # --- Connections from Core to Branches ---
    dot.edge('sequence', 'error_input', lhead='cluster_error_model', label='provides context for')
    dot.edge('sequence', 'meta_input', lhead='cluster_meta_model', label='provides context for')

    # Add a title
    dot.attr(label='MetaSpliceAI: Shared Position-Centric Data Representation', labelloc='t', fontsize='20', fontname='Helvetica-Bold')

    # Save and render the diagram
    output_filename = 'position_centric_data_representation'
    # Infer the project root by going up one level from the script's directory
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    dot.render(output_path, format='png', view=False, cleanup=True)
    print(f"Diagram saved as {output_path}.png")

if __name__ == '__main__':
    generate_position_centric_diagram()
