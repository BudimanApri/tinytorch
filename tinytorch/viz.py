from graphviz import Digraph

def trace(root):
    """
    Builds a set of all nodes and edges in the computational graph.
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_graph(root, format='png', rankdir='LR'):
    """
    Renders a computational graph using graphviz.
    
    Args:
        root (Value): The root Value node (usually the loss/output).
        format (str): Output format ('png', 'svg', etc.).
        rankdir (str): Graph layout direction ('LR' for left-to-right, 'TB' for top-to-bottom).
        
    Returns:
        Digraph: The graphviz Digraph object.
    """
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        
        # Build label displaying data and grad
        label = f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}"
        
        # Sleek node styling
        dot.node(
            name=uid,
            label=label,
            shape='record',
            style='filled,rounded',
            fillcolor='#EBF5FB',  # light blue
            color='#2E86C1',      # blue border
            fontname='Helvetica',
            fontsize='10'
        )
        
        if n._op:
            # Operation node (circle)
            op_uid = uid + n._op
            dot.node(
                name=op_uid,
                label=n._op,
                shape='circle',
                style='filled',
                fillcolor='#FDEDEC',  # light red
                color='#CB4335',      # red border
                fontname='Helvetica-Bold',
                fontsize='10'
            )
            # Connect the op node to the value node it produced
            dot.edge(op_uid, uid, color='#5D6D7E')
            
    for n1, n2 in edges:
        # Connect input node to the operation node of the output node
        dest = str(id(n2)) + n2._op if n2._op else str(id(n2))
        dot.edge(str(id(n1)), dest, color='#5D6D7E')
        
    return dot
