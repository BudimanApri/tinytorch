import os
from tinytorch.engine import Value
from tinytorch.viz import draw_graph

def main():
    # Construct the worked-example graph from Milestone 1, Example 2
    # c = a*b; e = c+b; d = e*e
    a = Value(-2.0)
    b = Value(3.0)
    c = a * b          # -6
    e = c + b          # -3
    d = e * e          # 9
    d.backward()

    print(f"Graph constructed. d.data = {d.data}, d.grad = {d.grad}")
    
    # Create the assets folder if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    # Render the graph
    dot = draw_graph(d)
    
    # Save the source dot file and render as png
    # Note: this requires the Graphviz system dot binary to be on PATH.
    try:
        output_path = dot.render("assets/worked_example", cleanup=True)
        print(f"Graph successfully rendered to {output_path}")
    except Exception as ex:
        print(f"Could not render graph image because Graphviz system binary was not found or failed: {ex}")
        print("Note: You must install Graphviz and add the 'dot' binary to your PATH to render images.")

if __name__ == "__main__":
    main()
