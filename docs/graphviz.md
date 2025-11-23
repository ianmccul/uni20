# Graphviz Basics

Graphviz is a suite of open-source tools for drawing graphs specified in the **DOT** language.  
In *uni20*, we will use Graphviz to visualize the **dependency DAG** of asynchronous tasks — for example, to inspect scheduler state, trace coroutine dependencies, or debug deadlocks.

## What Is Graphviz?

Graphviz reads text files written in the **DOT** format that describe nodes, edges, and attributes, then produces visual layouts in various formats such as **SVG**, **PNG**, or **PDF**.

Example DOT file (`dag.dot`):

```dot
digraph uni20_dag {
    node [shape=box, style=filled, fontname="monospace"];
    A [label="Task A: add", fillcolor="lightgreen"];
    B [label="Task B: mul", fillcolor="lightgray"];
    A -> B;
}
````

Render it using Graphviz’s command-line tool:

```bash
dot -Tsvg dag.dot -o dag.svg
```

This generates an SVG you can open in any web browser.

---

## Installing Graphviz

### Linux (Ubuntu / Debian)

```bash
sudo apt install graphviz
```

### macOS (Homebrew)

```bash
brew install graphviz
```

### Windows

Use the official MSI installer from: [https://graphviz.org/download/](https://graphviz.org/download/)

---

## Viewing and Exploring Graphs

* **Static images**

  * Use `dot -Tpng` or `dot -Tsvg` to produce a file you can open directly.
  * Example:

    ```bash
    dot -Tpng dag.dot -o dag.png
    ```

* **Interactive exploration**

  * `xdot`: an interactive viewer that lets you zoom, pan, and inspect node attributes.

    ```bash
    sudo apt install xdot
    xdot dag.dot
    ```

* **VS Code plugin**

  * The *Graphviz (dot) language support* extension allows editing and previewing `.dot` files interactively inside VS Code.


  ## Additional Graphviz Viewers and Tools

  ###  **Desktop Viewers**

  | Tool                 | Platform              | Notes                                                                                              |
  | -------------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
  | **xdot**             | Linux / macOS         | Classic interactive viewer; live updates if file changes. Supports zoom, pan, and node inspection. |
  | **KGraphViewer**     | Linux (KDE)           | Polished UI for `.dot` files; integrates with KDE.                                                 |
  | **ZGRViewer**        | Cross-platform (Java) | Open-source zoomable viewer; handles very large graphs.                                            |
  | **OmniGraffle**      | macOS (commercial)    | Can import DOT files; good for presentation-quality diagrams.                                      |
  | **yEd Graph Editor** | Cross-platform        | Imports/export DOT; has advanced layout options.                                                   |
  | **Gephi**            | Cross-platform        | General-purpose network visualization platform; can import DOT after simple conversion.            |

  ---

  ### **Web-Based and Interactive Viewers**

  | Tool               | Access                                                                         | Notes                                                                                                         |
  | ------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
  | **GraphvizOnline** | [dreampuf.github.io/GraphvizOnline](https://dreampuf.github.io/GraphvizOnline) | Simple online editor — paste DOT, view instantly.                                                             |
  | **Edotor**         | [edotor.net](https://edotor.net)                                               | Lightweight web viewer, live updates as you type.                                                             |
  | **Viz.js**         | JS library ([github.com/mdaines/viz.js](https://github.com/mdaines/viz.js))    | Compiles Graphviz to WebAssembly — render DOT directly in browsers or notebooks.                              |
  | **WebGraphviz**    | [webgraphviz.com](http://www.webgraphviz.com)                                  | Basic viewer; no interactivity but quick for testing.                                                         |
  | **d3-graphviz**    | [github.com/magjac/d3-graphviz](https://github.com/magjac/d3-graphviz)         | Integrates Graphviz layout with D3.js for dynamic/animated graphs — a future candidate for *uni20* live DAGs. |

  ---

  ### **IDE and Editor Integrations**

  | Editor             | Plugin                            | Description                                         |
  | ------------------ | --------------------------------- | --------------------------------------------------- |
  | **VS Code**        | *Graphviz (dot) Language Support* | Syntax highlighting and side-by-side preview.       |
  | **JetBrains IDEs** | *Graphviz Preview*                | Renders `.dot` graphs inside the editor.            |
  | **Emacs / Vim**    | Syntax modes available            | Highlighting and quick preview via external viewer. |

  ---

  ### **Other Layout Engines (Alternative to `dot`)**

  These are part of Graphviz itself — they control how the layout is computed:

  | Engine    | Type                    | Description                                           |
  | --------- | ----------------------- | ----------------------------------------------------- |
  | **dot**   | Hierarchical            | Best for DAGs and dependency trees.                   |
  | **neato** | Spring model            | Useful for undirected graphs.                         |
  | **fdp**   | Force-directed          | Similar to `neato`, better for large, organic graphs. |
  | **sfdp**  | Scalable force-directed | Handles thousands of nodes efficiently.               |
  | **twopi** | Radial layout           | Displays levels radiating from a root node.           |
  | **circo** | Circular layout         | Good for cyclic structures.                           |

  Use via:

  ```bash
  neato -Tsvg graph.dot -o graph.svg
  ```
---

## 📘 Further Reading

* Official Graphviz site: [https://graphviz.org](https://graphviz.org)
* DOT language reference: [https://graphviz.org/doc/info/lang.html](https://graphviz.org/doc/info/lang.html)
* Interactive Graphviz viewer (`xdot`): [https://pypi.org/project/xdot/](https://pypi.org/project/xdot/)
