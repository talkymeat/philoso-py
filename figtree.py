from trees import Tree, NonTerminal, Terminal
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from IPython.display import display, Image
from os import mkdir, listdir
# from cairosvg import svg2png
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM

begin = r"""\documentclass{standalone}
\usepackage[utf8]{inputenc}
\usepackage{qtree}


\begin{document}


"""

end = """


\end{document}"""

class FigTree:
    """A utility for displaying tree diagrams (via LaTeX and qtree) in Jupyter
    notebooks

    Attributes
    ----------

    tree (Tree):
        A tree in the trees.Tree format, which will be displayed as an SVG
    """
    file_num = 0

    def __init__(self, tree, hide_files = True):
        if isinstance(tree, Tree):
            self.tree = tree
            if not hide_files and not ".fig_tmp" in listdir():
                mkdir(".fig_tmp")
            self.dir = TemporaryDirectory()
            fpath = self.dir.name if hide_files else ".fig_tmp"
            self.fname = str(
                Path(f"{fpath}") /
                f"figtree_{FigTree.file_num}"
            )
            FigTree.file_num += 1
        else:
            raise AttributeError("FigTree needs to be initialised with a Tree")

    def _make_latex(self):
        return begin + self.tree.to_LaTeX() + end

    def _write_latex(self, tex):
        with open(f"{self.fname}.tex", 'w') as texfile:
            texfile.write(tex)

    def _make_svg(self):
        subprocess.run([
            "./tex2svg.sh",
            f"{self.fname}.tex",
            f"{self.fname}.svg"
        ])

    def _make_png(self):
        #drawing = svg2rlg(f"{self.fname}.svg")
        #renderPM.drawToFile(drawing, f"{self.fname}.png", fmt='PNG')
        #svg2png(url=f"{self.fname}.svg", write_to=f"{self.fname}.png")
        subprocess.run([
            "cairosvg",
            f"{self.fname}.svg",
            "-o",
            f"{self.fname}.png"
        ])

    def _show_png(self):
        display(Image(f"{self.fname}.png"))

    def show(self):
        self._write_latex(self._make_latex())
        self._make_svg()
        self._make_png()
        self._show_png()

def showtree(tree: Tree, hide_files = True):
    FigTree(tree, hide_files).show()
