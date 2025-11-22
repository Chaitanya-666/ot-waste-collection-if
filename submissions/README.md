# Submission Documents

This folder contains the project submission documents in both Markdown (`.md`) and LaTeX (`.tex`) formats.

## Document Formats

*   **Markdown (`.md`)**: These files are for easy viewing in any text editor or on platforms like GitHub. They provide a clean and readable version of the project report, dataset description, and results sheet.

*   **LaTeX (`.tex`)**: These files are provided for compiling professional-looking PDF documents. They contain the same content as their Markdown counterparts but are formatted for high-quality typesetting.

## Viewing the Documents

### Markdown Files (`.md`)

You can open the `.md` files (`project_report.md`, `dataset.md`, `result_sheet.md`) with any standard text editor or a Markdown viewer.

### LaTeX Files (`.tex`)

To view the `.tex` files as intended, you need to compile them into PDF documents. This requires a LaTeX distribution to be installed on your system (e.g., TeX Live, MiKTeX, MacTeX).

Once you have a LaTeX distribution installed, you can compile each `.tex` file into a PDF using a command-line terminal. For example, to compile the project report, navigate to this directory (`submissions`) in your terminal and run:

```bash
pdflatex project_report.tex
```

This will generate a `project_report.pdf` file. Repeat this process for the other `.tex` files:

```bash
pdflatex dataset.tex
pdflatex result_sheet.tex
```

If you run the `pdflatex` command twice, it will ensure that all cross-references and table of contents are correctly generated.
