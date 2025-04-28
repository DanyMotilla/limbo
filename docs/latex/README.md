# LIMBO Documentation

This directory contains the LaTeX documentation for the LIMBO project.

## Installation Instructions

### Linux (Ubuntu/Debian)

1. Install required LaTeX packages:
   ```bash
   # Update package list
   sudo apt-get update

   # Install core packages
   sudo apt-get install -y \
       texlive-latex-base \
       texlive-latex-extra \
       texlive-fonts-recommended \
       latexmk

   # Optional: Install full LaTeX distribution if needed
   # sudo apt-get install texlive-full
   ```

### macOS

1. Install MacTeX:
   ```bash
   # Using Homebrew
   brew install --cask mactex-no-gui

   # Or download from https://www.tug.org/mactex/mactex-download.html
   ```

### Windows

1. Install MiKTeX:
   - Download from https://miktex.org/download
   - Run the installer and select "Install missing packages on the fly"
   - After installation, open MiKTeX Console and update all packages

2. Install Perl (required for latexmk):
   - Download Strawberry Perl from http://strawberryperl.com/
   - Run the installer with default options

## VS Code Setup

1. Install VS Code extensions:
   ```bash
   # LaTeX Workshop
   code --install-extension James-Yu.latex-workshop

   # LaTeX language support
   code --install-extension mathematic.vscode-latex
   ```

2. Configure VS Code settings:
   - Open Command Palette (Ctrl+Shift+P)
   - Search for "Preferences: Open Settings (JSON)"
   - Add/update these settings:
   ```json
   {
     "latex-workshop.latex.outDir": "%DIR%/build",
     "latex-workshop.latex.tools": [
       {
         "name": "latexmk",
         "command": "latexmk",
         "args": [
           "-pdf",
           "-outdir=build",
           "-interaction=nonstopmode",
           "-synctex=1",
           "%DOC%"
         ]
       }
     ],
     "latex-workshop.latex.recipes": [
       {
         "name": "latexmk",
         "tools": ["latexmk"]
       }
     ],
     "latex-workshop.view.pdf.viewer": "tab"
   }
   ```

## Building the Documentation

### Using VS Code

1. Open the `docs/latex` folder in VS Code:
   ```bash
   code /path/to/limbo/docs/latex
   ```

2. Open `main.tex`

3. The documentation will automatically build when you save any .tex file
   - The PDF will be generated in the `build/` directory
   - You can view the PDF directly in VS Code by clicking the preview button

### Using Command Line

```bash
# Navigate to the latex directory
cd docs/latex

# Build the documentation
latexmk -pdf -outdir=build main.tex

# Clean auxiliary files but keep PDF
latexmk -c -outdir=build

# Clean all files including PDF
latexmk -C -outdir=build
```

## Troubleshooting

### Common Issues

1. **Missing Packages**
   ```bash
   # Linux/Ubuntu
   sudo apt-get install texlive-latex-extra

   # macOS
   tlmgr install <package-name>

   # Windows (MiKTeX)
   # Will install automatically when needed
   ```

2. **Unicode Character Errors**
   - Make sure you're using UTF-8 encoding
   - Check that `\usepackage[utf8]{inputenc}` is in `main.tex`

3. **PDF Not Updating**
   - Close the PDF viewer
   - Run `latexmk -C -outdir=build` to clean
   - Rebuild with `latexmk -pdf -outdir=build main.tex`

4. **VS Code Not Building**
   - Check that LaTeX Workshop is installed
   - Ensure the settings.json is correctly configured
   - Try reloading VS Code

### Getting Help

- Check the [LaTeX Workshop Wiki](https://github.com/James-Yu/LaTeX-Workshop/wiki)
- Visit the [LaTeX Project](https://www.latex-project.org/help/)
- Ask in the LIMBO team chat

## Contributing to the Documentation

1. Each section is in its own file in the `sections/` directory
2. Use the provided code highlighting styles for code blocks
3. Place images in the `images/` directory
4. Run `latexmk -c -outdir=build` before committing to clean auxiliary files

## Project Structure

```
latex/
├── .vscode/
│   └── settings.json      # LaTeX Workshop settings
├── build/                # Output directory for PDF and auxiliary files
├── sections/
│   ├── introduction.tex
│   ├── architecture.tex
│   ├── frontend.tex
│   ├── backend.tex
│   ├── api.tex
│   ├── development.tex
│   ├── deployment.tex
│   ├── contributing.tex
│   └── config.tex
├── images/               # Place images here
├── .latexmkrc           # latexmk configuration
├── .gitignore           # LaTeX-specific gitignore
├── main.tex             # Main document
└── README.md            # This file
```

## Features

- Automatic build on save
- PDF preview in VS Code
- Syntax highlighting for code blocks:
  - JavaScript/TypeScript
  - Python
  - JSON
  - YAML
  - Docker
- SyncTeX support (Ctrl+Click to sync between source and PDF)
- Intellisense for LaTeX commands
- Auto-formatting
- Clean auxiliary files automatically on failed builds
