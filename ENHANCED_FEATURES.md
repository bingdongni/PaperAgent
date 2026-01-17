# Enhanced Features Guide

This guide provides detailed information about PaperAgent's enhanced multimodal analysis and LaTeX formatting capabilities.

## Table of Contents

1. [Advanced LaTeX Formatting](#advanced-latex-formatting)
2. [Chart and Graph Analysis](#chart-and-graph-analysis)
3. [Mathematical Formula Recognition](#mathematical-formula-recognition)
4. [Document Structure Analysis](#document-structure-analysis)
5. [Academic Visualizations](#academic-visualizations)
6. [Integration and Usage](#integration-and-usage)

---

## Advanced LaTeX Formatting

### Complex Table Formatting

#### Booktabs Professional Tables

```python
from paperagent.tools.latex_advanced import ComplexTableFormatter
import pandas as pd

formatter = ComplexTableFormatter()

# Create sample data
df = pd.DataFrame({
    'Method': ['Baseline', 'Proposed', 'State-of-art'],
    'Accuracy': [85.2, 92.7, 88.4],
    'F1-Score': [83.1, 91.3, 86.8]
})

# Generate professional table
latex_code = formatter.create_booktabs_table(
    df=df,
    caption="Performance comparison of different methods",
    label="performance",
    bold_header=True
)

print(latex_code)
```

#### Multi-row and Multi-column Tables

```python
# Create table with merged cells
data = [
    ['Model', 'Dataset 1', 'Dataset 2', 'Dataset 3'],
    ['CNN', '85.2', '87.1', '84.6'],
    ['RNN', '82.3', '84.7', '83.1'],
    ['Transformer', '91.2', '93.4', '92.1']
]

merge_cells = [
    (0, 0, 1, 1),  # Merge (row=0, col=0) with rowspan=1, colspan=1
]

latex_code = formatter.create_multirow_table(
    data=data,
    headers=['Model', 'Dataset 1', 'Dataset 2', 'Dataset 3'],
    merge_cells=merge_cells,
    caption="Results across datasets",
    label="datasets"
)
```

### Algorithm Formatting

```python
from paperagent.tools.latex_advanced import AdvancedLaTeXFormatter

formatter = AdvancedLaTeXFormatter()

# Create algorithm
algorithm = formatter.create_algorithm(
    title="Gradient Descent",
    inputs=["$f(x)$: objective function", "$\\alpha$: learning rate"],
    outputs=["$x^*$: optimized parameters"],
    steps=[
        "Initialize $x_0$ randomly",
        "\\FOR{$t = 1$ to $T$}",
        "  Compute gradient $g_t = \\nabla f(x_t)$",
        "  Update $x_{t+1} = x_t - \\alpha g_t$",
        "\\ENDFOR",
        "\\RETURN $x_T$"
    ],
    label="gradient_descent"
)

print(algorithm)
```

### Theorem Environments

```python
# Create theorem
theorem = formatter.create_theorem_environment(
    env_type="theorem",
    content="For any convex function $f$, the gradient descent algorithm converges to the global minimum.",
    label="convex_convergence",
    title="Convergence of Gradient Descent"
)

# Create proof
proof = formatter.create_proof(
    content="""
    Let $f$ be a convex function. By definition of convexity:
    $$f(y) \\geq f(x) + \\nabla f(x)^T(y-x)$$

    Following the gradient descent update rule and applying the convexity condition,
    we can show that the function value decreases monotonically until convergence.
    """
)
```

### Mathematical Formatting

```python
from paperagent.tools.latex_advanced import MathematicalFormatter

math_formatter = MathematicalFormatter()

# Create complex equations
aligned_eqs = math_formatter.create_aligned_equations(
    equations=[
        ("\\mathcal{L}(\\theta)", "\\sum_{i=1}^{n} \\ell(y_i, f(x_i; \\theta))"),
        ("\\nabla \\mathcal{L}", "\\sum_{i=1}^{n} \\nabla \\ell(y_i, f(x_i; \\theta))"),
        ("\\theta^*", "\\arg\\min_{\\theta} \\mathcal{L}(\\theta)")
    ],
    label="loss_function"
)

# Create matrix
import numpy as np
matrix_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_latex = math_formatter.create_matrix(matrix_data, matrix_type="bmatrix")

# Create integral
integral = math_formatter.create_integral(
    expression="e^{-x^2}",
    lower_limit="-\\infty",
    upper_limit="\\infty",
    variable="x"
)
```

---

## Chart and Graph Analysis

### Analyzing Charts

```python
from paperagent.tools.chart_analyzer import ChartAnalyzer

analyzer = ChartAnalyzer()

# Analyze chart image
result = analyzer.analyze_chart("path/to/chart.png")

print(f"Chart Type: {result['chart_type']}")
print(f"Detected Colors: {result['colors']}")
print(f"Text Elements: {result['text_elements']}")
print(f"Insights: {result['insights']}")
```

### Chart Type Detection

The system can automatically detect:
- Bar charts (vertical and horizontal)
- Line charts with trend analysis
- Pie charts with slice counting
- Scatter plots
- Histograms
- Box plots
- Heatmaps

### Extracting Data from Charts

```python
# For bar charts
if result['chart_type'] == 'bar_chart':
    bars = result['insights']['bars']
    print(f"Number of bars: {result['insights']['num_bars']}")
    print(f"Orientation: {result['insights']['orientation']}")

    for i, bar in enumerate(bars):
        print(f"Bar {i+1}: Position={bar['position']}, Height={bar['height']}")

# For line charts
if result['chart_type'] == 'line_chart':
    print(f"Trend: {result['insights']['trend']}")
    print(f"Number of line segments: {result['insights']['num_line_segments']}")
```

---

## Mathematical Formula Recognition

### Recognizing Formulas

```python
from paperagent.tools.formula_recognizer import FormulaRecognizer

recognizer = FormulaRecognizer()

# Recognize formula from image
result = recognizer.recognize_formula("path/to/formula.png")

print(f"Recognized Text: {result['raw_text']}")
print(f"LaTeX Format: {result['latex']}")
print(f"Formula Type: {result['type']}")
print(f"Complexity: {result['structure']['complexity']}")
```

### Formula Structure Analysis

```python
structure = result['structure']

print(f"Has fraction: {structure['has_fraction']}")
print(f"Has integral: {structure['has_integral']}")
print(f"Has summation: {structure['has_sum']}")
print(f"Has derivative: {structure['has_derivative']}")
```

### Extracted Symbols

```python
symbols = result['symbols']

print(f"Operators: {symbols['operators']}")
print(f"Greek letters: {symbols['greek']}")
print(f"Special symbols: {symbols['special']}")
print(f"Variables: {symbols['variables']}")
```

---

## Document Structure Analysis

### Analyzing Document Structure

```python
from paperagent.tools.document_structure import DocumentStructureAnalyzer

analyzer = DocumentStructureAnalyzer()

# Analyze document
result = analyzer.analyze_document("path/to/paper.pdf")

print(f"Title: {result['metadata']['title']}")
print(f"Abstract: {result['metadata']['abstract']}")
print(f"Keywords: {result['metadata']['keywords']}")
```

### Section Extraction

```python
sections = result['sections']

for section in sections:
    print(f"Section: {section['title']}")
    print(f"Level: {section['level']}")
    print(f"Line: {section['line_number']}")
```

### Citation Analysis

```python
from paperagent.tools.document_structure import CitationAnalyzer

citation_analyzer = CitationAnalyzer()

citations = result['citations']
references = result['references']

analysis = citation_analyzer.analyze_citations(citations, references)

print(f"Total citations: {analysis['total_citations']}")
print(f"Total references: {analysis['total_references']}")
print(f"Citation style: {analysis['citation_style']}")
print(f"Citation distribution: {analysis['citation_distribution']}")
```

### Reference Extraction

```python
for ref in result['references']:
    print(f"Authors: {ref['authors']}")
    print(f"Title: {ref['title']}")
    print(f"Year: {ref['year']}")
    print(f"Venue: {ref['venue']}")
    print(f"DOI: {ref['doi']}")
    print("---")
```

---

## Academic Visualizations

### Creating Static Visualizations

```python
from paperagent.tools.visualization import AcademicVisualizer
import numpy as np

visualizer = AcademicVisualizer()

# Scatter plot
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

scatter_path = visualizer.create_scatter_plot(
    x=x,
    y=y,
    title="Correlation Analysis",
    xlabel="Feature X",
    ylabel="Feature Y",
    add_regression=True
)

print(f"Scatter plot saved to: {scatter_path}")
```

### Bar Charts

```python
categories = ['Method A', 'Method B', 'Method C', 'Method D']
values = [85.2, 89.7, 92.3, 88.1]
errors = [2.1, 1.8, 1.5, 2.3]

bar_path = visualizer.create_bar_chart(
    categories=categories,
    values=values,
    title="Performance Comparison",
    xlabel="Methods",
    ylabel="Accuracy (%)",
    error_bars=errors
)
```

### Heatmaps

```python
import pandas as pd

# Create correlation matrix
data = np.random.randn(10, 5)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

heatmap_path = visualizer.create_correlation_matrix(
    df=df,
    title="Feature Correlation Matrix"
)
```

### Box Plots

```python
# Create box plot data
data = [
    np.random.normal(100, 10, 200),
    np.random.normal(105, 12, 200),
    np.random.normal(110, 8, 200)
]

box_path = visualizer.create_box_plot(
    data=data,
    labels=['Group A', 'Group B', 'Group C'],
    title="Performance Distribution",
    ylabel="Score"
)
```

### Interactive Visualizations

```python
from paperagent.tools.visualization import InteractiveVisualizer

interactive_viz = InteractiveVisualizer()

# Create interactive scatter plot
df = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'group': np.random.choice(['A', 'B', 'C'], 200),
    'size': np.random.randint(1, 100, 200)
})

html_path = interactive_viz.create_interactive_scatter(
    df=df,
    x_col='x',
    y_col='y',
    title="Interactive Scatter Plot",
    color_col='group',
    size_col='size'
)

print(f"Interactive plot saved to: {html_path}")
```

### 3D Visualizations

```python
# Create 3D scatter plot
df_3d = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'z': np.random.randn(200),
    'category': np.random.choice(['Type 1', 'Type 2', 'Type 3'], 200)
})

plot_3d_path = interactive_viz.create_3d_scatter(
    df=df_3d,
    x_col='x',
    y_col='y',
    z_col='z',
    title="3D Feature Space",
    color_col='category'
)
```

---

## Integration and Usage

### Using the Enhanced Processor

```python
from paperagent.tools.enhanced_integration import EnhancedMultimodalProcessor

# Initialize processor
processor = EnhancedMultimodalProcessor()

# Or use the singleton instance
from paperagent.tools.enhanced_integration import enhanced_processor
```

### Auto-detect and Process Content

```python
# Automatically detect content type and process
result = processor.process_multimodal_content("path/to/file.pdf")

if result['status'] == 'success':
    print(result['analysis'])
```

### Process PDF Documents

```python
# Deep PDF analysis
pdf_result = processor.analyze_pdf("research_paper.pdf")

# Access different components
metadata = pdf_result['analysis']['metadata']
text_analysis = pdf_result['analysis']['text']['analysis']
tables = pdf_result['analysis']['tables']
images = pdf_result['analysis']['images']
```

### Process Charts

```python
chart_result = processor.analyze_chart("bar_chart.png")

if chart_result['status'] == 'success':
    analysis = chart_result['analysis']
    print(f"Chart Type: {analysis['chart_type']}")
    print(f"Insights: {analysis['insights']}")
```

### Create LaTeX Documents

```python
# Create algorithm
algorithm_latex = processor.create_algorithm_latex(
    title="K-Means Clustering",
    inputs=["Data points $X = \\{x_1, ..., x_n\\}$", "Number of clusters $k$"],
    outputs=["Cluster assignments $C$"],
    steps=[
        "Initialize $k$ centroids randomly",
        "\\REPEAT",
        "  Assign each point to nearest centroid",
        "  Update centroids as mean of assigned points",
        "\\UNTIL convergence"
    ],
    label="kmeans"
)

# Create theorem
theorem_latex = processor.create_theorem_latex(
    env_type="theorem",
    content="The k-means algorithm converges to a local minimum of the objective function.",
    label="kmeans_convergence",
    title="K-Means Convergence"
)
```

### Generate Visualizations

```python
import numpy as np

# Create scatter plot
scatter_data = {
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'title': "Feature Relationship",
    'xlabel': "Feature 1",
    'ylabel': "Feature 2"
}

plot_path = processor.create_visualization(
    plot_type='scatter',
    data=scatter_data,
    add_regression=True
)

print(f"Plot saved to: {plot_path}")
```

---

## Complete Workflow Example

Here's a complete example combining multiple features:

```python
from paperagent.tools.enhanced_integration import EnhancedMultimodalProcessor
import pandas as pd
import numpy as np

# Initialize
processor = EnhancedMultimodalProcessor()

# 1. Analyze research paper
print("Analyzing research paper...")
paper_analysis = processor.analyze_document("paper.pdf")

sections = paper_analysis['analysis']['sections']
citations = paper_analysis['analysis']['citations']

print(f"Found {len(sections)} sections and {len(citations)} citations")

# 2. Extract and analyze chart from paper
print("\nAnalyzing figure from paper...")
chart_result = processor.analyze_chart("paper_figure1.png")

print(f"Chart type: {chart_result['analysis']['chart_type']}")

# 3. Generate visualization for new results
print("\nGenerating new visualization...")
results_df = pd.DataFrame({
    'Method': ['Baseline', 'Proposed', 'SOTA'],
    'Accuracy': [85.2, 92.7, 88.4],
    'F1': [83.1, 91.3, 86.8]
})

viz_path = processor.create_visualization(
    plot_type='bar',
    data={
        'categories': results_df['Method'].tolist(),
        'values': results_df['Accuracy'].tolist(),
        'title': 'Performance Comparison',
        'ylabel': 'Accuracy (%)'
    }
)

# 4. Create LaTeX table for results
print("\nGenerating LaTeX table...")
table_latex = processor.create_complex_table_latex(
    df=results_df,
    caption="Experimental results comparison",
    label="results",
    use_booktabs=True
)

print(table_latex)

# 5. Create algorithm description
print("\nCreating algorithm...")
algo_latex = processor.create_algorithm_latex(
    title="Proposed Method",
    inputs=["Input data $X$"],
    outputs=["Predictions $Y$"],
    steps=[
        "Preprocess data",
        "Extract features",
        "Apply model",
        "Post-process results"
    ],
    label="proposed_method"
)

print(algo_latex)

print("\nWorkflow completed successfully!")
```

---

## Best Practices

### For LaTeX Formatting

1. **Use booktabs for tables**: Professional appearance
2. **Label everything**: Makes referencing easier
3. **Use descriptive captions**: Help readers understand content
4. **Consistent formatting**: Maintain style throughout document

### For Image Analysis

1. **High-quality images**: Better recognition accuracy
2. **Clear charts**: Ensure text is readable
3. **Proper preprocessing**: Enhance contrast for better OCR

### For Visualizations

1. **Choose appropriate chart type**: Match data characteristics
2. **Use color-blind friendly palettes**: Ensure accessibility
3. **Include error bars**: Show uncertainty
4. **High DPI**: 300 DPI for publications

---

## Troubleshooting

### LaTeX Compilation Issues

```python
# Check LaTeX installation
import subprocess
result = subprocess.run(['pdflatex', '--version'], capture_output=True)
print(result.stdout.decode())
```

### Image Recognition Problems

```python
# Verify pytesseract installation
import pytesseract
print(pytesseract.get_tesseract_version())

# Set custom path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Missing Dependencies

```bash
# Install all enhanced features dependencies
pip install opencv-python scikit-learn scikit-image
pip install plotly matplotlib seaborn
pip install pytesseract pdf2image pdfplumber
```

---

## API Reference

For detailed API documentation, see:
- [LaTeX Advanced API](../api/latex_advanced.md)
- [Chart Analyzer API](../api/chart_analyzer.md)
- [Formula Recognizer API](../api/formula_recognizer.md)
- [Document Structure API](../api/document_structure.md)
- [Visualization API](../api/visualization.md)

---

**Happy Research Writing! 📝🔬📊**
