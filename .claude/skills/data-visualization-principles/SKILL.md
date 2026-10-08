---
name: data-visualization-principles
description: Contains specific principles of data visualization for scientific analysis for reproducible plotting.
---

Use these guidelines whenever you produce data visualizations.

Color palettes:
It is important to always use color-blind friendly color palettes for visualization. Colors should be complementary but still distinguishable. Here is a sample hexcode-based color palette that follows these guidelines:
Okabe-Ito 8-Color Hex Codes
• Black: #000000
• Orange: #E69F00
• Sky Blue: #56B4E9
• Bluish Green: #009E73
• Yellow: #F0E442
• Blue: #0072B2
• Vermillion: #D55E00
• Reddish Purple: #CC79A7

Saving and editing:
Files should be saved in svg format within the file directory of the project for easy access. These need to be editable in Adobe Illustrator with high resolution for publishing. Considering adding specifications such as the following to ensure they are easy to edit:
    plt.rcParams['font.family'] = 'Arial'

Histograms and bar charts:
Ensure to maintain consistent sizing of bar charts, and typically start these at 0 for consistent and meaningful comparisons. Keep charts flat to the reader. The area of color is how the reader will compare differences.

Line charts:
Zoom in on the key trend for these. Keep common axes between subplots that are making the same type of comparison. Consider using min and max of the values being plotted as rough bounds, but always leave a bit of space between a point and the edge of the plot.

Labels:
Always label x and y axis with descriptive and concise labels based on the data. Use units, and make these publication-ready (do not use '-' characters for instance)

Statistics:
Don't overassume Gaussian distributions. Generally plot histograms of data distributions first so that the user can decide if normal distributions are appropriate. Always use error bars and consider uncertainty. 

Colormaps:
Use diverging colormaps when there is deviation across a middle value (such as log fold difference). Use gradient colormaps when visualizing continuous data (from low to high). Use discrete colormaps when visualizing discrete datatypes/categories.

Legends:
Always ensure that there is a legend, and position this consistently in the top right of figure panels. 