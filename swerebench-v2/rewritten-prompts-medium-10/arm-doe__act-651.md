# arm-doe__act-651

- repo: ARM-DOE/ACT
- language: python
- difficulty: medium

## Rewritten Prompt

The histogram plotting methods should support passing extra keyword arguments through to the underlying NumPy histogram functions. In particular, stacked bar graphs, stairstep histograms, and heatmap histograms should accept additional histogram options and use them when computing the histogram data.

Make sure this works consistently for the one-dimensional stacked bar and stairstep plots as well as the two-dimensional heatmap plot, while preserving their existing behavior and returned histogram data.

## Preserved Requirements

- Extra keyword arguments must be forwarded to the underlying NumPy histogram computation.
- The stacked bar histogram plot should accept and use additional histogram options.
- The stairstep histogram plot should accept and use additional histogram options.
- The heatmap histogram plot should accept and use additional histogram options.
- Existing behavior and returned histogram data should be preserved.
- The behavior should apply to both one-dimensional and two-dimensional histogram plots.

## Removed Noise

- Issue-template wording and informal report language
- Direct quoted code line from the original report
- External URLs and PR/test references
- Repository/file-location details
- Generated interface notes and signature metadata
- Implementation hint about a specific function call
- Mentions of confidence, difficulty, and other benchmark metadata

## Risk Notes

- The original report is specific about missing kwargs in one histogram call, but the supplied interface notes broaden the requirement to multiple histogram plot methods.
- The prompt should not overconstrain how the arguments are named internally, only that additional histogram options are accepted and used.

## Original Prompt

Missing kwargs in plot_stacked_bar_graph
### Description

I just saw this in the code and wanted to report it.  I think we need to pass in kawrgs to the below line in plot_stacked_bar_graph.

'''
my_hist, bins = np.histogram(xdata.values.flatten(), bins=bins, density=density)
'''

## Original Interface

Method: HistogramDisplay.plot_stacked_bar_graph(self, field, dsname=None, bins=10, sortby_field=None, sortby_bins=None, subplot_index=(0,), set_title=None, density=False, hist_kwargs=dict(), **kwargs)
Location: act/plotting/histogramdisplay.py – class HistogramDisplay
Inputs:
- **field** (str): name of the variable in the dataset to histogram.
- **dsname** (str | None, optional): name of the datastream; if None, ACT infers it.
- **bins** (int | array‑like, optional, default 10): number of bins or explicit bin edges for the histogram.
- **sortby_field** (str | None, optional): field used to sort stacked bars; when provided a second‑dimensional histogram is built.
- **sortby_bins** (array‑like | None, optional): custom bin edges for the sorting dimension.
- **subplot_index** (tuple, optional, default (0,)): subplot location within the figure.
- **set_title** (str | None, optional): title for the subplot.
- **density** (bool, optional, default False): if True, compute a probability density instead of raw counts.
- **hist_kwargs** (dict, optional, default {}): additional keyword arguments passed directly to ``numpy.histogram`` (e.g., ``range``, ``weights``).
- **kwargs** (dict): further keyword arguments forwarded to ``matplotlib.pyplot.bar``.
Outputs:
- Returns a ``dict`` containing at least the key ``'histogram'`` with the computed histogram counts (or density) as a NumPy array, plus any ancillary data (e.g., bin edges) produced by the method.
Description: Generates a one‑dimensional (or stacked two‑dimensional) bar‑graph histogram of ``field`` and returns the numeric histogram data. The added ``hist_kwargs`` allows callers to control the underlying NumPy histogram computation.

Method: HistogramDisplay.plot_stairstep_graph(self, field, dsname=None, bins=10, sortby_field=None, sortby_bins=None, subplot_index=(0,), set_title=None, density=False, hist_kwargs=dict(), **kwargs)
Location: act/plotting/histogramdisplay.py – class HistogramDisplay
Inputs:
- Same parameter list as ``plot_stacked_bar_graph`` (field, dsname, bins, sortby_field, sortby_bins, subplot_index, set_title, density, hist_kwargs, **kwargs).
Outputs:
- Returns a ``dict`` whose ``'histogram'`` entry holds the histogram array computed for the stairstep plot, together with any additional data produced by the method.
Description: Produces a stairstep (line) representation of a histogram for ``field``. The signature now includes ``hist_kwargs`` to forward extra arguments to ``numpy.histogram``, matching the stacked‑bar version.

Method: HistogramDisplay.plot_heatmap(self, xfield, yfield, dsname=None, x_bins=None, y_bins=None, subplot_index=(0,), set_title=None, density=False, set_shading='auto', hist_kwargs=dict(), **kwargs)
Location: act/plotting/histogramdisplay.py – class HistogramDisplay
Inputs:
- **xfield**, **yfield** (str): names of the two variables to form a 2‑D histogram.
- **dsname** (str | None, optional): datastream identifier.
- **x_bins**, **y_bins** (array‑like | None, optional): explicit bin edges for the x and y dimensions; if ``None`` defaults are derived from the data.
- **subplot_index**, **set_title**, **density**, **set_shading**, **kwargs**: same semantics as in other plot methods.
- **hist_kwargs** (dict, optional, default {}): extra keyword arguments forwarded to ``numpy.histogram2d`` (e.g., ``range``).
Outputs:
- Returns a ``dict`` containing at least ``'histogram'`` – a 2‑D NumPy array of counts/densities – and the bin edge arrays used for the heatmap.
Description: Constructs a 2‑D histogram heatmap of ``xfield`` versus ``yfield``. The added ``hist_kwargs`` permits fine‑grained control over the underlying ``numpy.histogram2d`` call (such as limiting the range).
