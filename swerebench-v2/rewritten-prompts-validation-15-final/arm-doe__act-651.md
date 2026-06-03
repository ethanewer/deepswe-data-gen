# arm-doe__act-651

- repo: ARM-DOE/ACT
- language: python
- difficulty: medium

## Rewritten Prompt

Make the histogram plotting methods forward optional NumPy histogram keyword arguments so callers can control the underlying binning behavior. `HistogramDisplay.plot_stacked_bar_graph`, `HistogramDisplay.plot_stairstep_graph`, and `HistogramDisplay.plot_heatmap` should all accept a `hist_kwargs` argument, defaulting to an empty dict, and pass those extra keywords through to `numpy.histogram` or `numpy.histogram2d` as appropriate.

Preserve the existing method signatures and behavior for the other arguments: `field`/`xfield`/`yfield`, `dsname`, bin controls, `subplot_index`, `set_title`, `density`, `sortby_field`, `sortby_bins`, `set_shading`, and `**kwargs`. The methods should continue returning a dictionary with the computed histogram data, including the `'histogram'` entry and any existing auxiliary outputs such as bin edges.

## Preserved Requirements

- HistogramDisplay.plot_stacked_bar_graph must accept hist_kwargs=dict() and forward it to numpy.histogram.
- HistogramDisplay.plot_stairstep_graph must accept hist_kwargs=dict() and forward it to numpy.histogram.
- HistogramDisplay.plot_heatmap must accept hist_kwargs=dict() and forward it to numpy.histogram2d.
- The existing public method names must remain importable and callable: HistogramDisplay.plot_stacked_bar_graph, HistogramDisplay.plot_stairstep_graph, HistogramDisplay.plot_heatmap.
- The existing parameters and defaults for field/xfield/yfield, dsname, bins/x_bins/y_bins, sortby_field, sortby_bins, subplot_index, set_title, density, set_shading, and **kwargs should be preserved.
- The methods should continue returning a dict containing at least a 'histogram' entry, with any existing ancillary data such as bin edges preserved.
- The added kwargs should control the underlying NumPy histogram computation without changing other plotting behavior.

## Removed Noise

- Issue-template phrasing and conversational introduction.
- The internal implementation hint pointing to a specific line of code.
- Repository/file path references.
- PR/test-style references and metadata.
- Redundant duplication of the same interface notes across sections.

## Risk Notes

- The notes imply `hist_kwargs` is defaulted with `dict()`, so behavior should match that callable default even if a safer default pattern is used internally.
- The exact shape and contents of the returned dict are only partially specified; preserve any existing keys beyond 'histogram' and bin edges.
- For stacked/stairstep variants, ensure forwarded histogram kwargs do not interfere with any existing sorting or subplot behavior.

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
