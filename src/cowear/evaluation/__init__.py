"""Paper evaluation and result-artifact helpers.

The executable evaluator is intentionally not imported here.  Keeping this
package initializer lightweight prevents importing PyTorch and the full data
pipeline when callers only need the protocol metrics.
"""
