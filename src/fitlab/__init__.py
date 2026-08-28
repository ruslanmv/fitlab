"""fitlab — will it run on my setup? Fit-check + benchmark wizard for local LLMs."""
try:
    from importlib.metadata import version as _v
    __version__ = _v("fitlab")
except Exception:  # editable/dev
    __version__ = "0.1.0"
