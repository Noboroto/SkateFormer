from importlib import import_module as _import_module

_real = _import_module(__name__ + ".torchlight")  # torchlight.torchlight
for _k in dir(_real):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_real, _k)
