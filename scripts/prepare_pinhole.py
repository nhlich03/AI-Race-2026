"""Backwards-compatible shim. The real logic now lives in prepare.py / colmap_io.py.

run_all.py imports prepare_source from here and expects the prepared source-dir
PATH back (a string), so we unwrap prepare.prepare_source's dict result.
"""
from prepare import prepare_source as _prepare_source


def prepare_source(src_train_dir, out_dir):
    return _prepare_source(src_train_dir, out_dir)["source"]
