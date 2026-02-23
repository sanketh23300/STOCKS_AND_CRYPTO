# Root-level entry point for Streamlit Community Cloud.
# Streamlit Cloud looks for this file by default.
# It executes the dashboard module from the dashboard/ subfolder.

import os, sys, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

runpy.run_path(
    os.path.join(ROOT, "dashboard", "app.py"),
    run_name="__main__",
)
