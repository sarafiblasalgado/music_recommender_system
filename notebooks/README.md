# Notebooks

This project's pipeline lives in `main.py` / `src/` rather than notebooks,
so results are fully reproducible from the command line. This folder is
left as a place to explore `results/metrics.csv` and the trained models
interactively if you prefer notebook-based exploration, e.g.:

```python
import sys; sys.path.insert(0, "..")
from src import data_loading as dl
# ...
```
