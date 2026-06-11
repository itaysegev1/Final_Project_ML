# legacy/

Original helper scripts that shipped with the *Epicurious — Recipes with
Rating and Nutrition* Kaggle dataset. They are kept for provenance only and
are **not imported anywhere** in the project:

- `recipe.py` — the BeautifulSoup/urllib scraper the dataset author used to
  build the data from live Epicurious HTML. Importing it requires `bs4`.
- `utils.py` — a one-hot helper (`sublists_to_binaries`) that is redundant
  here because `epi_r.csv` already ships the binary tag columns.

See the module docstring of `src/data_foundation.py` for why neither is used
by the pipeline.
