# Data

This directory is empty by design. The analysis expects the deposit from:

> Wang, A. & Horowitz, A. (2026). Data from: Dogs (*Canis familiaris*) distinguish
> conspecific emotional chemosignals. Dryad. doi:10.5061/dryad.bzkh189pt

Expected contents:

```
Part_1.xlsx                habituation-dishabituation trials
Part_2.xlsx                sample-point behavioural coding
statistical_analyses.R     the authors' original analysis
README.md
```

At the time of writing, Dryad's file endpoints (`/api/v2/files/{id}/download` and
`/downloads/file_stream/{id}`) require an OAuth bearer token and return HTTP 401/403
to anonymous requests, so the files could not be retrieved programmatically. Download
them through a browser and place them here, then set `use_synthetic=False` and pass
this directory to `load_deposit`.

Column names in the notebook's `COLUMN_MAP` are **inferred from the paper's Methods**,
not read off the workbooks. Reconcile them against the real headers before trusting any
output.
