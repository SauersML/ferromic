Get the phewas code:
```
# Mirror the repo's phewas/ directory locally
rm -rf -- phewas && mkdir -p phewas && \
curl -fsSL 'https://api.github.com/repos/SauersML/ferromic/git/trees/main?recursive=1' | \
python3 -c 'import sys, json, os, pathlib, urllib.request
root = "phewas/"
tree = json.load(sys.stdin)["tree"]
for it in tree:
    p = it.get("path","")
    if not (p.startswith(root) and it.get("type") == "blob"):
        continue
    dest = pathlib.Path(p)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/SauersML/ferromic/main/{p}"
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())'
```

## Running the PheWAS pipeline from the command line

The pipeline can be launched with overrides for common configuration values via the
`phewas.cli` module:

```
python3 -m phewas.cli [--min-cases-controls N] [--pop-label POPULATION]
```

* `--min-cases-controls` sets the minimum number of cases and controls required for a
  phenotype. The same threshold is applied to both cases and controls and overrides the
  defaults defined in `phewas.run`.
* `--pop-label` restricts the run to participants matching the provided population label
  generated during shared setup.

Omitting these options keeps the existing default behaviour of the pipeline.

Run extras like this (custom control example): 
```
python3 -m phewas.extra.custom_control_followup
```
