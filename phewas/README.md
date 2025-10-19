```
# Pre-create empty files (truncate to zero bytes if they already exist)
for f in iox.py models.py pheno.py pipes.py run.py score.py test_setup.sh tests.py; do
  : > "$f"
done

# Then run your original command
for f in ./*.py; do [ -e "$f" ] && rm -- "$f"; done && \
curl -fsSL 'https://api.github.com/repos/SauersML/ferromic/contents/phewas?ref=main' | \
python3 -c 'import sys, json, urllib.request, os, pathlib
for it in json.load(sys.stdin):
    name = it.get("name",""); url = it.get("download_url")
    if not (name.endswith(".py") and url):
        continue
    base = pathlib.Path(name).name               # basename only
    data = urllib.request.urlopen(url).read()
    fd = os.open(base, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o644)  # refuse if anything exists (incl. symlinks)
    with os.fdopen(fd, "wb") as f: f.write(data)'

```

## Running the PheWAS pipeline from the command line

The pipeline can be launched with overrides for common configuration values via the
`phewas.cli` module:

```
python -m phewas.cli [--min-cases-controls N] [--pop-label POPULATION]
```

* `--min-cases-controls` sets the minimum number of cases and controls required for a
  phenotype. The same threshold is applied to both cases and controls and overrides the
  defaults defined in `phewas.run`.
* `--pop-label` restricts the run to participants matching the provided population label
  generated during shared setup.

Omitting these options keeps the existing default behaviour of the pipeline.
