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
