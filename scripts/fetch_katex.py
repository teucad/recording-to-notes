"""Download KaTeX (math rendering) from the npm registry into app/static/vendor/katex."""
import io
import json
import tarfile
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "app" / "static" / "vendor" / "katex"
WANTED = ("package/dist/katex.min.js", "package/dist/katex.min.css")


def main() -> None:
    meta = json.load(urllib.request.urlopen("https://registry.npmjs.org/katex/latest", timeout=30))
    data = urllib.request.urlopen(meta["dist"]["tarball"], timeout=120).read()
    (DEST / "fonts").mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name in WANTED:
                target = DEST / Path(member.name).name
            elif member.name.startswith("package/dist/fonts/") and member.name.endswith(".woff2"):
                target = DEST / "fonts" / Path(member.name).name
            else:
                continue
            target.write_bytes(tar.extractfile(member).read())
    print(f"KaTeX {meta['version']} -> {DEST}")


if __name__ == "__main__":
    main()
