"""Run with python3 tests/check.py; requires Hugo on PATH."""

from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import shutil
import subprocess


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.links = []
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)

    def handle_data(self, data):
        self.text.append(data)


root = Path(__file__).resolve().parents[1]
with TemporaryDirectory(prefix="blog-check-") as directory:
    directory = Path(directory)
    content = directory / "content"
    public = directory / "public"
    shutil.copytree(root / "content", content)
    (content / "blog" / "math-check.md").write_text(
        r"""---
title: "Math regression check"
date: 2020-01-01
---
**Markdown still works.**

Inline: $x_i + y_i = z_i$ and $a < b$.

$$
\frac{1}{n} \sum_{i=1}^{n} x_i
$$

$$
\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}
$$
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["hugo", "--contentDir", str(content), "--destination", str(public),
         "--baseURL", "https://example.org/notes/", "--panicOnWarning"],
        cwd=root, check=True,
    )
    assert (public / "index.html").is_file(), "Homepage missing"
    listing = (public / "blog" / "index.html").read_text()
    assert "/notes/blog/math-check/" in listing, "New post missing from blog"
    post = (public / "blog" / "math-check" / "index.html").read_text()
    text = "".join(Page(post).text)
    for math in (r"$x_i + y_i = z_i$", r"$a < b$",
                 r"\frac{1}{n} \sum_{i=1}^{n} x_i",
                 r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}"):
        assert math in text, f"Markdown corrupted math: {math}"
    assert "<strong>Markdown still works.</strong>" in post
    for path in public.rglob("*.html"):
        for link in Page(path.read_text()).links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            assert url.path.startswith("/notes/"), f"Broken deployment prefix: {link}"
            target = public / url.path.removeprefix("/notes/")
            if url.path.endswith("/"):
                target = target / "index.html"
            assert target.is_file(), f"Broken local link in {path.name}: {link}"

print("Site checks passed: pages, new posts, math, and project-site links.")
