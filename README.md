# Notes & ideas

A minimal personal blog: Home, Blog, and individual Markdown posts.
Built with Hugo, with MathJax for Obsidian-style math.

## Preview locally

Install [Hugo](https://gohugo.io/installation/) **0.168.0 or later**, then run:

```sh
hugo server
```

Open <http://localhost:1313>. Edits rebuild the site and refresh your browser.

## Edit the homepage

Edit `content/_index.md` for your introduction and the blog's purpose.
Keep the `<!--more-->` divider: the quote above it sits beside your photo,
and the introduction below it appears underneath.
Change the site name in `hugo.toml`. Your photo is `static/images/profile.jpg`;
its placement and dimensions are set in `layouts/home.html`.

## Add a post

Create `content/blog/my-post.md`:

```markdown
---
title: "My post"
date: 2026-10-25
description: "An optional one-sentence preview."
---

Paste your Obsidian Markdown here.

Inline math: $E = mc^2$.

$$
\int_0^1 x^2\,dx = \frac{1}{3}
$$
```

Posts appear on the Blog page automatically, newest first. Add `draft: true`
to hide unfinished posts; use `hugo server --buildDrafts` to preview them.
Dates in the future are not published until a build runs after that date.
Delete `content/blog/welcome.md` when you no longer need the example.

Use standard Markdown links and images. Copy image files into `static/images/`
and reference them with relative paths such as `../../images/diagram.png`.
Obsidian-only wikilinks, embeds, and callouts are not converted.
MathJax needs an internet connection, including during local previews.
Escape literal dollar signs outside math as `\\$` so they survive Markdown
processing without starting an equation.

## Publish on GitHub Pages

Push this repository to GitHub on the `main` branch, then select
**Settings > Pages > Source > GitHub Actions** and run the **Publish blog**
workflow. Later pushes to `main` rebuild and publish automatically.
The workflow sets the correct URL, including a repository subpath.
For a custom domain, also update `baseURL` in `hugo.toml`.

## Check or build

```sh
python3 tests/check.py
hugo --minify --panicOnWarning
```

The build is written to `public/`, which is not committed.
