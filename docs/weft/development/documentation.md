# Maintain the bilingual handbook

The handbook uses MkDocs Material and static language files.
English pages use `topic.md`; Chinese counterparts use `topic.zh.md`.
They share code, images and a corresponding navigation structure.

## Build and preview

From the repository root:

```sh
python -m pip install -r third_party/weft/requirements-docs.txt
python -m mkdocs build --strict -f mkdocs.yml --site-dir /tmp/weft-handbook
python -m mkdocs serve -f mkdocs.yml
```

English builds at the site root; Chinese builds under `/zh/`.
The language selector links corresponding chapters.
Strict builds check internal pages and anchors.

## Update both editions

1. Edit both language versions for the same behavior and code.
2. Link internal pages by their neutral name, such as `../compiler/runtime.md`;
   the localization plugin resolves the active language.
3. Use source links for repository files outside the handbook.
4. Update both navigation labels in `mkdocs.yml` when adding a chapter.
5. Open both rendered pages and check code, tables, figures and narrow screens.

Describe available behavior directly. Put installation requirements and
measurement settings beside the commands or results they explain.

## Figures and specifications

Shared assets live in `docs/weft/assets`. Paper figures have bilingual captions
and an entry in [figure sources](../experiments/results.md).
Public measured CSV tables live in Bench's `docs/paper`; execution outputs remain
in Bench's ignored `output` directory.

Detailed semantic and pass contracts live in `docs/specification`.
The handbook teaches usage and structure. Approved semantic changes update
the corresponding contract and implementation together.

## Publish

`Build Triton and Weft` runs the strict bilingual build.
Its default-branch handbook job deploys through GitHub Pages to
`https://tianchen-ai.github.io/triton-weft/`.
