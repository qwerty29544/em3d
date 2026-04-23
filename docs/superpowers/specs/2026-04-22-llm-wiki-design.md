# LLM Wiki — Design Spec
**Date:** 2026-04-22
**Domain:** Physical mathematics — efficient Fourier methods for solving volume integral equations (VIE) in acoustics and electrodynamics
**Purpose:** Dissertation research wiki — literature synthesis + evolving thesis + writing support + code support
**Stage:** Mid-stage dissertation

---

## 1. Directory Structure

```
wiki/                          ← Obsidian vault root (git repo root)
│
├── raw/                       ← Immutable source documents (LLM reads, never writes)
│   ├── papers/                ← PDF academic papers
│   ├── notes/                 ← Author's own draft notes, handwritten scans
│   ├── web/                   ← Obsidian Web Clipper markdown exports
│   ├── docs/                  ← .docx files (reports, advisor feedback, drafts)
│   ├── slides/                ← .pptx files (presentations, lecture slides)
│   └── assets/                ← Downloaded images, figures, diagrams
│
├── literature/                ← Paper/source summaries (LLM-written)
├── concepts/                  ← Mathematical entities, methods, theorems
├── research/                  ← Author's thesis, proofs, conjectures, open questions
├── code/                      ← Solver notes, experiment results, implementation pages
├── writing/                   ← Draft dissertation sections, outlines, literature review passages
│
├── index.md                   ← Master catalog of all wiki pages (LLM-maintained)
├── log.md                     ← Append-only operation history (LLM-maintained)
└── CLAUDE.md                  ← Schema: wiki conventions and LLM workflows
```

**Conventions:**
- `raw/` is the source of truth — never modified after ingestion
- All five wiki zones are owned entirely by the LLM
- The vault is a git repository; version history is free
- All LLM responses and wiki content are in **Russian**, except: mathematical notation, code, citation keys, and proper names

---

## 2. Wiki Page Conventions

### Frontmatter (all pages)

```yaml
---
title: <human-readable title>
zone: literature | concepts | research | code | writing
tags: [tag1, tag2, ...]
sources: [literature/citation-key, ...]   # omit if not applicable
status: draft | stable | needs-review
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### Naming convention
`zone/kebab-case-title.md` — file names always use **English kebab-case** regardless of content language. Page titles in frontmatter and headings may be in Russian.
Examples: `concepts/volume-integral-equation.md`, `literature/vainikko-2000.md`, `code/gmres-fft-solver.md`

### Per-zone page structure

**`literature/`** — one page per source:
- Full citation
- Abstract summary (in Russian)
- Key contributions
- Relevant equations or theorems (with page references)
- Relationship to dissertation thesis
- Links to related `concepts/` pages

**`concepts/`** — one page per mathematical entity/method:
- Definition (formal + intuitive)
- Key properties and theorems
- Known algorithms and complexity
- Open problems
- Links to `literature/` pages that define/use it
- Links to `code/` pages that implement it

**`research/`** — free-form pages:
- Author's own arguments, proof sketches, conjectures
- Open questions and hypotheses
- Evolving thesis statement (dedicated page: `research/thesis.md`)
- Contradictions and unresolved tensions flagged by the LLM

**`code/`** — one page per solver or experiment:
- Purpose and algorithm used
- Parameters and configuration
- Complexity analysis
- Results and observations
- Links to `concepts/` pages it implements
- Links to source files if applicable

**`writing/`** — dissertation prose:
- Draft sections with clear headings
- Literature review passages
- Academic Russian register
- Ready to copy into the dissertation document

---

## 3. Core Workflows

### 3.1 Ingest
Trigger: "ingest `raw/papers/author-year.pdf`" (or docx, pptx, web, notes)

Steps:
1. Extract text/content using appropriate skill (pdf, docx, pptx skills)
2. Discuss key takeaways with the author
3. Create a `literature/` summary page
4. Create or update relevant `concepts/` pages
5. Update `research/thesis.md` if the source bears on the thesis
6. Flag contradictions with existing wiki content explicitly
7. Append entry to `log.md`: `## [YYYY-MM-DD] ingest | Source Title`
8. Update `index.md`

A single source typically touches 5–15 wiki pages.

### 3.2 Query
Trigger: author asks a question about the research domain

Steps:
1. Read `index.md` to find relevant pages
2. Drill into relevant pages
3. Synthesize answer in Russian with wiki page citations (`[[page-name]]`)
4. Offer to file the answer as a new `research/` or `writing/` page if valuable

### 3.3 Writing Support
Trigger: "напиши черновик раздела о..." (draft a section on...)

Steps:
1. Identify relevant `literature/` and `concepts/` pages via `index.md`
2. Draft prose in academic Russian
3. Save output to `writing/kebab-case-title.md`
4. Link the writing page from `index.md`

### 3.4 Code Support
Trigger: author describes a solver or experiment

Steps:
1. Create a `code/` page with purpose, algorithm, parameters, results
2. Link to the relevant `concepts/` pages
3. Update `index.md`
4. Offer to draft complexity analysis or comparison with related methods

### 3.5 Lint
Trigger: "проверь вики" (lint the wiki)

The LLM checks for:
- Orphan pages (no inbound links)
- Stale claims superseded by newer sources
- Concepts mentioned in `literature/` but lacking their own `concepts/` page
- Missing cross-links between related pages
- Data gaps suggesting new sources to seek
- Internal contradictions between pages

Reports findings in Russian with a prioritized action list.

---

## 4. CLAUDE.md Schema Contents

The `CLAUDE.md` file at vault root governs all future sessions. It contains:

1. **Language directive** — все ответы, резюме и содержимое вики на русском языке (except math notation, code, citation keys, proper names)
2. **Wiki structure reference** — zone descriptions, naming conventions, frontmatter schema
3. **Ingest protocol** — step-by-step procedure per source type
4. **Query protocol** — index-first navigation, cite by `[[wiki-link]]`, offer to file answers
5. **Writing protocol** — academic Russian register, save to `writing/`, pull from `literature/` and `concepts/`
6. **Code protocol** — page structure, link to concepts, document parameters and results
7. **Lint protocol** — checklist and reporting format
8. **Log format** — `## [YYYY-MM-DD] operation | title`
9. **Index format** — table of contents by zone, one-line description per page

---

## 5. Obsidian Configuration

- **Attachment folder:** `raw/assets/` — so downloaded images land in the right place
- **Recommended plugins:**
  - Dataview — query pages by frontmatter (e.g., all `concepts/` pages tagged `FFT`)
  - Obsidian Web Clipper — browser extension for clipping articles to `raw/web/`
  - Marp (optional) — render `writing/` pages as slide decks for presentations
- **Graph view** — best way to see wiki shape: hubs, orphans, zone clusters

---

## 6. Index and Log Formats

### index.md structure
```markdown
# Wiki Index

## Literature
- [[literature/vainikko-2000]] — Метод дискретных источников для VIE (Vainikko, 2000)

## Concepts
- [[concepts/volume-integral-equation]] — Объёмное интегральное уравнение: определение и свойства

## Research
- [[research/thesis]] — Текущая формулировка диссертационного тезиса

## Code
- [[code/gmres-fft-solver]] — GMRES+FFT решатель для акустического рассеяния

## Writing
- [[writing/literature-review-vie-methods]] — Черновик: обзор литературы по методам VIE
```

### log.md entry format
```markdown
## [2026-04-22] ingest | Vainikko (2000) — Discretization of VIE
- Создана страница: literature/vainikko-2000
- Обновлены страницы: concepts/volume-integral-equation, concepts/nyström-method
- Противоречий не обнаружено
```

---

## 7. Source Type Handling

| Source type | Raw location | Extraction method | Primary wiki output |
|-------------|-------------|-------------------|---------------------|
| PDF paper | `raw/papers/` | pdf skill | `literature/` page |
| DOCX report/feedback | `raw/docs/` | docx skill | `literature/` or `research/` page |
| PPTX slides | `raw/slides/` | pptx skill | `literature/` page with slide structure |
| Web article | `raw/web/` | read markdown | `literature/` page |
| Author notes | `raw/notes/` | read text | `research/` page |

---

## 8. Success Criteria

- Every ingested source produces at least one `literature/` page and updates relevant `concepts/` pages
- `index.md` always reflects current wiki state
- `log.md` has an entry for every operation
- All responses from the LLM are in Russian
- The wiki is navigable in Obsidian via graph view and internal links
- Writing outputs are dissertation-ready prose in academic Russian
- Code pages link back to the mathematical concepts they implement
