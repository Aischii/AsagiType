# Roadmap after the demo review

On 2026-11-20 the owner shared a 28-second demo of another manga translator: a dark UI, probably Chisa Translator. These decisions came out of a grilling session about it. We borrow its layout and workflow ideas, not its look: AsagiType keeps the Windows 2000 grey theme with asagi title bars.

## What the demo showed

1. Detect: "Detected 4 bubbles", with red numbered boxes on the Original pane. Box 4 was a false hit on the "©2025 Lanaadjie" credit line.
2. Translate: a modal with a progress bar and Cancel ("Translating 4 bubbles (gemini-3.5-flash-lite)…"). The balloons were cleaned to white, then lettered in English.
3. Editing a bubble: a cyan box with handles appeared on both panes. Changing the font size and the box width refit the text live.
4. The user deleted the false hit by hand, and the bubble count dropped from 4 to 3.
5. Layout: synced Original and Translated panes, page thumbnails, an "All Pages" scope, and right panel tabs Properties / Layers / Healing / Image. The typography and bubble-style controls sit in the Properties tab, and a status bar shows the bubble count.

## Decisions

| Topic | Decision |
| --- | --- |
| View | Side-by-side Original and Translated panes with synced zoom and scroll. A selection shows on both. View can switch back to a single pane |
| Box editing | Drag to move, 8 handles to resize, arrow keys nudge (Shift for 10 px). Lettering refits live |
| Typography | Font picker per balloon (overrides the role's font), size, 5 style presets, bold, italic, case (UPPER / Title / lower), and alignment. Spacing, rotation and stroke are not planned yet |
| Presets | Stored per project. A new project copies the defaults from Settings |
| Cleaning | Solid fill of the text area, with the balloon colour sampled and the fill clipped to the bubble outline so art and borders survive. Kept as a separate clean layer in `.at2k`. LaMa comes later for textured areas |
| Pipeline | One **Auto** button runs only the missing stages (detect + OCR, translate, clean) for This page or All pages, with a progress dialog and Cancel. Detect and Translate stay available as separate buttons |
| False hits | An **Ignore** flag: the box is kept but greyed out, and translate, clean and lettering skip it. Delete also works. No automatic guessing |
| Pages list | Thumbnails with the page id and a status tag (empty, detected, translated, cleaned) |
| Bubble style | Per-balloon fill colour (sampled by default), outline, and a shape preset: rectangle, rounded or ellipse |
| Layers tab | Lists the source, the clean layer and each text layer, with show/hide and lock |
| Healing brush | A fill brush (sampled colour) and a clone stamp (Alt-click sets the source). Strokes go into the clean layer and can be undone. LaMa later |
| Theme | Windows 2000 only. No dark variant |
| Not planned | Node editing of bubble outlines, text rotation, letter and line spacing, stroke |

## `.at2k` format v2

Each change below comes with a test, and v1 files must still open.

- `Balloon`:
  - `ignore: bool`
  - `font: Option<String>` (overrides the role's font)
  - `bold`, `italic`
  - `case: upper | title | lower | as_typed`
  - `align: left | center | right`
  - `preset: Option<u8>`
  - `fill: Option<[u8; 4]>`
  - `outline: Option<{color, width}>`
  - `shape: rect | rounded | ellipse`
- `Manifest.presets`: 5 named typography presets.
- `clean/<page>.png` entries: the clean layer, RGBA, at page size. They are written only when a page has been cleaned.

## Releases

### 0.3: editing

- Synced side-by-side panes, with a toggle back to a single pane.
- Move and resize handles, nudging, and live refit.
- Typography panel: font, size, presets, bold, italic, case and alignment. Needs `.at2k` v2.

### 0.4: cleaning and pipeline

- Solid-fill cleaning into the clean layer, clipped to the bubble. The Translated pane composites source, then clean layer, then lettering.
- The Auto button with This page / All pages scope, a progress dialog and Cancel.
- The Ignore flag.

### 0.5: panels

- Page thumbnails with status tags.
- Bubble fill, outline and shape controls.
- Layers tab.
- Healing brush: fill and clone stamp.

### 0.6: export

- JPG and CBZ export of the composited pages.
- Layered PSD export: source, clean layer and editable text.
- Model weights release and download on first run.

## Out of scope for now

- Copying the demo's dark theme or its exact controls.
- Vector node editing of bubbles.
- LaMa inpainting. It is still planned, after FFT support in candle.
