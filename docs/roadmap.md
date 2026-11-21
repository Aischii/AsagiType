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

### 0.3: editing (done)

- Synced side-by-side panes, with a toggle back to a single pane.
- Move and resize handles, nudging, and live refit.
- Typography panel: font, size, presets, bold, italic, case and alignment. Needs `.at2k` v2.

### 0.4: cleaning and pipeline (done)

- Solid-fill cleaning into the clean layer, clipped to the bubble. The Translated pane composites source, then clean layer, then lettering.
- The Auto button with This page / All pages scope, a progress dialog and Cancel.
- The Ignore flag.

### 0.5: live cleaning, content-aware fill and rotation (done)

Decided on 2026-11-20 after the owner tested 0.4. The owner found three problems:

- Text on the sky was filled with one flat colour.
- White rectangles covered art around captions.
- The clean layer went stale: deleting a box did not bring the original pixels back.

| Topic | Decision |
| --- | --- |
| Live cleaning | The auto clean layer is recomputed from the current boxes about 150 ms after any edit (delete, move, resize, Ignore, shape). Deleting a box restores the original there. The auto layer is regenerated on open and export, not stored. Healing-brush strokes (0.6) will go in a separate paint layer that is saved |
| Mask | Inside a detected bubble with a plain interior, the whole interior is filled as before. On art, sky or screentone, only the text pixels plus a 2–3 px halo are masked |
| Content-aware fill | Pure-Rust inpainting, chosen per area. Telea-style fast marching handles smooth areas (sky, gradients). Exemplar patch synthesis handles textured areas (screentone). LaMa comes later |
| Rotation detection | A projection search over −45° to 45° in 1° steps, on a binarised crop that also handles white text on dark |
| Rotation threshold | Detect + OCR measures every box. Angles under 3° count as level |
| Rotation use | OCR runs on the straightened crop. English is lettered at the same angle. Cleaning follows the tilted text through the glyph mask. The editor box stays upright |
| Bubble style | Per-balloon shape (Auto erase, rectangle, rounded or ellipse), fill colour and outline. A shape paints a new bubble on the clean layer |
| Outline protection | A dark stroke is erased only if it lies wholly inside the text box plus 3 px. Strokes that cross the box edge are kept: bubble outlines, panel borders, art. This fixed an open bubble whose outline got erased |
| Letters touching the outline | The merged stroke is kept and the box gets an amber "needs touch-up" mark. It is fixed with the healing brush in 0.6 |
| Mask view | View > Show clean mask draws the pixels that will be erased in red |
| Bubble segmentation | The Manga109 YOLO11n-seg balloon model, ported to candle, gives exact bubble masks that clip the interior fill. It was trained on Manga109, which is licensed for academic use only, so it is not bundled or released: the app loads it from the user's models folder and credits it in About. Without it, the stroke-protection rule applies |
| Order | First stroke protection, live cleaning, the glyph mask, inpainting, rotation and the mask view. Then the segmentation model |

### 0.6: panels

- Page thumbnails with status tags.
- Layers tab.
- Healing brush: fill and clone stamp, on the saved paint layer.

Decisions (2026-11-21):

| Topic | Decision |
| --- | --- |
| Delivery | Four parts, each tested and tried in the app before a commit: 1) undo and the paint layer format, 2) thumbnails, 3) Layers tab, 4) healing brush |
| Format | Version 3. `Page.paint` names `paint/<id>.png` (RGBA, page size), written on save when the page has paint. `Manifest.view` holds the Layers tab state. `Balloon.touch_fixed` records a cleared touch-up mark. Orphaned `paint/` and `clean/` entries are dropped on save |
| Undo | One history per project, 100 steps, in memory only. Ctrl+Z, Ctrl+Y (or Ctrl+Shift+Z) and an Edit menu; undo jumps to the page it changed. A drag or a field edit (click in to click out) is one step. Detect, Translate and Auto runs are one step per page. Paint strokes are steps too |
| Thumbnails | The source scan, built in the background on open. Tags are computed: Detected, Translated, untranslated count, amber Needs touch-up |
| Touch-up | Clears when brush strokes cover the marked spot, or with Mark fixed in Properties |
| Layers tab | Fixed stack: Source, Auto clean, Paint, Text. Eye and opacity per layer, saved in the project, view only: exports always use every layer. Text expands to its balloons (hide, lock) |
| Healing brush | Fill (J): a translucent red stroke while dragging, filled in the background on release from what you see minus lettering, Smooth or Texture chosen as in the cleaner. Clone stamp (S): Alt+click sets the source, aligned. Round, 2–200 px, `[` and `]`. No eraser: undo covers it |
| Shortcuts | V select, D draw balloon, J fill, S clone stamp; off while typing |
| Tilted captions | Fixed during part 1 after a broken clean on a tilted caption strip over dark art. Tilted Auto boxes (1° or more) are cleaned in their own straightened frame: the cleaner finds the strip of paper around the text line, cleans only that strip with its paper colour, and maps the result back. Without a strip it cleans the straightened box |

### 0.7: export

- JPG and CBZ export of the composited pages.
- Layered PSD export: source, clean layer and editable text.
- Model weights release and download on first run.

## Out of scope for now

- Copying the demo's dark theme or its exact controls.
- Vector node editing of bubbles.
- LaMa inpainting. It is still planned, after FFT support in candle.
