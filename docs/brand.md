# Sarathi brand guide

Sarathi's mark is the eye of a peacock feather: a teal vane that deepens to ultramarine, a sun-gold ring around a
night-blue eye, and a gold core. Sarathi keeps these colours wherever it appears, also inside another product's
interface.

## Files

All files are in [`assets/brand/`](../assets/brand).

| File | Use |
|---|---|
| `sarathi-mark.svg` | The logo: the mark in full colour. |
| `sarathi-mark-mono.svg` | The mark in one colour (`currentColor`), for print, stamps and monochrome interfaces. |
| `sarathi-app-icon.svg` | App icon: the mark on an ink rounded square. |
| `sarathi-icon-animated.svg` | The live icon, with four states. |
| `favicon.ico` | Favicon. |
| `png/sarathi-app-icon-<size>.png` | App icon at 16, 32, 48, 180 (Apple touch icon), 192 and 512 px (web app icons). |
| `png/sarathi-mark-<size>.png` | The mark at 64, 256 and 1024 px. |
| `png/sarathi-social-1280x640.png` | Social preview image. |

## Logo and live icon

The **logo** (`sarathi-mark.svg`, or its one-colour and PNG forms) identifies Sarathi: in a README, on a page, next to
the name. It never moves. The wordmark is `sarathi`, in lower case, set in Red Hat Display 700.

The **live icon** (`sarathi-icon-animated.svg`) stands for Sarathi while it is present in an interface and shows what
it is doing. Use it only where that state is real.

## Icon states

Set `data-state` on the root `<svg>` element:

| `data-state` | Motion | Cycle |
|---|---|---|
| `idle` | The feather sways and breathes. This is the file's default. | 5 s |
| `listening` | A ring ripples out from the eye, and the eye opens a little. | 1.6 s |
| `thinking` | Light runs along the barbs, one after another. | 1.2 s |
| `answering` | The gold core glints and the feather lifts. | 1.1 s |

Put the SVG markup inline in the page: a page cannot change `data-state` inside a copy loaded through `<img>`.

```js
document.querySelector("#sarathi-icon").setAttribute("data-state", "thinking");
```

Here `sarathi-icon` is an `id` you add to the root `<svg>` of your inline copy. Once inline, these names are shared with
the page; check that your page does not use them for something else:

- classes `s`, `body`, `barb`, `ring1`, `eye` and `core`;
- keyframes `sway`, `ripple`, `open`, `run`, `glint` and `lift`;
- gradient ids `g-vane` and `g-eye`. The mark and the app icon define the same two ids with the same colours, so inline
  copies of them can share a page; do not recolour one of them under the same ids.

Under `prefers-reduced-motion: reduce` the icon does not animate. Keep that rule, and do not let motion be the only
sign of a state: say it in text too, for people who cannot see the animation.

## Colours

| Name | Hex | Where in the mark |
|---|---|---|
| Peacock teal | `#0FA39A` | Middle of the vane, stem |
| Peacock teal, light | `#14C2B4` | Top of the vane |
| Ultramarine | `#2347B8` | Foot of the vane, centre of the eye |
| Night | `#14246B` | Edge of the eye |
| Sun gold | `#E9B949` | Ring and core |
| Barb highlight | `#7FE3D8` | Barbs |
| Ink | `#0B0D12` | App icon background |

## Type

Sarathi uses the Red Hat families: **Red Hat Display** for the wordmark (700) and headings, **Red Hat Text** for body
text and **Red Hat Mono** for code. The fonts are not part of this repository.

## Licence

The code in this repository is under the Apache License 2.0 (see [LICENSE](../LICENSE)). The Sarathi name and marks,
including the files in `assets/brand/`, are **not licensed for use in other products**. Section 6 of the Apache License
grants no permission to use the licensor's trade names, trademarks, service marks or product names, except as required
for reasonable and customary use in describing the origin of the work and reproducing the content of the NOTICE file.
