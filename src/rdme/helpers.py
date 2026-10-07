from __future__ import annotations

from collections.abc import Sequence

import torch

""" Small analysis utilities, used by the plotting stages of the scripts rather than by the
model. One function so far; the file exists so that an array utility does not have to live in
a physics module to be shared between scripts. """


def grab_closest_idxs(vector: torch.Tensor, values: Sequence[float]) -> list[int]:
    """ Indices of the entries of `vector` closest to each of `values`. """
    return [int(torch.argmin((vector - value).abs())) for value in values]


# Text width of the SciPost class (`article`, 11pt, a4paper, 30 mm side margins), in inches:
# 210mm - 2*30mm = 150mm. Figures drawn at this width go in with `\includegraphics` at scale
# 1, so nothing is resampled and the font sizes below survive into the PDF.
SCIPOST_TEXTWIDTH = 150 / 25.4

# Body size of the class; its `caption` setup inherits it. Figure text is set a bit smaller
# by default, which is conventional and still comfortably legible at this width.
SCIPOST_FONTSIZE = 11.0

_FONTS = {
    # SciPost sets its text in Bitstream Charter (the class pulls in `mathdesign`, which is
    # loaded with the charter option). matplotlib ships no Charter, and TeX's copy is Type 1,
    # which only the usetex path can use -- so without usetex we fall back through any system
    # Charter to STIX Two, a Times-like face that is far closer to Charter than CM is.
    "charter": (["XCharter", "Charter", "Bitstream Charter", "STIX Two Text"], "stix",
                r"\usepackage[charter]{mathdesign}"),
    "cm":      (["cmr10", "Computer Modern Roman", "DejaVu Serif"], "cm",
                r"\usepackage{lmodern}"),
    "stix":    (["STIX Two Text", "DejaVu Serif"], "stix",
                r"\usepackage{stix2}"),
}


def pub_style(font: str = "charter", base: float = SCIPOST_FONTSIZE - 2,
              usetex: bool = False) -> None:
    """ Publication rcParams, defaulting to the SciPost class's look. `font` picks a family
    from `_FONTS`; `base` is the figure's body size in points. `usetex=True` routes every
    label through a real LaTeX run, which is the only way to match the document exactly --
    it needs the font package installed, and dvipng for on-screen output, though saving to
    PDF needs neither. The mathtext path is the fast default and differs only in the math
    glyphs. """
    import warnings

    import matplotlib as mpl
    from matplotlib import font_manager

    if font not in _FONTS:
        raise ValueError(f"unknown font '{font}', expected one of {sorted(_FONTS)}")
    family, mathfont, preamble = _FONTS[font]

    # matplotlib falls back to DejaVu silently when a family is missing, which is easy not to
    # notice until the figure is next to the text it should match -- so say so.
    if not usetex:
        def installed(name: str) -> bool:
            try:  # findfont raises rather than returning None when the fallback is disabled
                font_manager.findfont(name, fallback_to_default=False)
                return True
            except ValueError:
                return False

        found = next((f for f in family if installed(f)), None)
        if found != family[0]:
            warnings.warn(f"'{family[0]}' not installed, falling back to "
                          f"'{found or 'matplotlib default'}'")

    mpl.rcParams.update({
        "text.usetex":          usetex,
        "text.latex.preamble":  preamble,
        "font.family":          "serif",
        "font.serif":           family,
        "mathtext.fontset":     mathfont,
        # cmr10 has no proper Unicode minus; let mathtext draw it instead of boxing it
        "axes.unicode_minus":   font != "cm",
        "axes.formatter.use_mathtext": True,

        "font.size":            base,
        "axes.titlesize":       base,
        "axes.labelsize":       base,
        "xtick.labelsize":      base - 1,
        "ytick.labelsize":      base - 1,
        "legend.fontsize":      base - 1,

        "axes.linewidth":       0.6,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            True,
        "ytick.right":          True,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "lines.linewidth":      1.2,
        "legend.frameon":       False,

        "figure.dpi":           150,
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
    })
