"""Green-screen chroma keying contracts.

Pixels are ``(r, g, b)`` tuples whose components are normalized to
``[0.0, 1.0]``.  Images/frames and alpha mattes are rectangular, row-major
two-dimensional ``list[list[...]]`` values addressed as ``plane[y][x]``.
Chroma is stored in BT.601 full-range YCbCr with neutral chroma at ``0.5``,
so a component below ``0.5`` is negative chroma rather than a missing offset.

``similarity`` and ``smoothness`` are also normalized to ``[0.0, 1.0]`` and
must satisfy ``smoothness > similarity``; the ramp has no zero-width
division.  The key colour is an RGB triple in ``[0.0, 1.0]``, but an exactly
black key ``(0.0, 0.0, 0.0)`` is rejected: it would key out every black
pixel and erase an otherwise black frame.

Numeric results are compared with an absolute tolerance of ``1e-6``.
Invalid pixels, keys, parameters, mattes, or frames raise the custom
:class:`KeyError` declared here; it deliberately shadows the built-in name
inside this module.
"""

from __future__ import annotations

__all__ = [
    "KeyError",
    "alpha_for",
    "chroma_distance",
    "composite",
    "despill_edge",
    "key_batch",
    "spill_suppress",
    "validate_params",
    "ycbcr",
]

Pixel = tuple[float, float, float]
Frame = list[list[Pixel]]


class KeyError(ValueError):
    """Raised for an invalid pixel, key, parameter, matte, or frame."""


def ycbcr(pixel: Pixel) -> tuple[float, float, float]:
    """Convert one normalized RGB pixel to BT.601 full-range YCbCr.

    ``pixel`` is an ``(r, g, b)`` tuple with each component in ``[0.0, 1.0]``.
    The conversion is::

        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.5 + (b - y) / 1.772
        cr = 0.5 + (r - y) / 1.402

    ``cb`` and ``cr`` are therefore centred on ``0.5``; neutral grey maps to
    ``(y, 0.5, 0.5)`` rather than to zero chroma.  The return value is a
    three-tuple of floats and is not clamped.  Values are checked at an
    absolute tolerance of ``1e-6``.

    A malformed pixel (wrong arity, non-finite component, or a component
    outside ``[0.0, 1.0]``) raises :class:`KeyError`.
    """
    raise NotImplementedError


def chroma_distance(pixel: Pixel, key: Pixel) -> float:
    """Return the Euclidean distance between two pixels in the Cb/Cr plane.

    Both pixels are converted with :func:`ycbcr`; luma is discarded.  With
    ``d_cb = cb_pixel - cb_key`` and ``d_cr = cr_pixel - cr_key`` the result
    is ``math.hypot(d_cb, d_cr)``.  This is a distance in chroma only, not an
    RGB or luma distance.  The value is checked at an absolute tolerance of
    ``1e-6``.

    A malformed pixel or key raises :class:`KeyError`.
    """
    raise NotImplementedError


def alpha_for(
    pixel: Pixel,
    key: Pixel,
    similarity: float,
    smoothness: float,
) -> float:
    """Compute the two-stage alpha for one pixel against a key colour.

    Let ``d = chroma_distance(pixel, key)``.  The result is the linear ramp::

        alpha = 0.0                              if d <= similarity
        alpha = (d - similarity) / smoothness    if similarity < d
                                                  < similarity + smoothness
        alpha = 1.0                              if d >= similarity + smoothness

    ``similarity`` is the lower keying threshold and ``smoothness`` is the
    width of the soft edge, so a single hard threshold is not sufficient.
    Both parameters are in ``[0.0, 1.0]`` and must satisfy
    ``smoothness > similarity``.  The result is in ``[0.0, 1.0]`` and is
    checked at an absolute tolerance of ``1e-6``.

    Malformed pixels, an invalid key (including an exactly black key), or
    invalid parameters raise :class:`KeyError` instead of dividing by zero.
    """
    raise NotImplementedError


def spill_suppress(pixel: Pixel, key: Pixel, amount: float) -> Pixel:
    """Reduce green-screen spill on only the key colour's dominant channel.

    The dominant channel is the first index attaining the maximum component
    of ``key``.  With ``amount`` in ``[0.0, 1.0]``, that one channel is scaled
    by ``(1.0 - amount)``; the other two channels are returned unchanged.  An
    ``amount`` of ``0.0`` is a no-op and an ``amount`` of ``1.0`` removes the
    dominant channel completely.  This avoids tinting the whole pixel, which
    would discolour skin and neutral detail.  Results are checked at an
    absolute tolerance of ``1e-6``.

    A malformed pixel, an invalid key (including black), or an out-of-range
    ``amount`` raises :class:`KeyError`.
    """
    raise NotImplementedError


def despill_edge(alpha: list[list[float]], radius: int) -> list[list[float]]:
    """Feather only the soft edge of an alpha matte, preserving its interior.

    ``alpha`` is a rectangular, row-major two-dimensional matte of values in
    ``[0.0, 1.0]``.  ``radius`` is a non-negative integer.  A pixel whose
    alpha is strictly between ``0.0`` and ``1.0`` is a soft-edge sample and is
    replaced by the arithmetic mean of all matte samples within Chebyshev
    distance ``radius`` (the square neighbourhood clipped at the image
    border).  Samples that are exactly ``0.0`` or exactly ``1.0`` are interior
    or background and are returned unchanged, so feathering never blurs the
    fully opaque interior.  ``radius == 0`` returns a deep copy unchanged.
    An empty matte maps to ``[]``.  Results are checked at an absolute
    tolerance of ``1e-6``.

    Ragged rows, non-finite or out-of-range samples, or a negative
    ``radius`` raise :class:`KeyError`.
    """
    raise NotImplementedError


def composite(fg: Pixel, bg: Pixel, alpha: float) -> tuple[float, float, float]:
    """Composite ``fg`` over ``bg`` with a scalar coverage value.

    With ``alpha`` in ``[0.0, 1.0]``, each output channel is the premultiplied
    source-over result::

        out[c] = fg[c] * alpha + bg[c] * (1.0 - alpha)

    For example, half-covered red over blue gives ``(0.5, 0.0, 0.5)``.  The
    foreground must be multiplied by alpha; an unpremultiplied sum leaves a
    dark or otherwise incorrect fringe.  Components are floats in
    ``[0.0, 1.0]`` and results are checked at an absolute tolerance of
    ``1e-6``.

    A malformed pixel or out-of-range alpha raises :class:`KeyError`.
    """
    raise NotImplementedError


def validate_params(
    key: Pixel,
    similarity: float,
    smoothness: float,
) -> list[str]:
    """Return all configuration problems, or ``[]`` when the setup is valid.

    A valid setup has all of the following:

    * ``key`` is a three-component finite RGB triple, each in ``[0.0, 1.0]``.
    * ``key`` is not exactly ``(0.0, 0.0, 0.0)``.  A black key would match
      every black pixel and key out a fully black frame, so it is rejected
      as an invalid chroma key rather than silently accepted.
    * ``similarity`` is finite and in ``[0.0, 1.0]``.
    * ``smoothness`` is finite and in ``[0.0, 1.0]``.
    * ``smoothness > similarity``.  Equality or inversion would make the
      alpha ramp zero-width or inverted and can cause a division by zero.

    The return value is a list of human-readable issue strings; each issue
    names its field (``key``, ``similarity``, or ``smoothness``).  Malformed
    ordinary values are reported in that list instead of being raised.  The
    full list is returned when several fields are wrong.  Values are checked
    at an absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def key_batch(frames: list[Frame], config: dict) -> dict:
    """Key a batch of RGB frames atomically and preserve their input order.

    ``frames`` is a list of rectangular two-dimensional frames, each of whose
    elements is an ``(r, g, b)`` pixel in ``[0.0, 1.0]``.  ``config`` is a
    mapping with ``"key"``, ``"similarity"``, and ``"smoothness"`` entries
    that must satisfy :func:`validate_params`.  Each frame is keyed with
    :func:`alpha_for`.

    The return value is::

        {"alpha": [alpha_plane, ...], "count": len(frames)}

    where every alpha plane has exactly the shape of its input frame and the
    planes follow the input frame order.  Every plane is a fresh buffer; no
    two frames share an alpha list, and no output aliases an input.  An empty
    batch returns ``{"alpha": [], "count": 0}``.  Results are checked at an
    absolute tolerance of ``1e-6``.

    The operation is all-or-nothing.  The configuration is validated and
    every frame is checked for rectangular shape, pixel arity, finiteness,
    and component range before any result is built.  An invalid config or
    frame raises :class:`KeyError`; no partial result is returned and the
    input frames are never mutated.
    """
    raise NotImplementedError
