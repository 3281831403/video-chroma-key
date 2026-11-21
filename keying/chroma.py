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

import math

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


_TOL = 1e-6
_BLACK = (0.0, 0.0, 0.0)


def _is_finite(value: object) -> bool:
    """Return ``True`` only for a real, finite scalar."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _components(value: object) -> tuple[float, float, float]:
    """Return the three float components of a finite RGB triple.

    Raises :class:`KeyError` for a non-tuple, wrong arity, non-finite
    component, or a component outside ``[0.0, 1.0]``.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise KeyError(f"expected a 3-component RGB pixel, got {value!r}")
    for component in value:
        if not _is_finite(component):
            raise KeyError(f"pixel component must be finite, got {component!r}")
        if component < 0.0 or component > 1.0:
            raise KeyError(f"pixel component must be in [0.0, 1.0], got {component!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _validated_key(key: object) -> tuple[float, float, float]:
    """Validate a key colour, rejecting an exactly black key."""
    validated = _components(key)
    if all(abs(component) <= _TOL for component in validated):
        raise KeyError("key colour must not be black")
    return validated


def _validated_params(similarity: object, smoothness: object) -> tuple[float, float]:
    """Validate the two alpha-ramp parameters and their ordering."""
    for name, value in (("similarity", similarity), ("smoothness", smoothness)):
        if not _is_finite(value):
            raise KeyError(f"{name} must be a finite number, got {value!r}")
        if value < 0.0 or value > 1.0:
            raise KeyError(f"{name} must be in [0.0, 1.0], got {value!r}")
    if not smoothness > similarity:
        raise KeyError(
            "smoothness must be greater than similarity, "
            f"got smoothness={smoothness!r} similarity={similarity!r}"
        )
    return float(similarity), float(smoothness)


def _validated_alpha(value: object) -> float:
    """Validate a scalar coverage value in ``[0.0, 1.0]``."""
    if not _is_finite(value):
        raise KeyError(f"alpha must be a finite number, got {value!r}")
    if value < 0.0 or value > 1.0:
        raise KeyError(f"alpha must be in [0.0, 1.0], got {value!r}")
    return float(value)



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
    r, g, b = _components(pixel)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (b - y) / 1.772
    cr = 0.5 + (r - y) / 1.402
    return (y, cb, cr)


def chroma_distance(pixel: Pixel, key: Pixel) -> float:
    """Return the Euclidean distance between two pixels in the Cb/Cr plane.

    Both pixels are converted with :func:`ycbcr`; luma is discarded.  With
    ``d_cb = cb_pixel - cb_key`` and ``d_cr = cr_pixel - cr_key`` the result
    is ``math.hypot(d_cb, d_cr)``.  This is a distance in chroma only, not an
    RGB or luma distance.  The value is checked at an absolute tolerance of
    ``1e-6``.

    A malformed pixel or key raises :class:`KeyError`.
    """
    _, pixel_cb, pixel_cr = ycbcr(_components(pixel))
    _, key_cb, key_cr = ycbcr(_validated_key(key))
    return math.hypot(pixel_cb - key_cb, pixel_cr - key_cr)


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
    similarity, smoothness = _validated_params(similarity, smoothness)
    key = _validated_key(key)
    distance = chroma_distance(pixel, key)

    if distance <= similarity:
        return 0.0
    if distance >= similarity + smoothness:
        return 1.0
    alpha = (distance - similarity) / smoothness
    return min(1.0, max(0.0, alpha))


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
    r, g, b = _components(pixel)
    key = _validated_key(key)
    if not _is_finite(amount):
        raise KeyError(f"amount must be a finite number, got {amount!r}")
    if amount < 0.0 or amount > 1.0:
        raise KeyError(f"amount must be in [0.0, 1.0], got {amount!r}")

    channels = [r, g, b]
    dominant = key.index(max(key))
    channels[dominant] *= 1.0 - amount
    return (channels[0], channels[1], channels[2])


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
    if isinstance(radius, bool) or not isinstance(radius, int):
        raise KeyError(f"radius must be an integer, got {radius!r}")
    if radius < 0:
        raise KeyError(f"radius must be non-negative, got {radius!r}")
    if not isinstance(alpha, (list, tuple)):
        raise KeyError(f"alpha matte must be a list of rows, got {alpha!r}")

    rows = len(alpha)
    if rows == 0:
        return []
    if not isinstance(alpha[0], (list, tuple)):
        raise KeyError(f"alpha row must be a list, got {alpha[0]!r}")
    cols = len(alpha[0])
    for row in alpha:
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise KeyError("alpha matte must be rectangular")

    samples: list[list[float]] = []
    for row in alpha:
        validated_row = []
        for value in row:
            if not _is_finite(value):
                raise KeyError(f"alpha sample must be finite, got {value!r}")
            if value < 0.0 or value > 1.0:
                raise KeyError(f"alpha sample must be in [0.0, 1.0], got {value!r}")
            validated_row.append(float(value))
        samples.append(validated_row)

    if radius == 0 or cols == 0:
        return [row[:] for row in samples]

    feathered: list[list[float]] = []
    for y in range(rows):
        out_row = []
        for x in range(cols):
            value = samples[y][x]
            if value == 0.0 or value == 1.0:
                out_row.append(value)
                continue
            y_lo, y_hi = max(0, y - radius), min(rows, y + radius + 1)
            x_lo, x_hi = max(0, x - radius), min(cols, x + radius + 1)
            total = 0.0
            count = 0
            for ny in range(y_lo, y_hi):
                source_row = samples[ny]
                for nx in range(x_lo, x_hi):
                    total += source_row[nx]
                    count += 1
            out_row.append(total / count)
        feathered.append(out_row)
    return feathered


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
    fg_r, fg_g, fg_b = _components(fg)
    bg_r, bg_g, bg_b = _components(bg)
    alpha = _validated_alpha(alpha)
    inverse = 1.0 - alpha
    return (
        fg_r * alpha + bg_r * inverse,
        fg_g * alpha + bg_g * inverse,
        fg_b * alpha + bg_b * inverse,
    )


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
    problems: list[str] = []

    key_ok = False
    if not isinstance(key, (tuple, list)) or len(key) != 3:
        problems.append(f"key must be a 3-component RGB triple, got {key!r}")
    else:
        key_ok = True
        for index, component in enumerate(key):
            if not _is_finite(component):
                problems.append(f"key component {index} must be finite, got {component!r}")
                key_ok = False
            elif component < 0.0 or component > 1.0:
                problems.append(
                    f"key component {index} must be in [0.0, 1.0], got {component!r}"
                )
                key_ok = False
        if key_ok and all(abs(component) <= _TOL for component in key):
            problems.append("key must not be black: (0.0, 0.0, 0.0) would key out every black pixel")

    similarity_ok = _is_finite(similarity) and 0.0 <= similarity <= 1.0
    if not similarity_ok:
        problems.append(f"similarity must be finite and in [0.0, 1.0], got {similarity!r}")

    smoothness_ok = _is_finite(smoothness) and 0.0 <= smoothness <= 1.0
    if not smoothness_ok:
        problems.append(f"smoothness must be finite and in [0.0, 1.0], got {smoothness!r}")

    if similarity_ok and smoothness_ok and not smoothness > similarity:
        problems.append(
            "smoothness must be greater than similarity, "
            f"got smoothness={smoothness!r} similarity={similarity!r}"
        )

    return problems


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
    if not isinstance(config, dict):
        raise KeyError(f"config must be a mapping, got {config!r}")
    for field in ("key", "similarity", "smoothness"):
        if field not in config:
            raise KeyError(f"config is missing {field!r}")

    key = config["key"]
    similarity = config["similarity"]
    smoothness = config["smoothness"]
    problems = validate_params(key, similarity, smoothness)
    if problems:
        raise KeyError("; ".join(problems))

    # Validate every frame fully before building any result, so an invalid
    # frame late in the batch cannot leave a partial result behind.
    checked: list[list[list[tuple[float, float, float]]]] = []
    if not isinstance(frames, (list, tuple)):
        raise KeyError(f"frames must be a list, got {frames!r}")
    for index, frame in enumerate(frames):
        if not isinstance(frame, (list, tuple)):
            raise KeyError(f"frame {index} must be a list of rows")
        if len(frame) == 0:
            checked.append([])
            continue
        if not isinstance(frame[0], (list, tuple)):
            raise KeyError(f"frame {index} row must be a list")
        width = len(frame[0])
        validated_frame = []
        for y, row in enumerate(frame):
            if not isinstance(row, (list, tuple)) or len(row) != width:
                raise KeyError(f"frame {index} must be rectangular")
            validated_row = []
            for x, pixel in enumerate(row):
                try:
                    validated_row.append(_components(pixel))
                except KeyError as exc:
                    raise KeyError(f"frame {index} pixel ({y}, {x}): {exc}") from exc
            validated_frame.append(validated_row)
        checked.append(validated_frame)

    alpha_planes = []
    for frame in checked:
        plane = []
        for row in frame:
            # A fresh list per row and per frame: no shared or aliased buffers.
            plane.append([alpha_for(pixel, key, similarity, smoothness) for pixel in row])
        alpha_planes.append(plane)

    return {"alpha": alpha_planes, "count": len(alpha_planes)}
