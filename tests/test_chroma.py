"""Red tests for the green-screen chroma keying kernels.

The tests are intentionally read-only: fix ``keying/chroma.py`` instead of
editing this file.  Every test currently fails with ``NotImplementedError``
from one of the eight stubs.
"""

import pytest

from keying.chroma import (
    KeyError,
    alpha_for,
    chroma_distance,
    composite,
    despill_edge,
    key_batch,
    spill_suppress,
    validate_params,
    ycbcr,
)

TOL = 1e-6
GREEN = (0.0, 1.0, 0.0)


def _assert_plane_close(actual, expected, tol=TOL):
    assert len(actual) == len(expected)
    for row, expected_row in zip(actual, expected):
        assert len(row) == len(expected_row)
        assert row == pytest.approx(expected_row, abs=tol)


def test_chroma_distance_uses_chroma_plane_not_rgb():
    pixel = (0.0, 0.3, 0.0)

    distance = chroma_distance(pixel, GREEN)

    assert distance == pytest.approx(0.373720820039, abs=TOL)
    assert abs(distance - 0.7) > 0.1


def test_ycbcr_keeps_chroma_centered_at_half():
    black = ycbcr((0.0, 0.0, 0.0))
    white = ycbcr((1.0, 1.0, 1.0))

    assert black == pytest.approx((0.0, 0.5, 0.5), abs=TOL)
    assert white == pytest.approx((1.0, 0.5, 0.5), abs=TOL)


def test_alpha_has_soft_edge_instead_of_single_threshold():
    pixel = (0.0, 0.3, 0.0)

    alpha = alpha_for(pixel, GREEN, 0.2, 0.4)

    assert 0.0 < alpha < 1.0
    assert alpha == pytest.approx(0.434302050098, abs=TOL)


def test_smoothness_controls_transition_width():
    pixel = (0.0, 0.3, 0.0)

    narrow = alpha_for(pixel, GREEN, 0.2, 0.3)
    wide = alpha_for(pixel, GREEN, 0.2, 0.6)

    assert narrow == pytest.approx(0.579069400131, abs=TOL)
    assert wide == pytest.approx(0.289534700065, abs=TOL)
    assert narrow > wide


def test_spill_suppress_only_lowers_key_channel():
    result = spill_suppress((0.2, 0.8, 0.3), GREEN, 0.5)

    assert result[0] == pytest.approx(0.2, abs=TOL)
    assert result[1] == pytest.approx(0.4, abs=TOL)
    assert result[2] == pytest.approx(0.3, abs=TOL)


def test_despill_edge_preserves_matte_interior():
    matte = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.5, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]

    feathered = despill_edge(matte, 1)

    assert feathered[1][2] == pytest.approx(0.611111111111, abs=TOL)
    for row in range(len(matte)):
        for col in range(len(matte[row])):
            if matte[row][col] in (0.0, 1.0):
                assert feathered[row][col] == pytest.approx(
                    matte[row][col], abs=TOL
                )


def test_composite_premultiplies_foreground():
    result = composite((0.8, 0.4, 0.2), (0.1, 0.2, 0.5), 0.25)

    assert result == pytest.approx((0.275, 0.25, 0.425), abs=TOL)


def test_validate_params_rejects_smoothness_not_above_similarity():
    problems = validate_params(GREEN, 0.6, 0.4)

    assert problems
    assert any("smoothness" in problem.lower() for problem in problems)
    with pytest.raises(KeyError):
        alpha_for((0.0, 0.3, 0.0), GREEN, 0.6, 0.4)


def test_batch_keeps_frame_alpha_buffers_independent():
    config = {"key": GREEN, "similarity": 0.1, "smoothness": 0.2}
    frame_a = [[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]]
    frame_b = [[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]]

    result = key_batch([frame_a, frame_b], config)
    alpha = result["alpha"]

    assert result["count"] == 2
    _assert_plane_close(alpha[0], [[0.0, 1.0]])
    _assert_plane_close(alpha[1], [[1.0, 0.0]])
    assert alpha[0] is not alpha[1]

    bad_frame = [[(0.0, 1.0, 0.0), (1.5, 0.0, 0.0)]]
    with pytest.raises(KeyError):
        key_batch([frame_a, bad_frame], config)


def test_black_key_is_rejected_for_black_frames():
    problems = validate_params((0.0, 0.0, 0.0), 0.1, 0.2)

    assert problems
    assert any("black" in problem.lower() for problem in problems)
    with pytest.raises(KeyError):
        key_batch(
            [[[(0.0, 0.0, 0.0)]]],
            {"key": (0.0, 0.0, 0.0), "similarity": 0.1, "smoothness": 0.2},
        )


def test_alpha_of_key_color_is_zero():
    alpha = alpha_for(GREEN, GREEN, 0.2, 0.3)

    assert alpha == pytest.approx(0.0, abs=TOL)


def test_distant_color_has_full_alpha():
    alpha = alpha_for((1.0, 0.0, 0.0), GREEN, 0.2, 0.3)

    assert alpha == pytest.approx(1.0, abs=TOL)


def test_empty_matte_returns_empty():
    assert despill_edge([], 1) == []
    assert key_batch(
        [], {"key": GREEN, "similarity": 0.1, "smoothness": 0.2}
    ) == {"alpha": [], "count": 0}


def test_similarity_half_places_lower_alpha_threshold():
    key = (0.5, 0.5, 0.5)
    pixel = (0.0, 0.0, 1.0)

    distance = chroma_distance(pixel, key)
    alpha = alpha_for(pixel, key, 0.5, 0.6)

    assert distance == pytest.approx(0.506568562148, abs=TOL)
    assert alpha == pytest.approx(0.010947603580, abs=TOL)


def test_batch_success_preserves_frame_order():
    frames = [
        [[(0.0, 1.0, 0.0)]],
        [[(0.0, 0.3, 0.0)]],
        [[(1.0, 0.0, 0.0)]],
    ]

    result = key_batch(
        frames,
        {"key": GREEN, "similarity": 0.2, "smoothness": 0.4},
    )
    alpha = result["alpha"]

    assert result["count"] == 3
    assert [plane[0][0] for plane in alpha] == pytest.approx(
        [0.0, 0.434302050098, 1.0], abs=TOL
    )
