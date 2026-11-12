"""Tests for partial surveys and per-team statistics."""

import numpy as np
import pytest

from tart_position_cal.calibrate import (
    calibrate,
    calibrate_irls,
    compute_residuals,
    load_measurements,
)
from tart_position_cal.teams import (
    UNASSIGNED,
    compute_leverages,
    load_cell_colours,
    measurement_table,
    parse_team_names,
    team_name,
    team_statistics,
)

N_ANT = 24


def _layout():
    """Five-arm, 24-antenna layout like a TART (mm)."""
    radii = np.array([337, 828, 1198, 1462, 1686])
    pos = []
    for k in range(N_ANT):
        arm, step = divmod(k, 5)
        ang = np.radians(72 * arm + 25 * step)
        pos.append([radii[step] * np.sin(ang), radii[step] * np.cos(ang)])
    return np.array(pos)


def _dist(p):
    return np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))


def _write_ods(path, radius, matrix, colours=None, sheet="Sheet1"):
    """Write a survey sheet in the tart-position-cal layout.

    *radius* and *matrix* hold numbers, strings or None (blank); *colours*
    maps ``(row, col)`` in sheet coordinates to ``#rrggbb``.
    """
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.style import Style, TableCellProperties
    from odf.table import Table, TableCell, TableRow
    from odf.text import P

    colours = colours or {}
    doc = OpenDocumentSpreadsheet()
    style_names = {}
    for colour in sorted(set(colours.values())):
        st = Style(name=f"bg{len(style_names)}", family="table-cell")
        st.addElement(TableCellProperties(backgroundcolor=colour))
        doc.automaticstyles.addElement(st)
        style_names[colour] = st

    def cell(value, r, c):
        kw = {}
        if (r, c) in colours:
            kw["stylename"] = style_names[colours[(r, c)]]
        if value is None:
            return TableCell(**kw)
        if isinstance(value, str):
            tc = TableCell(valuetype="string", **kw)
        else:
            tc = TableCell(valuetype="float", value=float(value), **kw)
        tc.addElement(P(text=str(value)))
        return tc

    tab = Table(name=sheet)
    rows = [[""] + [f"A {i}" for i in range(N_ANT)]]
    rows.append(["Center"] + list(radius))
    for i in range(N_ANT):
        rows.append([f"A {i}"] + list(matrix[i]))
    for r, vals in enumerate(rows):
        tr = TableRow()
        for c, v in enumerate(vals):
            tr.addElement(cell(v if v != "" else None, r, c))
        tab.addElement(tr)
    doc.spreadsheet.addElement(tab)
    doc.save(str(path))


@pytest.fixture
def partial_survey(tmp_path):
    """A partial survey: some radii and pairs blank, one annotated cell,
    two teams by colour, on a sheet not called Sheet1."""
    pos = _layout()
    d = _dist(pos)
    radius = [None] * N_ANT
    for i in (0, 5, 10, 15, 20):
        radius[i] = round(float(np.hypot(*pos[i])), 1)
    matrix = [[None] * N_ANT for _ in range(N_ANT)]
    colours = {}
    for i in range(N_ANT):
        for j in range(i + 1, N_ANT):
            if (i + j) % 3 == 0 or j == i + 1:
                matrix[i][j] = round(float(d[i, j]), 1)
                colours[(2 + i, 1 + j)] = "#ff0000" if i < 12 else "#00ff00"
    matrix[1][2] = f"{d[1, 2]:.0f}**"  # a field annotation
    for i in (0, 5, 10, 15, 20):
        colours[(1, 1 + i)] = "#ff00ff"
    path = tmp_path / "survey.ods"
    _write_ods(path, radius, matrix, colours, sheet="Tab 1")
    return path, pos


def test_load_measurements_partial(partial_survey):
    path, pos = partial_survey
    radius, m_ij, n_ant = load_measurements(path, sheet="Tab 1")
    assert n_ant == N_ANT
    assert np.sum(~np.isnan(radius)) == 5
    assert np.isnan(radius[1])
    # annotated cell read as its leading number, and symmetrised
    assert m_ij[1, 2] == pytest.approx(round(_dist(pos)[1, 2]))
    assert m_ij[2, 1] == m_ij[1, 2]
    assert np.isnan(m_ij[0, 4])


def test_load_measurements_rejects_non_number(tmp_path):
    matrix = [[None] * N_ANT for _ in range(N_ANT)]
    matrix[0][1] = "tape broke"
    path = tmp_path / "bad.ods"
    _write_ods(path, [100.0] * N_ANT, matrix)
    with pytest.raises(ValueError, match="tape broke"):
        load_measurements(path)


def test_calibrate_with_missing_radii(partial_survey):
    path, pos = partial_survey
    radius, m_ij, _ = load_measurements(path, sheet="Tab 1")
    initial = pos + np.random.default_rng(1).normal(0, 20, pos.shape)
    for fn in (calibrate, calibrate_irls):
        res = fn(radius, m_ij, initial, rot_index=0, rot_degrees=0.0)
        r_res, ij_res, _ = compute_residuals(res, radius, m_ij, N_ANT)
        assert np.all(np.isfinite(res.x))
        assert np.isnan(r_res[1])
        assert np.nanmax(np.abs(r_res)) < 2.0
        assert np.median(np.abs(ij_res)) < 1.0


def test_chirality_with_missing_radius(partial_survey):
    """The chirality normalisation must not go NaN on a missing radius."""
    path, pos = partial_survey
    radius, m_ij, _ = load_measurements(path, sheet="Tab 1")
    cross = pos[0, 0] * pos[6, 1] - pos[0, 1] * pos[6, 0]
    assert np.isnan(radius[6])
    res = calibrate(
        radius, m_ij, pos, rot_index=0, rot_degrees=0.0,
        chirality_index=6, chirality_sign=np.sign(cross),
    )
    assert np.all(np.isfinite(res.x))


def test_load_cell_colours(partial_survey):
    path, _ = partial_survey
    radius_colours, pair_colours = load_cell_colours(path, N_ANT, sheet="Tab 1")
    assert radius_colours[0] == "#ff00ff"
    assert radius_colours[1] is None
    assert pair_colours[(0, 1)] == "#ff0000"
    assert pair_colours[(12, 13)] == "#00ff00"
    assert pair_colours[(0, 4)] is None  # blank, uncoloured


def test_load_cell_colours_lower_triangle(tmp_path):
    """A pair entered below the diagonal takes that cell's colour."""
    matrix = [[None] * N_ANT for _ in range(N_ANT)]
    matrix[5][2] = 1000.0  # pair (2, 5), lower triangle
    colours = {(2 + 5, 1 + 2): "#4a86e8", (2 + 2, 1 + 5): "#ff0000"}
    path = tmp_path / "lower.ods"
    _write_ods(path, [None] * N_ANT, matrix, colours)
    _, pair_colours = load_cell_colours(path, N_ANT)
    assert pair_colours[(2, 5)] == "#4a86e8"


def test_load_cell_colours_missing_sheet(partial_survey):
    path, _ = partial_survey
    with pytest.raises(ValueError, match="Tab 1"):
        load_cell_colours(path, N_ANT, sheet="Sheet1")


def test_team_names():
    assert team_name("#ff0000") == "red"
    assert team_name("#4a86e8") == "blue"
    assert team_name("#123456") == "#123456"
    assert team_name(None) == UNASSIGNED
    assert team_name("#ffffff") == UNASSIGNED
    names = parse_team_names(["#FF0000=Team Rocket", "00ff00 = Greens"])
    assert names == {"#ff0000": "Team Rocket", "#00ff00": "Greens"}
    assert team_name("#ff0000", names) == "Team Rocket"
    with pytest.raises(ValueError):
        parse_team_names(["red"])


def _team_survey(noise_a, noise_b, blunder_b=0.0, seed=0):
    """Full survey of the layout; team A owns i < 12, team B the rest."""
    rng = np.random.default_rng(seed)
    pos = _layout()
    d = _dist(pos)
    m_ij = np.full((N_ANT, N_ANT), np.nan)
    pair_colours = {}
    for i in range(N_ANT):
        for j in range(i + 1, N_ANT):
            team_a = i < 12
            m_ij[i, j] = d[i, j] + rng.normal(0, noise_a if team_a else noise_b)
            pair_colours[(i, j)] = "#ff0000" if team_a else "#00ff00"
    m_ij[20, 22] += blunder_b
    m_ij = np.where(np.isnan(m_ij), m_ij.T, m_ij)
    radius = np.hypot(pos[:, 0], pos[:, 1])
    return radius, m_ij, pos, ["#ff00ff"] * N_ANT, pair_colours


def test_team_ranking_by_noise():
    radius, m_ij, pos, rc, pc = _team_survey(noise_a=0.5, noise_b=3.0)
    res = calibrate(radius, m_ij, pos, rot_index=0, rot_degrees=0.0)
    rows = measurement_table(res, radius, m_ij, N_ANT, rc, pc)
    stats, sigma, _ = team_statistics(rows)
    assert stats[("red", "pair")]["rms"] < stats[("green", "pair")]["rms"]
    # corrected errors recover the injected noise levels roughly
    assert stats[("red", "pair")]["rms"] == pytest.approx(0.5, rel=0.5)
    assert stats[("green", "pair")]["rms"] == pytest.approx(3.0, rel=0.5)
    assert ("magenta", "radius") in stats


def test_blunder_flagged_as_outlier():
    radius, m_ij, pos, rc, pc = _team_survey(noise_a=0.5, noise_b=0.5, blunder_b=100.0)
    res = calibrate_irls(radius, m_ij, pos, rot_index=0, rot_degrees=0.0)
    rows = measurement_table(res, radius, m_ij, N_ANT, rc, pc)
    stats, _, limit = team_statistics(rows)
    assert stats[("green", "pair")]["outliers"] >= 1
    # ~150 measurements with 0.5 mm noise: at most a chance 3-sigma excursion
    assert stats[("red", "pair")]["outliers"] <= 1
    worst = max((r for r in rows if r["checked"]), key=lambda r: abs(r["error"]))
    assert (worst["i"], worst["j"]) == (20, 22)
    assert abs(worst["error"]) > limit


def test_sole_constraint_is_unchecked():
    """A measurement that alone fixes an antenna has leverage ~1 and is
    left out of the statistics instead of being scored as perfect."""
    radius, m_ij, pos, rc, pc = _team_survey(noise_a=0.5, noise_b=0.5)
    # Antenna 23: keep only its radius and one pair, so it has two
    # constraints for two coordinates -- no redundancy at all.
    m_ij[23, :] = np.nan
    m_ij[:, 23] = np.nan
    m_ij[22, 23] = m_ij[23, 22] = _dist(pos)[22, 23] + 40.0  # wrong, but undetectable
    res = calibrate(radius, m_ij, pos, rot_index=0, rot_degrees=0.0)
    _, ij_res, pairs = compute_residuals(res, radius, m_ij, N_ANT)
    _, h_pairs = compute_leverages(res, N_ANT, len(ij_res))
    k = [tuple(p) for p in pairs.tolist()].index((22, 23))
    assert h_pairs[k] > 0.9
    rows = measurement_table(res, radius, m_ij, N_ANT, rc, pc)
    row = next(r for r in rows if (r["i"], r["j"]) == (22, 23))
    assert not row["checked"]
    assert np.isnan(row["error"])


def test_parse_exclusions():
    from tart_position_cal.calibrate import parse_exclusions

    radii, pairs = parse_exclusions(["r18,r19", "12-3 A4-A5", "R20"], N_ANT)
    assert radii == {18, 19, 20}
    assert pairs == {(3, 12), (4, 5)}
    for bad in (["r24"], ["3-3"], ["x7"], ["1-99"]):
        with pytest.raises(ValueError):
            parse_exclusions(bad, N_ANT)


def test_excluded_measurements_scored_but_not_fitted():
    """A biased radius set that drags an antenna is excluded from the fit;
    the fit recovers, and the excluded radii are still charged to the team."""
    from tart_position_cal.calibrate import apply_exclusions

    radius, m_ij, pos, rc, pc = _team_survey(noise_a=0.3, noise_b=0.3)
    biased = radius.copy()
    biased[18:24] -= 24.0  # one team's radii all short, as at UP Maputo
    fit_r, fit_m = apply_exclusions(biased, m_ij, set(range(18, 24)), set())
    res = calibrate_irls(fit_r, fit_m, pos, rot_index=0, rot_degrees=0.0)
    rows = measurement_table(
        res, fit_r, fit_m, N_ANT, rc, pc, radius_all=biased, m_ij_all=m_ij
    )
    excl = [r for r in rows if r["excluded"]]
    assert sorted(r["i"] for r in excl) == list(range(18, 24))
    assert all(r["checked"] and r["error"] == pytest.approx(24.0, abs=2.0) for r in excl)
    stats, _, _ = team_statistics(rows)
    assert stats[("magenta", "radius")]["outliers"] == 6
    assert len(rows) == N_ANT + np.sum(~np.isnan(np.triu(m_ij, 1)) & (np.triu(m_ij, 1) != 0))

