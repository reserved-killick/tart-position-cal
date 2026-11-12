"""Per-team residual statistics for colour-coded survey spreadsheets.

When several teams share the survey, each team's measurements are marked
by the background colour of their cells.  This module reads those colours
from the ODS file, ties every measurement to its team, and reports how
well each team's measurements agree with the calibrated solution.

Raw residuals are a poor yardstick on their own: a measurement that is the
only thing fixing an antenna's position is fitted exactly and so has a
residual of zero, however wrong it is.  Each residual is therefore divided
by ``sqrt(1 - h)``, where ``h`` is its leverage (the diagonal of the hat
matrix of the linearised fit).  That turns it into an estimate of the
measurement's actual error, and measurements with almost no redundancy
(``1 - h`` below :data:`MIN_REDUNDANCY`) are counted as *unchecked* rather
than being credited as perfect.
"""

import csv

import numpy as np

from tart_position_cal.calibrate import compute_residuals

#: Measurements whose redundancy ``1 - h`` falls below this are unchecked.
MIN_REDUNDANCY = 0.1

#: A checked measurement is an outlier when its corrected error exceeds
#: this many robust standard deviations of all checked measurements.
OUTLIER_SIGMA = 3.0

#: Friendly names for the usual spreadsheet palette colours.
COLOUR_NAMES = {
    "#ff0000": "red",
    "#00ff00": "green",
    "#0000ff": "blue",
    "#4a86e8": "blue",
    "#ff00ff": "magenta",
    "#00ffff": "cyan",
    "#ffff00": "yellow",
    "#ff9900": "orange",
    "#9900ff": "purple",
    "#000000": "black",
}

#: Background colours that mean "no team".
NO_COLOUR = (None, "", "transparent", "#ffffff")

UNASSIGNED = "(no colour)"

_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_MAX_ROWS = 256
_MAX_COLS = 64


# ---------------------------------------------------------------------------
# Reading cell colours from the spreadsheet
# ---------------------------------------------------------------------------


def _ods_grid(path, sheet):
    """Return the sheet as a list of rows of ``(has_value, colour)`` tuples."""
    from odf import table, teletype
    from odf.namespaces import FONS, STYLENS, TABLENS
    from odf.opendocument import load
    from odf.style import Style, TableCellProperties

    doc = load(str(path))

    # style name -> (background colour or None, parent style name)
    styles = {}
    for st in list(doc.automaticstyles.getElementsByType(Style)) + list(
        doc.styles.getElementsByType(Style)
    ):
        colour = None
        for props in st.getElementsByType(TableCellProperties):
            colour = props.attributes.get((FONS, "background-color")) or colour
        styles[st.getAttribute("name")] = (
            colour,
            st.attributes.get((STYLENS, "parent-style-name")),
        )

    def colour_of(name, depth=0):
        if name is None or name not in styles or depth > 10:
            return None
        colour, parent = styles[name]
        return colour if colour is not None else colour_of(parent, depth + 1)

    sheets = {t.getAttribute("name"): t for t in doc.spreadsheet.getElementsByType(table.Table)}
    if sheet not in sheets:
        raise ValueError(
            f"No sheet named {sheet!r} in {path!r} (sheets: {', '.join(sheets)})"
        )
    tab = sheets[sheet]

    # Column default cell styles, used when a cell names no style of its own.
    col_styles = []
    for col in tab.getElementsByType(table.TableColumn):
        rep = int(col.attributes.get((TABLENS, "number-columns-repeated"), 1))
        name = col.attributes.get((TABLENS, "default-cell-style-name"))
        col_styles.extend([name] * min(rep, _MAX_COLS))

    grid = []
    for row in tab.getElementsByType(table.TableRow):
        rrep = int(row.attributes.get((TABLENS, "number-rows-repeated"), 1))
        cells = []
        for cell in row.childNodes:
            if cell.qname[1] not in ("table-cell", "covered-table-cell"):
                continue
            crep = int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))
            has_value = (
                cell.attributes.get((_OFFICE_NS, "value")) is not None
                or teletype.extractText(cell).strip() != ""
            )
            for _ in range(min(crep, _MAX_COLS)):
                c = len(cells)
                sname = cell.attributes.get((TABLENS, "style-name"))
                if sname is None and c < len(col_styles):
                    sname = col_styles[c]
                cells.append((has_value, colour_of(sname)))
            if len(cells) >= _MAX_COLS:
                break
        grid.extend([cells] * min(rrep, _MAX_ROWS))
        if len(grid) >= _MAX_ROWS:
            break
    return grid, tab


def load_cell_colours(path, n_ant, sheet="Sheet1"):
    """Read the background colour of every measurement cell.

    Returns ``(radius_colours, pair_colours)``: a list of *n_ant* colours
    for the radius ("Center") row, and a dict mapping each antenna pair
    ``(i, j)`` with ``i < j`` to the colour of the cell holding its
    measurement.  The upper-triangle cell is used when it holds a value
    (as :func:`~tart_position_cal.calibrate.load_measurements` does),
    otherwise the lower-triangle one.  Colours are lower-case ``#rrggbb``
    strings, or ``None`` for an uncoloured cell.

    The layout matches ``load_measurements``: a header row of antenna
    labels ``"A 0"``, ``"A 1"``, ..., then the radius row, then one row per
    antenna.
    """
    from odf import table, teletype
    from odf.namespaces import TABLENS

    grid, tab = _ods_grid(path, sheet)

    # Locate the header row and the column of each antenna label.
    header_row, label_col = None, {}
    for r, row in enumerate(tab.getElementsByType(table.TableRow)):
        c = 0
        found = {}
        for cell in row.childNodes:
            if cell.qname[1] not in ("table-cell", "covered-table-cell"):
                continue
            crep = int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))
            txt = teletype.extractText(cell).strip()
            if txt.startswith("A ") and txt[2:].strip().isdigit():
                found[int(txt[2:])] = c
            c += min(crep, _MAX_COLS)
        if 0 in found:
            header_row, label_col = r, found
            break
        if r > 20:
            break
    if header_row is None:
        raise ValueError(f"Could not find the 'A 0' header row in {path!r}")
    missing = [i for i in range(n_ant) if i not in label_col]
    if missing:
        raise ValueError(f"Header row in {path!r} lacks columns for antennas {missing}")

    def cell(r, c):
        if r < len(grid) and c < len(grid[r]):
            has_value, colour = grid[r][c]
            return has_value, (colour.lower() if colour else None)
        return False, None

    radius_colours = [cell(header_row + 1, label_col[i])[1] for i in range(n_ant)]
    pair_colours = {}
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            up_val, up_col = cell(header_row + 2 + i, label_col[j])
            lo_val, lo_col = cell(header_row + 2 + j, label_col[i])
            pair_colours[(i, j)] = up_col if (up_val or not lo_val) else lo_col
    return radius_colours, pair_colours


def team_name(colour, names=None):
    """Name the team that owns *colour*, using *names* (colour -> name) first."""
    if colour in NO_COLOUR:
        return UNASSIGNED
    if names and colour in names:
        return names[colour]
    return COLOUR_NAMES.get(colour, colour)


def parse_team_names(specs):
    """Parse ``["#ff0000=Team Red", ...]`` into ``{"#ff0000": "Team Red"}``."""
    out = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--team expects COLOUR=NAME, got {spec!r}")
        colour, name = spec.split("=", 1)
        colour = colour.strip().lower()
        if not colour.startswith("#"):
            colour = "#" + colour
        out[colour] = name.strip()
    return out


# ---------------------------------------------------------------------------
# Leverage-corrected residuals
# ---------------------------------------------------------------------------


def compute_leverages(result, n_ant, n_pairs):
    """Leverages ``h`` of the radius and inter-antenna measurements.

    Uses the Jacobian of the (weighted) residual vector at the solution,
    ``result.jac``, as returned by :func:`scipy.optimize.least_squares`
    and kept by ``calibrate`` / ``calibrate_irls``.  The pseudo-inverse
    copes with any directions the survey leaves unconstrained.

    Returns ``(h_radius, h_pairs)``, arrays of length *n_ant* and *n_pairs*.
    """
    jac = np.asarray(result.jac.todense() if hasattr(result.jac, "todense") else result.jac)
    cov = np.linalg.pinv(jac.T @ jac)
    h = np.einsum("ij,jk,ik->i", jac, cov, jac)
    h = np.clip(h, 0.0, 1.0)
    return h[:n_ant], h[n_ant : n_ant + n_pairs]


def measurement_table(
    result,
    radius,
    m_ij,
    n_ant,
    radius_colours,
    pair_colours,
    names=None,
    radius_all=None,
    m_ij_all=None,
):
    """One record per measurement, with team, residual and corrected error.

    *radius* and *m_ij* are the measurements the fit used.  If some were
    excluded from the fit, pass the full set as *radius_all* / *m_ij_all*:
    the excluded ones are then scored too, against the solution they took
    no part in, so their error is simply their residual.

    Each record is a dict with keys ``kind`` (``"radius"`` or ``"pair"``),
    ``i``, ``j`` (``None`` for radii), ``measured``, ``residual`` (fitted
    minus measured, mm), ``leverage``, ``checked``, ``excluded`` and
    ``error`` (the residual divided by ``sqrt(1 - h)``, or NaN when
    unchecked), plus ``colour`` and ``team``.
    """
    r_res, ij_res, pairs = compute_residuals(result, radius, m_ij, n_ant)
    h_rad, h_ij = compute_leverages(result, n_ant, len(ij_res))

    rows = []

    def add(kind, i, j, measured, res, h, colour, excluded=False):
        redundancy = 1.0 - h
        checked = excluded or redundancy >= MIN_REDUNDANCY
        rows.append(
            dict(
                kind=kind,
                i=i,
                j=j,
                measured=measured,
                residual=res,
                leverage=h,
                checked=checked,
                excluded=excluded,
                error=res / np.sqrt(redundancy) if checked else np.nan,
                colour=colour,
                team=team_name(colour, names),
            )
        )

    for i in range(n_ant):
        if not np.isnan(radius[i]):
            add("radius", i, None, radius[i], r_res[i], h_rad[i], radius_colours[i])
    for k, (i, j) in enumerate(pairs):
        add("pair", int(i), int(j), m_ij[i, j], ij_res[k], h_ij[k], pair_colours[(int(i), int(j))])

    if radius_all is not None or m_ij_all is not None:
        ra = radius if radius_all is None else np.asarray(radius_all, dtype=float)
        ma = m_ij if m_ij_all is None else np.asarray(m_ij_all, dtype=float)
        r_all, ij_all, pairs_all = compute_residuals(result, ra, ma, n_ant)
        for i in range(n_ant):
            if np.isnan(radius[i]) and not np.isnan(ra[i]):
                add("radius", i, None, ra[i], r_all[i], 0.0, radius_colours[i], excluded=True)
        for k, (i, j) in enumerate(pairs_all):
            i, j = int(i), int(j)
            if np.isnan(m_ij[i, j]):
                add("pair", i, j, ma[i, j], ij_all[k], 0.0, pair_colours[(i, j)], excluded=True)
    return rows


def _team_stats(rows, outlier_limit):
    errs = np.array([r["error"] for r in rows if r["checked"]])
    n = len(rows)
    n_chk = len(errs)
    if n_chk == 0:
        return dict(n=n, n_checked=0)
    a = np.abs(errs)
    return dict(
        n=n,
        n_checked=n_chk,
        median=float(np.median(a)),
        rms=float(np.sqrt(np.mean(errs**2))),
        p90=float(np.percentile(a, 90)),
        max=float(a.max()),
        outliers=int(np.sum(a > outlier_limit)),
    )


def team_statistics(rows):
    """Per-team statistics of the corrected errors.

    Returns ``(stats, sigma, outlier_limit)`` where *stats* maps
    ``(team, kind)`` to a dict of ``n``, ``n_checked``, ``median``,
    ``rms``, ``p90``, ``max`` (all |error| in mm) and ``outliers``;
    *sigma* is the robust (MAD-based) standard deviation of all checked
    inter-antenna errors and *outlier_limit* = ``OUTLIER_SIGMA * sigma``.
    """
    pair_errs = np.array([r["error"] for r in rows if r["checked"] and r["kind"] == "pair"])
    if len(pair_errs):
        sigma = 1.4826 * float(np.median(np.abs(pair_errs - np.median(pair_errs))))
    else:
        sigma = np.nan
    outlier_limit = OUTLIER_SIGMA * sigma if sigma > 0 else np.inf

    stats = {}
    teams = sorted({r["team"] for r in rows})
    for team in teams:
        for kind in ("pair", "radius"):
            sel = [r for r in rows if r["team"] == team and r["kind"] == kind]
            if sel:
                stats[(team, kind)] = _team_stats(sel, outlier_limit)
    return stats, sigma, outlier_limit


def print_team_report(rows, worst_per_team=3):
    """Print the per-team league table to stdout."""
    stats, sigma, limit = team_statistics(rows)
    teams = sorted({t for (t, _k) in stats})

    print("\nPer-team measurement quality")
    print("  error = residual / sqrt(1 - leverage): the estimated error of each")
    print(f"  measurement in mm.  'checked' = redundancy 1-h >= {MIN_REDUNDANCY};")
    print("  unchecked measurements are the only constraint on something, so the")
    print("  fit cannot tell whether they are right, and they are left out.")
    if np.isfinite(limit):
        print(
            f"  Outlier: |error| > {OUTLIER_SIGMA:.0f} sigma = {limit:.1f} mm "
            f"(robust sigma of all checked inter-antenna errors: {sigma:.2f} mm)."
        )

    hdr = (
        f"  {'Rank':>4}  {'Team':<14} {'N':>4} {'chk':>4} {'median':>7} "
        f"{'RMS':>7} {'p90':>7} {'max':>7} {'outliers':>8}"
    )

    def line(rank, team, s):
        if s["n_checked"] == 0:
            return f"  {rank:>4}  {team:<14} {s['n']:>4} {0:>4}  (nothing checked yet)"
        return (
            f"  {rank:>4}  {team:<14} {s['n']:>4} {s['n_checked']:>4} "
            f"{s['median']:>7.2f} {s['rms']:>7.2f} {s['p90']:>7.2f} "
            f"{s['max']:>7.2f} {s['outliers']:>8d}"
        )

    sections = (("pair", "Inter-antenna distances"), ("radius", "Radius (centre) distances"))
    for kind, title in sections:
        present = [t for t in teams if (t, kind) in stats]
        if not present:
            continue
        ranked = sorted(
            (t for t in present if stats[(t, kind)]["n_checked"] > 0),
            key=lambda t: stats[(t, kind)]["rms"],
        )
        unranked = [t for t in present if t not in ranked]
        print(f"\n  {title} (mm), ranked by RMS error:")
        print(hdr)
        for rank, team in enumerate(ranked, 1):
            print(line(rank, team, stats[(team, kind)]))
        for team in unranked:
            print(line("-", team, stats[(team, kind)]))

    ranked = sorted(
        (t for t in teams if stats.get((t, "pair"), {}).get("n_checked", 0) > 0),
        key=lambda t: stats[(t, "pair")]["rms"],
    )
    if len(ranked) >= 2:
        print(f"\n  Best inter-antenna measurements:  {ranked[0]}")
        print(f"  Worst inter-antenna measurements: {ranked[-1]}")

    if worst_per_team:
        print(f"\n  Largest errors per team (up to {worst_per_team}):")
        for team in teams:
            chk = [r for r in rows if r["team"] == team and r["checked"]]
            chk.sort(key=lambda r: -abs(r["error"]))
            items = []
            for r in chk[:worst_per_team]:
                what = f"r[{r['i']}]" if r["kind"] == "radius" else f"[{r['i']},{r['j']}]"
                flag = "!" if abs(r["error"]) > limit else ""
                flag += " (excl.)" if r.get("excluded") else ""
                items.append(f"{what} {r['error']:+.1f}{flag}")
            print(f"    {team:<14} " + (", ".join(items) if items else "-"))
    return stats


def write_measurements_csv(rows, path):
    """Write the per-measurement table (see :func:`measurement_table`) as CSV.

    Adds an ``outlier`` column: True where the measurement's |error|
    exceeds the outlier limit of :func:`team_statistics`.
    """
    keys = [
        "team", "colour", "kind", "i", "j", "measured",
        "residual", "leverage", "checked", "excluded", "error", "outlier",
    ]
    _stats, _sigma, limit = team_statistics(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in keys}
            out["outlier"] = bool(r["checked"] and abs(r["error"]) > limit)
            for k in ("residual", "leverage", "error"):
                out[k] = "" if np.isnan(out[k]) else f"{out[k]:.3f}"
            w.writerow(out)


def plot_team_residuals(rows, outpath):
    """One plot with a histogram per team, each drawn as an outline in the
    team's colour.

    Every scored measurement of a team is included (radii and inter-antenna
    distances, excluded ones too); unchecked measurements are left out, as
    in the statistics.  The axis spans twice the outlier limit either side
    of zero; errors beyond it are piled into the end bins and counted in the
    title, so a few gross blunders do not squash the rest of the
    distribution.
    """
    import matplotlib.pyplot as plt

    _stats, _sigma, limit = team_statistics(rows)
    teams = sorted({r["team"] for r in rows if r["checked"]})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    title = "Measurement errors by team"
    if teams:
        allv = np.array([r["error"] for r in rows if r["checked"]])
        span = 2.0 * limit if np.isfinite(limit) and limit > 0 else np.max(np.abs(allv))
        span = max(span, 1e-3)
        bins = np.linspace(-span, span, 41)
        n_beyond = int(np.sum(np.abs(allv) > span))
        for team in teams:
            sel = [r for r in rows if r["team"] == team and r["checked"]]
            err = np.clip([r["error"] for r in sel], -span, span)
            colour = sel[0]["colour"]
            colour = colour if colour not in NO_COLOUR else "#777777"
            n_out = int(sum(abs(r["error"]) > limit for r in sel))
            ax.hist(err, bins=bins, histtype="step", linewidth=2.2, color=colour,
                    label=f"{team}: n={len(sel)}, outliers={n_out}")
        if np.isfinite(limit):
            for x in (-limit, limit):
                ax.axvline(x, color="black", linestyle="--", linewidth=0.8)
        if n_beyond:
            title += f"\n({n_beyond} beyond \u00b1{span:.1f} mm are piled into the end bins)"
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("residual / sqrt(1 - leverage) (mm); dashed lines: outlier limit")
    ax.set_ylabel("number of measurements")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
