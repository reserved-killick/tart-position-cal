"""CLI entry point for tart-position-cal antenna position calibration."""

import argparse
import json
import os
import sys

import numpy as np

from tart_position_cal.calibrate import (
    apply_exclusions,
    calibrate,
    calibrate_irls,
    compute_residuals,
    fetch_initial_positions,
    load_initial_positions,
    load_measurements,
    parse_exclusions,
    plot_differences,
    plot_positions,
    plot_residual_histogram,
    print_residual_report,
    result_to_json,
    result_to_positions,
)
from tart_position_cal.teams import (
    NO_COLOUR,
    load_cell_colours,
    measurement_table,
    parse_team_names,
    plot_team_residuals,
    print_team_report,
    write_measurements_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate TART antenna positions from survey measurements."
    )
    parser.add_argument(
        "--measurements",
        required=True,
        help="Path to .ods file with antenna distance measurements.",
    )
    parser.add_argument(
        "--sheet",
        default="Sheet1",
        help="Name of the worksheet holding the measurements (default: Sheet1).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="LIST",
        help="Leave measurements out of the fit: rN for the radius of antenna "
        "N, I-J for the distance between antennas I and J, comma-separated "
        "(e.g. 'r18,r19,3-12'). Repeatable. Excluded measurements are still "
        "scored in the per-team statistics.",
    )
    parser.add_argument(
        "--team",
        action="append",
        metavar="COLOUR=NAME",
        help="Name the team whose cells have background COLOUR, e.g. "
        "'#ff0000=Team Red'. Repeat for each team. Teams are found from the "
        "cell colours either way; this only sets the names shown.",
    )
    parser.add_argument(
        "--no-teams",
        action="store_true",
        help="Skip the per-team statistics, even if the cells are colour-coded.",
    )
    parser.add_argument(
        "--measurements-csv",
        help="Write a per-measurement table (team, residual, leverage, "
        "corrected error) to this CSV file.",
    )
    parser.add_argument(
        "--positions",
        help="Path to JSON file with initial antenna positions (metres).",
    )
    parser.add_argument(
        "--api",
        help="TART API base URL to fetch initial positions from (e.g. https://api.elec.ac.nz/tart/na-unam).",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=30.0,
        help="Network timeout in seconds for --api requests (default: 30).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output calibrated positions JSON file.",
    )
    parser.add_argument(
        "--rot-index",
        type=int,
        default=0,
        help="Antenna index used for global rotation constraint.",
    )
    parser.add_argument(
        "--rot-degrees",
        type=float,
        default=0.0,
        help="Target geographic angle in degrees for the constrained antenna.",
    )
    parser.add_argument(
        "--chirality-index",
        type=int,
        default=None,
        help="Antenna index used to break reflection (chirality) degeneracy. "
        "Requires --rot-index and --chirality-sign.",
    )
    parser.add_argument(
        "--chirality-sign",
        choices=["positive", "negative"],
        default=None,
        help="Required with --chirality-index. The sign of "
        "p_ref x p_chirality in the actual, as-built array (NOT the "
        "design/initial-positions file) -- determine this by observing "
        "whether the chirality antenna lies to the left (positive) or "
        "right (negative) of the reference antenna's bearing line. See "
        "doc/SYMMETRY.md.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum optimizer iterations (default: 500).",
    )
    parser.add_argument(
        "--max-position-error",
        type=float,
        default=4200.0,
        help="Half-width of search bounds around initial positions in mm (default: 4200).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generation of diagnostic plots.",
    )
    parser.add_argument(
        "--plot-dir",
        default=".",
        help="Directory to save diagnostic plots (default: current directory).",
    )
    parser.add_argument(
        "--irls",
        action="store_true",
        help="Use Iteratively Reweighted Least Squares for robust calibration.",
    )
    parser.add_argument(
        "--irls-weight-fn",
        choices=["tukey", "huber"],
        default="tukey",
        help="Weight function for IRLS: 'tukey' (bisquare) or 'huber' (default: tukey).",
    )
    parser.add_argument(
        "--irls-max-iter",
        type=int,
        default=10,
        help="Maximum IRLS outer iterations (default: 10).",
    )
    parser.add_argument(
        "--title",
        help="Optional title prepended to all output file names (e.g. 'site-name').",
    )

    args = parser.parse_args()

    # --- Load measurements ---
    print(f"Loading measurements from {args.measurements}")
    radius, m_ij, n_ant = load_measurements(args.measurements, sheet=args.sheet)
    radius_all, m_ij_all = radius, m_ij
    excl_radii, excl_pairs = parse_exclusions(args.exclude, n_ant)
    if excl_radii or excl_pairs:
        radius, m_ij = apply_exclusions(radius, m_ij, excl_radii, excl_pairs)
        names = [f"r{i}" for i in sorted(excl_radii)] + [f"{i}-{j}" for i, j in sorted(excl_pairs)]
        print(f"  Excluding from the fit: {', '.join(names)}")
    n_pairs = int(np.sum(~np.isnan(m_ij[np.triu_indices(n_ant, 1)])))
    n_rad = int(np.sum(~np.isnan(radius)))
    print(f"  Found {n_ant} antennas: {n_rad} radius and {n_pairs} inter-antenna measurements")
    offdiag = ~np.isnan(m_ij) & ~np.eye(n_ant, dtype=bool)
    n_per_ant = offdiag.sum(axis=1) + ~np.isnan(radius)
    thin = [i for i in range(n_ant) if n_per_ant[i] < 3]
    if thin:
        print(
            "  WARNING: fewer than 3 measurements to antenna(s) "
            + ", ".join(f"{i} ({n_per_ant[i]})" for i in thin)
            + "; their positions are poorly constrained or not at all."
        )

    # --- Load initial positions ---
    if args.positions:
        print(f"Loading initial positions from {args.positions}")
        initial_guess = load_initial_positions(args.positions)
    elif args.api:
        print(f"Fetching initial positions from {args.api}")
        initial_guess = fetch_initial_positions(args.api, timeout=args.api_timeout)
    else:
        parser.error("One of --positions or --api is required.")
        sys.exit(1)

    if initial_guess.shape[0] != n_ant:
        print(
            f"WARNING: measurements have {n_ant} antennas but "
            f"positions have {initial_guess.shape[0]}"
        )

    if args.chirality_index is not None and args.chirality_sign is None:
        parser.error(
            "--chirality-index requires --chirality-sign (positive or "
            "negative), determined from the actual as-built array -- "
            "see doc/SYMMETRY.md."
        )
    chirality_sign = {"positive": 1.0, "negative": -1.0}.get(args.chirality_sign)

    # --- Calibrate ---
    if args.irls:
        print(
            f"Running IRLS calibration (rot_index={args.rot_index}, "
            f"rot_degrees={args.rot_degrees}, "
            f"weight_fn={args.irls_weight_fn})"
        )
        result = calibrate_irls(
            radius,
            m_ij,
            initial_guess,
            rot_index=args.rot_index,
            rot_degrees=args.rot_degrees,
            chirality_index=args.chirality_index,
            chirality_sign=chirality_sign,
            max_position_error=args.max_position_error,
            maxiter=args.max_iter,
            irls_max_iter=args.irls_max_iter,
            weight_function=args.irls_weight_fn,
        )
        print(f"  Final objective: {result.fun:.4f}")
        print(f"  Function evaluations: {result.nfev}")
    else:
        print(
            f"Running calibration (rot_index={args.rot_index}, "
            f"rot_degrees={args.rot_degrees})"
        )
        result = calibrate(
            radius,
            m_ij,
            initial_guess,
            rot_index=args.rot_index,
            rot_degrees=args.rot_degrees,
            chirality_index=args.chirality_index,
            chirality_sign=chirality_sign,
            max_position_error=args.max_position_error,
            maxiter=args.max_iter,
        )

        print(f"  Optimisation finished: {result.message}")
        print(f"  Final objective: {result.fun:.4f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")

    # --- Save output ---
    json_out = result_to_json(result, n_ant, title=args.title)

    def _titled(path):
        """Return path with --title prepended to the filename, if set."""
        if not args.title:
            return path
        d, base = os.path.split(path)
        return os.path.join(d, f"{args.title}_{base}")

    output_path = _titled(args.output)
    with open(output_path, "w") as f:
        json.dump(json_out, f)
    print(f"Calibrated positions written to {output_path}")

    # --- Diagnostics ---
    print_residual_report(result, radius, m_ij, n_ant)
    if excl_radii or excl_pairs:
        r_all, _, _ = compute_residuals(result, radius_all, m_ij_all, n_ant)
        print("\nExcluded measurements (not fitted), residual against the solution:")
        for i in sorted(excl_radii):
            print(f"  r[{i}] = {r_all[i]:+.1f}")
        x = result_to_positions(result, n_ant)
        for i, j in sorted(excl_pairs):
            res_ij = np.hypot(*(x[i] - x[j])) - m_ij_all[i, j]
            print(f"  res[{i},{j}] = {res_ij:+.1f}")

    # --- Per-team statistics (colour-coded cells) ---
    team_rows = None
    if not args.no_teams or args.measurements_csv:
        try:
            radius_colours, pair_colours = load_cell_colours(
                args.measurements, n_ant, sheet=args.sheet
            )
        except Exception as exc:  # colours are a bonus; never fail the calibration
            print(f"\nCould not read cell colours for team statistics: {exc}")
        else:
            team_rows = measurement_table(
                result,
                radius,
                m_ij,
                n_ant,
                radius_colours,
                pair_colours,
                names=parse_team_names(args.team),
                radius_all=radius_all,
                m_ij_all=m_ij_all,
            )
            colours = {r["colour"] for r in team_rows} - set(NO_COLOUR)
            if args.no_teams:
                pass
            elif colours:
                print_team_report(team_rows)
            else:
                print("\n(No colour-coded cells found, so no per-team statistics.)")
            if args.measurements_csv:
                csv_path = _titled(args.measurements_csv)
                write_measurements_csv(team_rows, csv_path)
                print(f"Per-measurement table written to {csv_path}")

    # --- Plots ---
    if not args.no_plots:
        final_pos = result_to_positions(result, n_ant)
        plot_path = _titled(os.path.join(args.plot_dir, "final_positions.png"))
        plot_positions(initial_guess, final_pos, plot_path)
        print(f"Saved {os.path.basename(plot_path)} to {args.plot_dir}")

        plot_path = _titled(os.path.join(args.plot_dir, "differences.png"))
        plot_differences(initial_guess, final_pos, plot_path)
        print(f"Saved {os.path.basename(plot_path)} to {args.plot_dir}")

        _, ij_res, _ = compute_residuals(result, radius, m_ij, n_ant)
        plot_path = _titled(os.path.join(args.plot_dir, "residual_histogram.png"))
        plot_residual_histogram(ij_res, plot_path)
        print(f"Saved {os.path.basename(plot_path)} to {args.plot_dir}")

        if team_rows is not None and not args.no_teams:
            if {r["colour"] for r in team_rows} - set(NO_COLOUR):
                plot_path = _titled(os.path.join(args.plot_dir, "team_residuals.png"))
                plot_team_residuals(team_rows, plot_path)
                print(f"Saved {os.path.basename(plot_path)} to {args.plot_dir}")
