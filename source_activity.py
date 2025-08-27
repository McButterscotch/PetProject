# estimate_activity_from_coinc.py
# usage:
#   python estimate_activity_from_coinc.py output/output_vereos.root \
#          --stats output/stats_vereos.txt \
#          --sources "(-50,0,0),(50,0,0)" \
#          --emin 350 --emax 650 --cwin 6.0 --use-tof
#
# What it prints:
#   - coincidence counts per source
#   - relative activity ratio
#   - (if --stats provided) absolute activity estimates in Bq
#
# What it plots:
#   - transaxial scatter of hit positions (auto-chosen plane)
#   - TOF-localized points colored by source assignment

import re
import ast
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt

C_LIGHT_MM_PER_NS = 299.792458  # mm/ns

# ---------- helpers ----------
def arrays_from_any_tree(f, need=()):
    """Return dict of arrays from the first tree that contains the requested branches."""
    for k in f.keys():
        if not (k.lower().startswith("tree hits") or k.lower().startswith("tree singles")
                or k.lower().startswith("hits") or k.lower().startswith("singles")):
            continue
        t = f[k]
        keys_lc = {kk.lower(): kk for kk in t.keys()}
        found = {}
        for name, aliases in need.items():
            actual = None
            for a in aliases:
                if a.lower() in keys_lc:
                    actual = keys_lc[a.lower()]
                    break
            if actual is None:
                found = None
                break
            found[name] = actual
        if found is not None:
            return t.arrays(list(found.values()), library="np", how=dict), found, k
    raise RuntimeError("Could not find a tree with required branches.")

def auto_energy_to_keV(E):
    # crude unit detection: if median is ~MeV (0.1..2), convert to keV
    Es = E[(E > 0) & np.isfinite(E)]
    med = np.median(Es) if Es.size else 0.0
    return (E * 1000.0) if (0.1 < med < 2.0) else E

def build_coincidences(times_ns, ids=None, window_ns=6.0):
    """Greedy time-window pairing (all pairs within window)."""
    order = np.argsort(times_ns)
    t = times_ns[order]
    idx = np.arange(len(t))
    pairs = []
    j = 0
    for i in range(len(t)):
        while j < len(t) and (t[j] - t[i]) <= window_ns:
            j += 1
        # pair i with hits (i+1 ... j-1)
        for k in range(i + 1, j):
            if ids is not None and ids[order[i]] == ids[order[k]]:
                continue
            pairs.append((order[i], order[k]))
    return pairs

def lor_mid_dir(r1, r2):
    d = r2 - r1
    n = np.linalg.norm(d)
    if n == 0:
        return r1, np.array([1.0, 0.0, 0.0])
    return 0.5*(r1+r2), d/n

def tof_point(mid, u, dt_ns):
    s = 0.5 * C_LIGHT_MM_PER_NS * dt_ns
    return mid + s * u

def choose_plane(X, Y, Z):
    vars_ = np.array([np.var(X), np.var(Y), np.var(Z)])
    a, b = vars_.argsort()[-2:][::-1]
    coords = [X, Y, Z]
    labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    return coords[a], coords[b], labels[a], labels[b], a, b

def parse_sources(s):
    """
    Parse sources like:
      "(-50,0,0),(50,0,0)"
    or
      "[-50,0,0] [50,0,0]"
    Returns numpy array of shape (Nsrc,3).
    """
    # find all groups like (x,y,z) or [x,y,z]
    matches = re.findall(r'[\(\[]\s*([^\)\]]+)\s*[\)\]]', s)
    coords = []
    for m in matches:
        parts = [float(p) for p in m.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad source specification: {m}")
        coords.append(parts)
    return np.array(coords, dtype=float)

def parse_stats_file(path):
    """
    Very light-weight parser. Looks for numbers of generated primaries per source
    and total run time if present. You can adapt the regexes to your Stats format.
    """
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    # Example patterns (adjust to your actual stats file lines)
    #  - "Nb of primaries for source hot_sphere_source_1 : 20000"
    #  - "Elapsed time : 2 s"
    prims = {}
    for name, cnt in re.findall(r"primar(?:y|ies).*?source\s+([^\s:]+)\s*[:=]\s*([0-9]+)", text, re.IGNORECASE):
        prims[name] = prims.get(name, 0) + int(cnt)

    # elapsed simulated time
    T = None
    m = re.search(r"(elapsed|run).{0,20}([0-9]+(?:\.[0-9]+)?)\s*s", text, re.IGNORECASE)
    if m:
        T = float(m.group(2))

    return prims, T

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to ROOT coincidences/hits file")
    ap.add_argument("--stats", default=None, help="Path to SimulationStatisticsActor text file (optional)")
    ap.add_argument("--sources", default="(-50,0,0),(50,0,0)", help="Comma-separated tuples in mm")
    ap.add_argument("--emin", type=float, default=350.0)
    ap.add_argument("--emax", type=float, default=650.0)
    ap.add_argument("--cwin", type=float, default=6.0, help="Coincidence window [ns]")
    ap.add_argument("--use-tof", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    SRC = parse_sources(args.sources)  # shape (Nsrc, 3)

    f = uproot.open(args.root)
    need = dict(
        x=["PostPosition_X","GlobalPosX","Position_X","PosX"],
        y=["PostPosition_Y","GlobalPosY","Position_Y","PosY"],
        z=["PostPosition_Z","GlobalPosZ","Position_Z","PosZ"],
        E=["TotalEnergyDeposit","Energy","Edep","E"],
        t=["GlobalTime","Time","Global_Time"],
        id=["PreStepUniqueVolumeID","VolumeID","CrystalID"]
    )
    arrs, branches, used_tree = arrays_from_any_tree(f, need)
    X, Y, Z = arrs[branches["x"]], arrs[branches["y"]], arrs[branches["z"]]
    E = auto_energy_to_keV(arrs[branches["E"]])
    T = arrs[branches["t"]]
    VID = arrs[branches["id"]] if branches.get("id") in arrs else None

    # Photopeak selection
    keep = (E >= args.emin) & (E <= args.emax) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z, T = X[keep], Y[keep], Z[keep], T[keep]
    VID = VID[keep] if VID is not None else None

    # Build coincidences
    pairs = build_coincidences(T, ids=VID, window_ns=args.cwin)

    # Localize events
    assigned_src = []
    loc_points = []
    for i, j in pairs:
        r1 = np.array([X[i], Y[i], Z[i]])
        r2 = np.array([X[j], Y[j], Z[j]])
        mid, u = lor_mid_dir(r1, r2)

        if args.use_tof:
            dt = float(T[j] - T[i])  # ns
            p  = tof_point(mid, u, dt)
        else:
            p  = mid

        loc_points.append(p)
        # assign to nearest source by Euclidean distance
        d2 = np.sum((SRC - p)**2, axis=1)
        assigned_src.append(int(np.argmin(d2)))

    loc_points = np.array(loc_points)
    assigned_src = np.array(assigned_src, dtype=int)

    # Coincidence counts per source
    counts = np.bincount(assigned_src, minlength=len(SRC))
    total_coinc = counts.sum()

    # Relative activity estimate (assuming identical detection efficiency across sources)
    rel = counts / counts.sum() if counts.sum() > 0 else counts

    print(f"Tree used: {used_tree}")
    print(f"Energy window: [{args.emin:.1f},{args.emax:.1f}] keV; Coinc. window: {args.cwin:.2f} ns")
    print(f"Coincidences: {total_coinc:,}")
    for sidx, c in enumerate(counts):
        print(f"  Source {sidx}: coincidences={c:,}  relative={rel[sidx]:.3f}")

    # Absolute activity via sensitivity calibration (optional)
    if args.stats:
        prims, Tstats = parse_stats_file(args.stats)
        # Fallback to time range in data if stats time not found
        if Tstats is None:
            Tstats = float(np.max(T) - np.min(T))
        # Ground-truth total activity from stats: A_total = (sum primaries)/T
        Nprim_total = sum(prims.values()) if prims else None
        if Nprim_total is None or Tstats <= 0:
            print("\n[warn] Could not read primaries/time from stats file; skipping absolute activity.")
        else:
            A_total_true = Nprim_total / Tstats  # decays per second = Bq
            # Empirical sensitivity (coincidences per Bq per second)
            S = total_coinc / (A_total_true * Tstats) if A_total_true > 0 else np.nan
            print(f"\nStats: primaries(total)={Nprim_total:,}, simulated time={Tstats:g} s")
            print(f"Empirical sensitivity S ≈ {S:.6f} coincidences / (Bq·s)")
            # Activity per source from counts_i = S * A_i * T
            A_est = counts / (S * Tstats) if S > 0 else np.zeros_like(counts, dtype=float)
            for sidx, a in enumerate(A_est):
                print(f"  Estimated activity Source {sidx}: {a:.1f} Bq")
            print(f"  Sum est: {A_est.sum():.1f} Bq (true total ≈ {A_total_true:.1f} Bq)")
    else:
        print("\nTip: pass --stats stats_vereos.txt to get absolute activities (Bq).")

    # -------- optional plot ----------
    if args.plot:
        # hits plane
        A1, B1, labA, labB, ia, ib = choose_plane(X, Y, Z)
        plt.figure(figsize=(7,7))
        plt.scatter(A1, B1, s=1, alpha=0.25)
        for s in SRC:
            plt.scatter(s[ia], s[ib], marker="x", s=80)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Transaxial detection position (Hits)")
        plt.xlabel(labA.replace(" (mm)",""))
        plt.ylabel(labB.replace(" (mm)",""))
        plt.tight_layout()

        # localized points colored by source
        if loc_points.size:
            plt.figure(figsize=(7,7))
            A2, B2, labA2, labB2, ia2, ib2 = choose_plane(loc_points[:,0], loc_points[:,1], loc_points[:,2])
            plt.scatter(A2, B2, c=assigned_src, s=4, alpha=0.6, cmap="tab10")
            for s in SRC:
                plt.scatter(s[ia2], s[ib2], marker="x", s=100)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("Localized coincidence points (TOF or midpoints), colored by source assignment")
            plt.xlabel(labA2.replace(" (mm)",""))
            plt.ylabel(labB2.replace(" (mm)",""))
            plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
