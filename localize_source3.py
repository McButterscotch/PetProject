# localize_sources_with_grid.py
# usage:
#   python localize_sources_with_grid.py output/output_vereos.root \
#       --n-sources 2 --emin 350 --emax 650 --cwin 6.0 --use-tof --plot
#
# Notes:
# - Prefers Singles5 if present; otherwise falls back gracefully.
# - "Grid" comes from per-crystal centers computed from the ROOT (only crystals that recorded hits).
# - Outputs (a) estimated centers and (b) nearest grid-crystal centers (with IDs) for direct comparison.

import argparse, re
import numpy as np
import uproot
import matplotlib.pyplot as plt

C_LIGHT_MM_PER_NS = 299.792458  # mm/ns

# ---------- tree & branch helpers ----------
def list_preferred_trees(f):
    names = [k for k in f.keys()]
    singles = []
    for k in names:
        m = re.search(r"Singles\s*([0-9]+)", k, re.IGNORECASE)
        if m:
            singles.append((int(m.group(1)), k))
        elif re.search(r"Singles(?!\s*[0-9])", k, re.IGNORECASE):
            singles.append((0, k))
    # sort preferring highest index first (Singles5 .. 1 .. Singles)
    singles_sorted = [k for _, k in sorted(singles, key=lambda x: -x[0])]
    hits = [k for k in names if re.search(r"\bHits\b", k, re.IGNORECASE)]
    return singles_sorted + hits

def arrays_from_preferred_tree(f, need):
    for tn in list_preferred_trees(f):
        t = f[tn]
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
            arrs = t.arrays(list(found.values()), library="np", how=dict)
            return arrs, found, tn
    raise RuntimeError("No suitable Singles*/Hits tree with required branches found.")

def auto_energy_to_keV(E):
    Es = E[(E > 0) & np.isfinite(E)]
    med = np.median(Es) if Es.size else 0.0
    return (E * 1000.0) if (0.1 < med < 2.0) else E

# ---------- coincidences & localization ----------
def build_coincidences(times_ns, ids=None, window_ns=6.0):
    order = np.argsort(times_ns)
    t = times_ns[order]
    pairs, j = [], 0
    for i in range(len(t)):
        while j < len(t) and (t[j] - t[i]) <= window_ns:
            j += 1
        for k in range(i + 1, j):
            if ids is not None and ids[order[i]] == ids[order[k]]:
                continue
            pairs.append((order[i], order[k]))
    return pairs

def lor_mid_dir(r1, r2):
    d = r2 - r1
    n = np.linalg.norm(d)
    return (0.5*(r1+r2), d/n if n>0 else np.array([1.,0.,0.]))

def tof_point(mid, u, dt_ns):
    return mid + (0.5 * C_LIGHT_MM_PER_NS * dt_ns) * u

# ---------- simple k-means++ (no sklearn) ----------
def kmeans_pp(X, k, max_iter=100, tol=1e-4, rng=None):
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[rng.integers(n)]
    for c in range(1, k):
        d2 = np.min(((X[:,None,:]-centers[None,:c,:])**2).sum(axis=2), axis=1)
        probs = d2 / d2.sum()
        centers[c] = X[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d2 = ((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i]
                                for i in range(k)])
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers, labels

# ---------- crystal grid from ROOT ----------
def crystal_centers_from_hits(X, Y, Z, VID, min_hits=1):
    """
    Compute per-crystal centers by averaging positions per PreStepUniqueVolumeID.
    Only crystals with >= min_hits after selection are returned.
    """
    if VID is None:
        raise RuntimeError("PreStepUniqueVolumeID (or equivalent) not found; cannot form crystal grid.")
    # group by ID
    order = np.argsort(VID, kind="mergesort")
    VID_sorted = VID[order]
    Xs, Ys, Zs = X[order], Y[order], Z[order]
    # find group boundaries
    edges = np.flatnonzero(np.r_[True, VID_sorted[1:] != VID_sorted[:-1], True])
    centers = []
    ids = []
    for a, b in zip(edges[:-1], edges[1:]):
        if (b - a) >= min_hits:
            centers.append([Xs[a:b].mean(), Ys[a:b].mean(), Zs[a:b].mean()])
            ids.append(VID_sorted[a])
    return np.asarray(centers), np.asarray(ids)

def nearest_grid(crystal_centers, crystal_ids, points):
    """
    For each point, find nearest crystal center (brute-force).
    Returns indices into crystal_centers and distances.
    """
    if crystal_centers.shape[0] == 0:
        raise RuntimeError("No crystal centers available to compare against.")
    # (Npoints x Ncrystals) distances squared
    d2 = ((points[:,None,:] - crystal_centers[None,:,:])**2).sum(axis=2)
    j = np.argmin(d2, axis=1)
    return j, np.sqrt(d2[np.arange(points.shape[0]), j])

# ---------- plotting helpers ----------
def choose_plane(X, Y, Z):
    vars_ = np.array([np.var(X), np.var(Y), np.var(Z)])
    a, b = vars_.argsort()[-2:][::-1]
    coords = [X, Y, Z]; labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    return coords[a], coords[b], labels[a], labels[b], a, b

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to ROOT file")
    ap.add_argument("--emin", type=float, default=350.0, help="Photopeak min (keV)")
    ap.add_argument("--emax", type=float, default=650.0, help="Photopeak max (keV)")
    ap.add_argument("--cwin", type=float, default=6.0, help="Coincidence window (ns)")
    ap.add_argument("--use-tof", action="store_true", help="Localize along LOR using Δt")
    ap.add_argument("--n-sources", type=int, default=2, help="How many sources to estimate")
    ap.add_argument("--min-hits-per-crystal", type=int, default=1, help="Min hits to keep a crystal in grid")
    ap.add_argument("--subsample", type=int, default=0, help="Use at most N localized points (0=all)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    with uproot.open(args.root) as f:
        need = dict(
            x = ["PostPosition_X","GlobalPosX","Position_X","PosX"],
            y = ["PostPosition_Y","GlobalPosY","Position_Y","PosY"],
            z = ["PostPosition_Z","GlobalPosZ","Position_Z","PosZ"],
            E = ["TotalEnergyDeposit","Energy","Edep","E"],
            t = ["GlobalTime","Time","Global_Time"],
            id= ["PreStepUniqueVolumeID","VolumeID","CrystalID"]
        )
        arrs, branches, used_tree = arrays_from_preferred_tree(f, need)

    X, Y, Z = arrs[branches["x"]], arrs[branches["y"]], arrs[branches["z"]]
    E = auto_energy_to_keV(arrs[branches["E"]])
    T = arrs[branches["t"]]
    VID = arrs.get(branches["id"], None)

    # photopeak selection
    keep = (E >= args.emin) & (E <= args.emax) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z, T = X[keep], Y[keep], Z[keep], T[keep]
    VID = VID[keep] if VID is not None else None

    # build coincidences
    pairs = build_coincidences(T, ids=VID, window_ns=args.cwin)
    if not pairs:
        raise RuntimeError("No coincidences formed; check energy/time windows or tree choice.")

    # localize points
    pts = []
    for i, j in pairs:
        r1 = np.array([X[i], Y[i], Z[i]])
        r2 = np.array([X[j], Y[j], Z[j]])
        mid, u = lor_mid_dir(r1, r2)
        p = tof_point(mid, u, float(T[j] - T[i])) if args.use_tof else mid
        pts.append(p)
    P = np.asarray(pts)

    # optional subsample for speed
    if args.subsample and P.shape[0] > args.subsample:
        sel = np.random.default_rng(0).choice(P.shape[0], args.subsample, replace=False)
        P_used = P[sel]
    else:
        P_used = P

    # estimate source centers by clustering
    centers0, labels_used = kmeans_pp(P_used, args.n_sources, rng=0)
    # refine on all points
    d2_all = ((P[:,None,:]-centers0[None,:,:])**2).sum(axis=2)
    labels_all = np.argmin(d2_all, axis=1)
    centers = np.array([P[labels_all==i].mean(axis=0) for i in range(args.n_sources)])

    # build crystal grid from ROOT and compare
    grid_centers, grid_ids = crystal_centers_from_hits(X, Y, Z, VID, min_hits=args.min_hits_per_crystal)
    jj, dmin = nearest_grid(grid_centers, grid_ids, centers)

    # report
    print(f"Tree used: {used_tree}")
    Tspan_s = float(T.max() - T.min()) * 1e-9
    print(f"Acquisition time (from ROOT GlobalTime): {Tspan_s:.6g} s")
    print(f"Localized coincidences: {P.shape[0]:,}")
    print(f"Crystals in grid (with ≥{args.min_hits_per_crystal} hits): {grid_centers.shape[0]:,}\n")

    for i in range(args.n_sources):
        est = centers[i]
        idx = jj[i]
        grid = grid_centers[idx]
        cid = grid_ids[idx]
        print(f"Source {i}:")
        print(f"  Estimated center (mm): [{est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f}]")
        print(f"  Nearest crystal center (mm): [{grid[0]:.2f}, {grid[1]:.2f}, {grid[2]:.2f}]  (ID={cid})")
        print(f"  Distance to grid center: {dmin[i]:.2f} mm\n")

    # plots
    if args.plot:
        # choose a good transaxial plane for display
        Agrid, Bgrid, labA, labB, ia, ib = choose_plane(grid_centers[:,0], grid_centers[:,1], grid_centers[:,2])
        plt.figure(figsize=(7,7))
        plt.scatter(Agrid, Bgrid, s=6, alpha=0.35, linewidths=0, label="Crystal grid (hit crystals)")

        Aloc, Bloc, _, _, _, _ = choose_plane(P[:,0], P[:,1], P[:,2])
        plt.scatter(Aloc, Bloc, s=2, alpha=0.25, linewidths=0, label="Localized points")

        # estimated centers and snapped grid points (same plane indices ia,ib)
        plt.scatter(centers[:,ia], centers[:,ib], marker="x", s=160, linewidths=2, label="Estimated centers")
        plt.scatter(grid_centers[jj,ia], grid_centers[jj,ib], marker="s", s=60, linewidths=1, facecolors="none", label="Nearest crystal")

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(labA.replace(" (mm)","")); plt.ylabel(labB.replace(" (mm)",""))
        plt.title("Source localization vs. detector crystal grid")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
