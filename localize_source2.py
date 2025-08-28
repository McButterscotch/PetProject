# localize_sources_from_data.py
# usage:
#   python localize_sources_from_data.py output/output_vereos.root \
#       --n-sources 2 --emin 350 --emax 650 --cwin 6.0 --use-tof --plot
#
# Notes:
# - Uses TOF by default (recommended). If you omit --use-tof it clusters LOR midpoints.
# - No predefined positions required. It outputs estimated 3D centers + covariance.

import argparse, numpy as np, uproot, matplotlib.pyplot as plt

C_LIGHT_MM_PER_NS = 299.792458  # mm/ns

# ---------- utils to read ROOT ----------
def arrays_from_any_tree(f, need=()):
    for k in f.keys():
        lk = k.lower()
        if not (lk.startswith("tree hits") or lk.startswith("tree singles")
                or lk.startswith("hits") or lk.startswith("singles")):
            continue
        t = f[k]
        keys_lc = {kk.lower(): kk for kk in t.keys()}
        found = {}
        for name, aliases in need.items():
            actual = None
            for a in aliases:
                if a.lower() in keys_lc:
                    actual = keys_lc[a.lower()]; break
            if actual is None:
                found = None; break
            found[name] = actual
        if found is not None:
            arrs = t.arrays(list(found.values()), library="np", how=dict)
            return arrs, found, k
    raise RuntimeError("No suitable tree with requested branches found.")

def auto_energy_to_keV(E):
    Es = E[(E > 0) & np.isfinite(E)]
    med = np.median(Es) if Es.size else 0.0
    return (E * 1000.0) if (0.1 < med < 2.0) else E

# ---------- coincidence building ----------
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
    d = r2 - r1; n = np.linalg.norm(d)
    return (0.5*(r1+r2), d/n if n>0 else np.array([1.,0.,0.]))

def tof_point(mid, u, dt_ns):
    return mid + (0.5*C_LIGHT_MM_PER_NS*dt_ns) * u

# ---------- simple k-means++ clustering ----------
def kmeans_pp(X, k, max_iter=100, tol=1e-4, rng=None):
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    # init: first center random; others by D^2 sampling
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[rng.integers(n)]
    for c in range(1, k):
        d2 = np.min(((X[:,None,:]-centers[None,:c,:])**2).sum(axis=2), axis=1)
        probs = d2 / d2.sum()
        centers[c] = X[rng.choice(n, p=probs)]
    # iterate
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        d2 = ((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        new_centers = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i]
                                for i in range(k)])
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol: break
    return centers, labels

def cluster_stats(X, labels, k):
    stats = []
    for i in range(k):
        Xi = X[labels==i]
        if Xi.size == 0:
            stats.append(dict(n=0, mean=np.full(3, np.nan), cov=np.full((3,3), np.nan)))
        else:
            mu = Xi.mean(axis=0)
            cov = np.cov(Xi.T) if Xi.shape[0] > 1 else np.zeros((3,3))
            stats.append(dict(n=Xi.shape[0], mean=mu, cov=cov))
    return stats

# ---------- plotting helpers ----------
def choose_plane(X, Y, Z):
    vars_ = np.array([np.var(X), np.var(Y), np.var(Z)])
    a, b = vars_.argsort()[-2:][::-1]
    coords = [X, Y, Z]; labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    return coords[a], coords[b], labels[a], labels[b], a, b

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to ROOT file (Singles/Hits)")
    ap.add_argument("--emin", type=float, default=350.0)
    ap.add_argument("--emax", type=float, default=650.0)
    ap.add_argument("--cwin", type=float, default=6.0, help="Coincidence window [ns]")
    ap.add_argument("--use-tof", action="store_true", help="Localize along LOR using Δt")
    ap.add_argument("--n-sources", type=int, default=2, help="How many sources to estimate")
    ap.add_argument("--subsample", type=int, default=0, help="Use at most N localized points (0=all)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # read arrays
    with uproot.open(args.root) as f:
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

    # photopeak selection
    keep = (E >= args.emin) & (E <= args.emax) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z, T = X[keep], Y[keep], Z[keep], T[keep]
    VID = VID[keep] if VID is not None else None

    # build coincidences
    pairs = build_coincidences(T, ids=VID, window_ns=args.cwin)
    if not pairs:
        raise RuntimeError("No coincidences formed; check energy/time windows.")

    # localize per pair
    pts = []
    for i, j in pairs:
        r1 = np.array([X[i], Y[i], Z[i]])
        r2 = np.array([X[j], Y[j], Z[j]])
        mid, u = lor_mid_dir(r1, r2)
        if args.use_tof:
            dt = float(T[j] - T[i])
            p = tof_point(mid, u, dt)
        else:
            p = mid
        pts.append(p)
    P = np.asarray(pts)

    # optional subsample for speed
    if args.subsample and P.shape[0] > args.subsample:
        sel = np.random.default_rng(0).choice(P.shape[0], args.subsample, replace=False)
        P_sub = P[sel]
    else:
        P_sub = P

    # k-means++ clustering to estimate source centers
    centers, labels_sub = kmeans_pp(P_sub, args.n_sources, rng=0)
    # assign all points to nearest found centers (for counts and refined stats)
    d2_all = ((P[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    labels_all = np.argmin(d2_all, axis=1)

    # recompute centers on all data for stability
    centers_refined, _ = kmeans_pp(P, args.n_sources, rng=0)
    d2_all = ((P[:,None,:]-centers_refined[None,:,:])**2).sum(axis=2)
    labels_all = np.argmin(d2_all, axis=1)

    stats = cluster_stats(P, labels_all, args.n_sources)

    # print results
    print(f"Tree used: {used_tree}")
    Tspan_s = float(T.max() - T.min()) * 1e-9
    print(f"Acquisition time (from ROOT GlobalTime): {Tspan_s:.6g} s")
    print(f"Localized coincidences: {P.shape[0]:,}")
    for i, st in enumerate(stats):
        mu = st["mean"]; n = st["n"]
        cov = st["cov"]
        std = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(3, np.nan)
        print(f"Source {i}: n={n:,}  center ≈ [{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}] mm"
              f"  spread σ ≈ [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}] mm")

    # plots
    if args.plot:
        # hits plane (for context)
        A1, B1, labA, labB, ia, ib = choose_plane(X, Y, Z)
        plt.figure(figsize=(7,7))
        plt.scatter(A1, B1, s=1, alpha=0.25)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Transaxial detection position (Hits)")
        plt.xlabel(labA.replace(" (mm)","")); plt.ylabel(labB.replace(" (mm)",""))
        plt.tight_layout()

        # localized points colored by cluster with estimated centers
        A2, B2, labA2, labB2, ia2, ib2 = choose_plane(P[:,0], P[:,1], P[:,2])
        plt.figure(figsize=(7,7))
        plt.scatter(A2, B2, c=labels_all, s=4, alpha=0.6, cmap="tab10")
        C2A, C2B = centers_refined[:, ia2], centers_refined[:, ib2]
        plt.scatter(C2A, C2B, marker="x", s=120, linewidths=2)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Estimated source positions from localized coincidences")
        plt.xlabel(labA2.replace(" (mm)","")); plt.ylabel(labB2.replace(" (mm)",""))
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

