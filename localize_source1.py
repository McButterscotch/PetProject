import sys, numpy as np, uproot, matplotlib.pyplot as plt

ROOT_PATH = sys.argv[1] 
PHOTOPEAK_WINDOW_KEV = (350.0, 650.0)
SOURCE_POS_MM = [(-50.0, 0.0, 0.0), (50.0, 0.0, 0.0)]  # from your script

def find_branch(t, candidates):
    """Return actual branch name matching any of the case-insensitive candidates."""
    keys = {k.lower(): k for k in t.keys()}
    for c in candidates:
        if c.lower() in keys:
            return keys[c.lower()]
    return None

def read_any_hits(root_path):
    f = uproot.open(root_path)
    # Prefer Singles*, else Hits*. Accept names like "Tree Singles;1"
    singles = [k for k in f.keys() if k.lower().startswith("tree singles") or k.lower().startswith("singles")]
    hits    = [k for k in f.keys() if k.lower().startswith("tree hits")    or k.lower().startswith("hits")]
    trees = singles + hits if singles else hits
    if not trees:
        raise RuntimeError("No Singles* or Hits* trees found.")

    all_xyzE = []
    used_tree = None

    for tn in trees:
        t = f[tn]
        # Try multiple naming patterns
        bx = find_branch(t, ["PostPosition_X","GlobalPosX","GlobalPosition_X","PosX","Position_X"])
        by = find_branch(t, ["PostPosition_Y","GlobalPosY","GlobalPosition_Y","PosY","Position_Y"])
        bz = find_branch(t, ["PostPosition_Z","GlobalPosZ","GlobalPosition_Z","PosZ","Position_Z"])
        bE = find_branch(t, ["TotalEnergyDeposit","Energy","energy","Edep","E"])

        if bx and by and bz and bE:
            arr = t.arrays([bx,by,bz,bE], library="np")
            x, y, z, E = arr[bx], arr[by], arr[bz], arr[bE]
            if len(x) > 0:
                all_xyzE.append((x,y,z,E))
                used_tree = tn  # note the last used tree (first good one)
                # If we already harvested from a “Hits” tree, we can stop early
                if "hits" in tn.lower():
                    break

    if not all_xyzE:
        raise RuntimeError("Found trees but no matching (x,y,z,E) branches. "
                           "Open the file and list branches to adjust aliases.")

    X = np.concatenate([a[0] for a in all_xyzE])
    Y = np.concatenate([a[1] for a in all_xyzE])
    Z = np.concatenate([a[2] for a in all_xyzE])
    E = np.concatenate([a[3] for a in all_xyzE])

    # Auto energy units: if median in (0.1..2) it's likely MeV → convert to keV
    Es = E[(E > 0) & np.isfinite(E)]
    med = np.median(Es) if Es.size > 0 else 0.0
    if 0.1 < med < 2.0:   # MeV-ish
        E = E * 1000.0

    print(f"Using tree: {used_tree}, samples: {len(X)}, median energy (keV): {np.median(E[(E>0)&np.isfinite(E)]):.1f}")
    return X, Y, Z, E

def main():
    X, Y, Z, E = read_any_hits(ROOT_PATH)

    # Photopeak cut
    emin, emax = PHOTOPEAK_WINDOW_KEV
    keep = (E >= emin) & (E <= emax) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[keep], Y[keep], Z[keep]

    # Decide transaxial plane automatically: pick two coords with largest variance
    vars_ = np.array([np.var(X), np.var(Y), np.var(Z)])
    dims = vars_.argsort()[-2:][::-1]  # indices of top-2 variances
    labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    coord = [X, Y, Z]

    a, b = dims[0], dims[1]
    A, B = coord[a], coord[b]

    # Optional thinning if huge
    nmax = 400_000
    if A.size > nmax:
        idx = np.random.choice(A.size, nmax, replace=False)
        A, B = A[idx], B[idx]

    plt.figure(figsize=(7,7))
    plt.scatter(A, B, s=3, alpha=0.5, linewidths=0)  # hits

    # Overlay sources projected into the same plane
    for sx, sy, sz in SOURCE_POS_MM:
        S = [sx, sy, sz]
        print(S)
        plt.scatter(S[a], S[b], marker="x", s=80)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(labels[a].replace(" (mm)",""))
    plt.ylabel(labels[b].replace(" (mm)",""))
    plt.title("Transaxial detection position (Hits)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()