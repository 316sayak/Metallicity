"""
KS19 photoionization-model gallery for all components of the final catalogue (Metal_properties_N_line.csv).

Output (OUT, on the SSD), PNG for a static web page (GitHub Pages):
  corner/<qso>_sys<S>_comp<C>_corner.png      corner plot of the KS19 posterior (KS19_posteriors_abund/abund_*.dat;
                                              sampled parameters n, [alpha/H], [C/H], [N/H], N_HI with independent
                                              flat priors on the abundances; [C/alpha], [N/alpha] derived)
  ionization_models/<qso>_sys<S>_comp<C>_model.png
        top: observed columns (detections with 1 sigma; upper limits as arrows) and the KS19 model (median, 16-84%
             over 300 posterior draws); bottom: residual log N_obs - log N_model
             (detections: error = obs and model errors in quadrature; limits: arrow from limit - model median)
  velocity_plots/<qso>_sys<S>_vplot.png        normalised COS spectrum used for the VPFIT fits
        (ind_abs/line_catalog/metal_fit/fields/<qso>/fit_spec/*norm*) around
        every H I Lyman line (up to 6) and every metal ion used in the KS19 inputs of the system (commented lines
        excluded): all major transitions of each ion (doublets complete) plus the transition named for a limit
        (e.g. NIII_763); display-only extra ions in EXTRA_PANELS (PKS0552 sys 3: C II).
        Transitions redward of COS use the NUV/STIS spectrum (fit_spec/civ_nuv/*normed.dat, STIS LSF lsf_stis_n1.dat).
        Profiles: comp_final_sys*_<qso>.dat (H I only from metalfit_read.dat for metal-free systems), plus detected
        input ions missing there (metalfit_read.dat within 150 km/s; PG1522 sys 26 C IV from the STIS VPFIT fit).
        Dashed coloured: individual components, drawn only in panels of their own ion; red: total model of all
        ions (blends included). Line data: VPFIT 12.4 atom.dat + missing lines from vpfit12/atom.dat (+ N III 763).
        Vertical dashed lines: KS19-modelled components (labels = comp numbers). Interloping absorption is not modelled.
  galaxy_images/<qso>_sys<S>_hst.png, _muse.png  HST cutout of the smallest-D/Rvir host(s) and MUSE white-light image
        with the host and its group members (see the "galaxy" stage)
  final_results/ (summary figures, added on request; manifest final_results/results.json -> top of index.html)
  index.html, README.md, catalogue.csv
Ionization-model grids: the same KS19 Cloudy grids and interpolation in z (0-1) as pie.py.
Usage: python make_gallery.py [corner] [model] [vplot] [galaxy] [index]   (no argument: all)
"""
import os, sys, warnings, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

KS = "/Users/dsayak/Work/PC/Metallicity_Project/catalog_refined/Modeling_KS19/"
DATA = "/Users/dsayak/Work/PC/Metallicity_Project/catalog_refined/"
CATD = "/Users/dsayak/Work/PC/Metallicity_Project/catalog/"
OUT = "/Volumes/PortableSSD/PC/MUSEQuBES/KS19_model_gallery/"
sys.path.insert(0, KS); sys.path.insert(0, KS + "diagnostics")
import vptools as V                                                                     # noqa: E402

# atomic data: vptools reads the VPFIT 12.4 atom.dat, which lacks e.g. N IV 765, C II 903; add every missing
# line of the fuller vpfit12 atom.dat (+ N III 763 by hand) for the plots only (vptools fits are unchanged)
_ATOM2 = "/Users/dsayak/Work/PC/vpfit_install/vpfit12/atom.dat"
for _l in open(_ATOM2):
    _m = re.match(r"^(\S+(?: [IVX]+)?)\s+([\d.]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)", _l)
    if _m and _m.group(1)[0].isupper() and not _l.startswith(("!", " ", "<", ">", "__", "??")):
        try:
            _ion, _t = _m.group(1).replace(" ", ""), tuple(float(x) for x in _m.groups()[1:])
        except ValueError:
            continue
        _L = V.ATOMS.setdefault(_ion, [])
        if all(abs(_t[0] - x[0]) > 0.05 for x in _L):
            _L.append(_t)
for _ion, _t in [("NIII", (763.334, 0.0820, 3.0e9))]:
    if all(abs(_t[0] - x[0]) > 0.05 for x in V.ATOMS.setdefault(_ion, [])):
        V.ATOMS[_ion].append(_t)


def named_line(nm):
    """input-file name -> (ion, rest wavelength or None). 'CII_903a' -> stronger of the 903 doublet, 'b' -> 2nd."""
    ion, _, t = nm.partition("_")
    if not t or ion not in V.ATOMS:
        return ion, None
    mm = re.match(r"(\d+)([a-z]?)", t)
    w0 = int(mm.group(1))                                      # names carry the truncated wavelength
    c = sorted([x for x in V.ATOMS[ion] if w0 <= x[0] < w0 + 1], key=lambda x: -x[1]) or \
        sorted([x for x in V.ATOMS[ion] if abs(x[0] - w0) < 1.5], key=lambda x: abs(x[0] - w0))
    if not c:
        return ion, None
    k = "ab".index(mm.group(2)) if mm.group(2) in "ab" and mm.group(2) else 0
    return ion, c[min(k, len(c) - 1)][0]


DO = set(sys.argv[1:]) or {"corner", "model", "vplot", "galaxy", "index"}
for d in ["corner", "ionization_models", "velocity_plots"]:
    os.makedirs(OUT + d, exist_ok=True)
cat = pd.read_csv(KS + "Metal_properties_N_line.csv", index_col=0).sort_values(["sl", "sys", "comp"]).reset_index(drop=True)
post = lambda r: np.loadtxt(KS + f"KS19_posteriors/corner2_sys{r.sys}_comp{r.comp}_{r.sl}.dat.dat")
FLAG = {0: "upper limit", 1: "detection", 2: "lower limit", 6: "unconstrained"}
ALPHA = ("O", "Si", "S", "Mg", "Ne")


def tag(r):
    return f"{r.sl}_sys{r.sys}_comp{r.comp}"


def inp_lines(r):
    L = []
    for l in open(DATA + f"sys{r.sys}_comp{r.comp}_{r.sl}.dat"):
        p = l.split()
        if len(p) >= 3 and not l.lstrip().startswith("#"):
            L.append((p[0], float(p[1]), float(p[2])))
    return L


# ---------------------------------------------------------------------------------------------- corner plots
if "corner" in DO:
    import corner
    for r in cat.itertuples():
        p = np.loadtxt(KS + f"KS19_posteriors_abund/abund_sys{r.sys}_comp{r.comp}_{r.sl}.dat")[:, :5]   # n, a/H, C/H, N/H, NHI
        fig = corner.corner(p, labels=[r"log $n_{\rm H}$", r"[$\alpha$/H]", r"[C/H]", r"[N/H]", r"log $N_{\rm HI}$"],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f", smooth=1, smooth1d=1,
                            title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 10})
        fig.suptitle(f"{r.sl}  sys {r.sys}  comp {r.comp}   z = {r.zHI:.5f}   log N(HI) = {r.NHI:.2f}\n"
                     f"[α/H]: {FLAG[r.Z_flag]},  [C/H]: {FLAG[r.CH_flag]},  [N/H]: {FLAG[r.NH_flag]}   "
                     f"(derived [C/α] = {r.CAL:+.2f}: {FLAG[r.CAL_flag]},  [N/α] = {r.NAL:+.2f}: {FLAG[r.NAL_flag]})", fontsize=10, y=1.03)
        fig.savefig(OUT + f"corner/{tag(r)}_corner.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    print("corner plots:", len(cat))

# ---------------------------------------------------------------------------------------------- ionization models
if "model" in DO:
    from scipy.interpolate import interp1d
    src = open(KS + "pie.py").read()
    exec(src[:src.index("# sys_id_this = 27")])                                     # pie_model, read_grid, elem_dictt ...
    zs, zf, el_ = ["00", "02", "04", "06", "08", "10"], ["0", "2", "4", "6", "8", "10"], [149938, 149935, 150169, 151217, 152767, 154407]
    allz = []
    for a, b, e in zip(zs, zf, el_):
        elem_all, ion_states = read_grid(f"/Users/dsayak/Work/CloudyGrid_KS19/z{a}/", f"grid_piez{b}.ionfrac", e)
        allz.append(ion_states)
    fz = interp1d(np.round(np.arange(0, 1.1, 0.2), 1), allz, axis=0)
    PRED = [("CIV*", "Carbon", 4), ("OVI*", "Oxygen", 6), ("SiIV*", "Silicon", 4), ("OIII*", "Oxygen", 3)]
    rng = np.random.default_rng(5)
    for r in cat.itertuples():
        if not os.path.exists(DATA + f"sys{r.sys}_comp{r.comp}_{r.sl}.dat"):    # input file absent (Q0107 sys52 comp2)
            print("model: no input file, kept old plot:", r.sl, r.sys, r.comp); continue
        pm = pie_model(DATA + f"sys{r.sys}_comp{r.comp}_{r.sl}.dat", fz(min(max(r.zHI, 0), 1)), elem_all)
        ions, el, ist, Nv, Ne = list(pm.ions), list(pm.el_names), list(pm.Istates), np.atleast_1d(pm.Nvals), np.atleast_1d(pm.Nerrs)
        pred = []                                                                   # no predicted ions shown
        names = ions + [p[0] for p in pred]; allel = el + [p[1] for p in pred]; allis = ist + [p[2] for p in pred]
        p = post(r); sub = p[rng.choice(len(p), 300, replace=False)]
        M = np.array([[np.log10(pm.Ni(e, i, s[0], s[1], s[4], s[2], s[3])[0]) for e, i in zip(allel, allis)] for s in sub])
        lo, md, hi = np.percentile(M, [16, 50, 84], axis=0); x = np.arange(len(names))
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(max(6, .62 * len(names) + 1.5), 5.2), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2.3, 1], hspace=.05))
        ax.fill_between(x, lo, hi, color="C3", alpha=.25, lw=0); ax.plot(x, md, "C3-", lw=1.3, label="KS19 model (16–84%)")
        n = len(ions); low = np.isclose(Ne[:n], 1.0); det = (Ne[:n] > 0) & ~low; upl = Ne[:n] < 0
        ax.errorbar(x[:n][det], Nv[:n][det], yerr=Ne[:n][det], fmt="o", color="k", ms=6, zorder=4, label="observed")
        ax.errorbar(x[:n][upl], Nv[:n][upl], yerr=.25, uplims=True, fmt="v", color="k", mfc="w", ms=6, zorder=4, label="upper limit")
        if low.any():
            ax.errorbar(x[:n][low], Nv[:n][low], yerr=.25, lolims=True, fmt="^", color="k", mfc="w", ms=6, zorder=4, label="lower limit")
        ax.set_ylabel(r"log $N$ [cm$^{-2}$]"); ax.legend(fontsize=7, loc="upper right", ncol=3, frameon=False)
        ax.set_title(f"{r.sl}  sys {r.sys}  comp {r.comp}   z = {r.zHI:.5f}:  log n = {r.n:.2f},  [α/H] = {r.Z:.2f} "
                     f"({FLAG[r.Z_flag]}),  [C/H] = {r.CH:.2f} ({FLAG[r.CH_flag]}),  [N/H] = {r.NH:.2f} ({FLAG[r.NH_flag]})", fontsize=8)
        sig = np.hypot(np.where(det, Ne[:n], 0), (hi[:n] - lo[:n]) / 2)
        res = Nv[:n] - md[:n]; R = 1.0                                              # residual panel cropped to +-1 dex
        rc = np.clip(res, -0.93 * R, 0.93 * R); out = np.abs(res) > 0.93 * R
        axr.axhline(0, color="C3", lw=1); axr.axhspan(-.2, .2, color="0.9", zorder=0)
        d_in = det & ~out
        axr.errorbar(x[:n][d_in], res[d_in], yerr=sig[d_in], fmt="o", color="k", ms=5)
        axr.errorbar(x[:n][upl & ~out], res[upl & ~out], yerr=.2, uplims=True, fmt="v", color="k", mfc="w", ms=5)
        axr.errorbar(x[:n][low & ~out], res[low & ~out], yerr=.2, lolims=True, fmt="^", color="k", mfc="w", ms=5)
        for i_ in np.nonzero(out)[0]:                                               # off-scale: hollow triangle at the edge
            axr.plot(x[i_], rc[i_], "^" if res[i_] > 0 else "v", ms=8, mfc="none", mec="0.35", mew=1.2)
        for i_ in np.nonzero((upl & (res < 0)) | (low & (res > 0)))[0]:            # model violates the limit
            axr.plot(x[i_], rc[i_], "v" if upl[i_] else "^", ms=9, mfc="none", mec="C3", mew=1.5)
        axr.set_ylabel(r"obs $-$ model" + "\n[dex]", fontsize=9); axr.set_ylim(-R, R)
        axr.set_xticks(x, names, rotation=0, fontsize=8); axr.set_xlim(-.5, len(names) - .5)
        fig.savefig(OUT + f"ionization_models/{tag(r)}_model.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("model plots:", len(cat))

# ---------------------------------------------------------------------------------------------- velocity plots
if "vplot" in DO:
    def comp_list(q, s, zs_):
        f = CATD + f"comp_final_sys{s}_{q}.dat"
        out = []
        if os.path.exists(f):
            for l in open(f):
                p = l.replace("H I ", "HI ").split()
                if len(p) >= 7 and p[0] in V.ATOMS:
                    try:
                        out.append((p[0], float(p[1]), float(p[3]), float(p[5])))
                    except ValueError:
                        pass
            return out, "comp_final"
        zc = np.mean(zs_)
        mf = [c for c in V.metalfit_comps(q) if abs(c[1] - zc) / (1 + zc) * V.C < 400]
        return mf, "metalfit_read (no comp_final)"

    # profiles of ions detected in the KS19 inputs but absent from comp_final: metalfit_read within 150 km/s, or
    # dedicated fits (PG1522 sys 26 C IV: STIS VPFIT, diagnostics/run_pg1522_z0729_metals/CIV_stis/*.f26)
    EXTRA_PANELS = {("PKS0552-640", 3): ["CII"]}
    MANUAL = {("PG1522+101", 26): [("CIV", 0.7287382993, 13.74593, 12.947146)]}

    def add_missing(q, s, g, comps):
        have = {c[0] for c in comps}; add = list(MANUAL.get((q, s), []))
        mf = None
        for r in g.itertuples():
            for nm, N, e in inp_lines(r):
                ion, _ = named_line(nm)
                if ion == "HI" or e <= 0 or ion in have or any(a[0] == ion for a in add):
                    continue
                mf = V.metalfit_comps(q) if mf is None else mf
                add += [c for c in mf if c[0] == ion and abs(c[1] - r.zHI) / (1 + r.zHI) * V.C < 150]
        return comps + [a for a in add if a[0] not in have]

    CCOL = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    for (q, s), g in cat.groupby(["sl", "sys"]):
        fs = [x for x in os.listdir(V.FIELDS + f"{q}/fit_spec") if "norm" in x.lower()][0]      # spectrum used for the fits
        wv, fn, en = np.loadtxt(V.FIELDS + f"{q}/fit_spec/{fs}", unpack=True, usecols=(0, 1, 2))
        ok = np.isfinite(fn) & np.isfinite(en); wv, fn, en = wv[ok], fn[ok], en[ok]
        # NUV (STIS) spectrum used for transitions redward of COS (civ_nuv/*normed.dat; STIS pixel LSF lsf_stis_n1)
        nuvd = V.FIELDS + f"{q}/fit_spec/civ_nuv/"
        nf = [x for x in os.listdir(nuvd) if x.endswith("normed.dat")] if os.path.isdir(nuvd) else []
        if nf:
            wx, fx, ex = np.loadtxt(nuvd + nf[0], unpack=True, usecols=(0, 1, 2))
            ok = np.isfinite(fx) & np.isfinite(ex); wx, fx, ex = wx[ok], fx[ok], ex[ok]
            kst = np.loadtxt(nuvd + "lsf_stis_n1.dat")[:, 1]; kst = kst / kst.sum()
        else:
            wx = np.array([])
        z0 = g.zHI.values[np.argmax(g.NHI.values)]
        comps, csrc = comp_list(q, s, g.zHI.values)
        comps = add_missing(q, s, g, comps)
        # panels: H I Lyman series + every metal ion used in the KS19 modelling of the system's components
        # (commented-out input lines excluded): the transition named for a limit (e.g. NIII_763) and, for fitted
        # ions (no suffix), all major transitions (f*lam >= 0.15 of the strongest, max 4)
        want = {}

        def major(ion):                                   # f*lam >= 0.15 of the strongest, max 5 (doublets kept)
            tl = V.ATOMS[ion]; top = max(f * l for l, f, _ in tl)
            nmax = {"OI": 2}.get(ion, 5)                 # O I: only the two strongest (1302, 989)
            return [t[0] for t in sorted([t for t in tl if t[1] * t[0] >= 0.15 * top], key=lambda t: -t[1] * t[0])[:nmax]]

        for r in g.itertuples():
            for nm, N, e in inp_lines(r):
                ion, lam = named_line(nm)
                if ion == "HI" or ion not in V.ATOMS:
                    continue
                want.setdefault(ion, set()).update(round(x, 3) for x in major(ion) + ([lam] if lam else []))
        for ion in EXTRA_PANELS.get((q, s), []):          # display-only ions (not in the KS19 inputs)
            want.setdefault(ion, set()).update(round(x, 3) for x in major(ion))

        def covered(lam, w=None):
            if w is None:
                return covered(lam, wv) or covered(lam, wx)
            return np.sum(np.abs(w / (lam * (1 + z0)) - 1) * V.C < 300) > 50      # real data in the window (gaps)

        def nuv_model(w, cc):
            return np.convolve(np.pad(V.model(w, cc, conv=False), len(kst) // 2, mode="edge"), kst[::-1], mode="valid")[:len(w)]

        groups = []
        for ion, ls in want.items():
            sel = sorted(l for l in ls if covered(l))
            if sel:
                groups.append((min(sel), [(ion, l) for l in sel]))
        hi_tr = [("HI", l[0]) for l in V.lyman(6) if covered(l[0])]
        tr = hi_tr + [t for _, grp in sorted(groups, key=lambda g_: g_[0]) for t in grp]
        ncol = 1 if len(tr) <= 9 else 2; nrow = int(np.ceil(len(tr) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 1.15 * nrow + .9), sharex=True, squeeze=False)
        axs_flat = axs.T.ravel()
        vr = 400
        for ax, (ion, lam) in zip(axs_flat, tr):
            cos = covered(lam, wv)
            W, Fn, En, mod = (wv, fn, en, V.model) if cos else (wx, fx, ex, nuv_model)
            lc = lam * (1 + z0); m = (W > lc * (1 - vr / V.C) - 1) & (W < lc * (1 + vr / V.C) + 1)
            v = V.C * (W[m] / lc - 1)
            ax.step(v, Fn[m], where="mid", color="k", lw=.55); ax.step(v, En[m], where="mid", color="0.65", lw=.4)
            try:                                                                    # individual components, then total
                for ic, cc in enumerate(comps):
                    if cc[0] != ion:                                                # other ions: in the total only
                        continue
                    mc = mod(W[m], [cc])
                    if mc.min() < 0.98:
                        ax.plot(v, mc, color=CCOL[ic % len(CCOL)], lw=.7, ls="--", alpha=.9)
                ax.plot(v, mod(W[m], comps), "C3", lw=1.1)
            except Exception as ex:
                print(f"  model failed {q} sys{s} {ion} {lam:.0f}: {ex!r}")
            for r in g.itertuples():
                vv = V.C * (r.zHI - z0) / (1 + z0)
                ax.axvline(vv, color="C0", ls="--", lw=.6)
            ax.axhline(1, color="0.75", lw=.4); ax.axhline(0, color="0.75", lw=.4)
            ax.set_ylim(-.1, 1.45); ax.set_xlim(-vr, vr)
            lab = f"{ion} {lam:.0f}" if sum(1 for i_, l_ in tr if i_ == ion and round(l_) == round(lam)) == 1 else f"{ion} {lam:.1f}"
            ax.text(.01, .08, lab + ("" if cos else "  (STIS)"), transform=ax.transAxes, fontsize=8, bbox=dict(fc="w", ec="none", alpha=.7, pad=.5))
        for ax in axs_flat[len(tr):]:
            ax.axis("off")
        top = axs[0, 0]
        for r in g.itertuples():
            top.text(V.C * (r.zHI - z0) / (1 + z0), 1.47, f"c{r.comp}", color="C0", fontsize=7, ha="center", va="bottom")
        for ax in axs[-1]:
            ax.set_xlabel(f"v [km/s]  (z = {z0:.5f})")
        fig.suptitle(f"{q}  sys {s}: KS19 components {', '.join('c' + str(c) for c in g.comp)}   "
                     f"(dashed: individual components of the panel ion; red: total Voigt model of all ions; COS LSF, STIS LSF for STIS panels)", fontsize=9, y=1.0)
        fig.tight_layout(h_pad=0.05, rect=(0, 0, 1, .985))
        fig.savefig(OUT + f"velocity_plots/{q}_sys{s}_vplot.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    print("velocity plots:", cat.groupby(["sl", "sys"]).ngroups)

# ---------------------------------------------------------------------------------------------- host galaxy images
# galaxy_images/<qso>_sys<S>_hst.png : HST cutout (north up, east left) of the smallest-D/Rvir host of the system
#     (two panels if the components have different hosts); D, log M*, SFR, D/Rvir; arrow towards the QSO; 10 kpc bar
# galaxy_images/<qso>_sys<S>_muse.png: full MUSE white-light image; host circled (yellow), other members of the
#     host's group (group_galaxies2.dat) in cyan squares, QSO marked
if "galaxy" in DO:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    from reproject import reproject_interp
    from matplotlib.patches import Circle, Rectangle
    GS = "/Users/dsayak/Work/PC/MUSEQuBES/Emission_Nebula/environment/group_stud/"
    MH = GS + "metal_coherence/morph_hst/"
    WL = "/Users/dsayak/Work/PC/MUSEQuBES/Resource/Survey_final/white_im_MUSEQuBES/"
    os.makedirs(OUT + "galaxy_images", exist_ok=True)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    hosts = pd.read_csv(GS + "metal_coherence/minDRvir/clouds_with_host_minDRvir.csv", index_col=0)
    fg = pd.read_csv("/Users/dsayak/Work/PC/MUSEQuBES/Resource/musequbes_fg.csv").set_index("ID")
    mor = pd.read_csv(MH + "morph_galaxies.csv").set_index("gal_id")
    grp = pd.read_csv(GS + "group_galaxies2.dat", sep=r"\s+", comment="#", names=["id", "qso", "zq", "zg", "m", "b", "rvir", "grp"])
    qtab = pd.read_csv("/Users/dsayak/Work/PC/MUSEQuBES/Resource/muse_q_v2.txt", sep=r"\s+", comment="#", usecols=[0, 1, 2],
                       names=["qso", "qra", "qdec"]).set_index("qso")
    GALINFO = {}

    def stretch(im, lo=1, hi=99.7):
        a, b = np.nanpercentile(im, [lo, hi]); x = np.clip((im - a) / (b - a), 0, 1)
        return np.arcsinh(10 * x) / np.arcsinh(10)

    def hst_panel(ax, h, hr, q):
        f = fits.open(MH + f"cutouts/{h}_HST.fits"); sci = f["SCI"].data; w = WCS(f["SCI"].header)
        c0 = SkyCoord(f["SCI"].header["RA"], f["SCI"].header["DEC"], unit="deg")
        re_as = mor.loc[h, "Re_arcsec_med"] if h in mor.index else 0.5
        half = float(np.clip(4 * re_as, 3.0, 12.0)); ps = half / 100                       # 200x200 output, north up
        wn = WCS(naxis=2); wn.wcs.ctype = ["RA---TAN", "DEC--TAN"]; wn.wcs.crval = [c0.ra.deg, c0.dec.deg]
        wn.wcs.crpix = [100.5, 100.5]; wn.wcs.cdelt = [-ps / 3600, ps / 3600]
        im, _ = reproject_interp((sci, w), wn, shape_out=(200, 200))
        ax.imshow(stretch(im, 5, 99.8), origin="lower", cmap="gray_r", extent=[half, -half, -half, half])
        ax.set_xlim(half, -half); ax.set_ylim(-half, half)
        qc = SkyCoord(qtab.loc[q, "qra"], qtab.loc[q, "qdec"], unit="deg")
        dx, dy = [v.to(u.arcsec).value for v in c0.spherical_offsets_to(qc)]                # east, north
        L = 0.75 * half / np.hypot(dx, dy)
        ax.annotate("", xy=(dx * L, dy * L), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="crimson", lw=1.4))
        ax.text(dx * L * 1.08, dy * L * 1.08, "QSO", color="crimson", fontsize=8, ha="center", va="center")
        kpc_as = cosmo.kpc_proper_per_arcmin(hr.zgal).value / 60
        bar = 10 / kpc_as
        ax.plot([-half * 0.9 + bar, -half * 0.9], [-half * 0.85] * 2, color="k", lw=2)
        ax.text(-half * 0.9 + bar / 2, -half * 0.78, "10 kpc", ha="center", fontsize=8)
        sfr = f"< {10 ** hr.logSFR:.2f}" if hr.SFRul else f"{10 ** hr.logSFR:.2f}"
        ax.set_title(f"{h}  (z = {hr.zgal:.4f})", fontsize=9)
        ax.text(0.03, 0.97, f"D = {hr.b_host:.0f} kpc,  D/R$_{{\\rm vir}}$ = {hr.DRvir:.2f}\nlog M$_\\star$ = {hr.logM:.2f}\n"
                f"SFR {sfr} M$_\\odot$ yr$^{{-1}}$\n{'group member' if hr.env == 'group' else 'isolated'}",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8))
        ax.set_xlabel(r'$\Delta\alpha$ ["]', fontsize=8); ax.set_ylabel(r'$\Delta\delta$ ["]', fontsize=8); ax.tick_params(labelsize=7)

    for (q, s), g in hosts.groupby(["sl", "sys"]):
        hl = list(dict.fromkeys(g.sort_values("comp").host))
        hrows = {h: g[g.host == h].iloc[0] for h in hl}
        fig, axs = plt.subplots(1, len(hl), figsize=(4.2 * len(hl), 4.3), squeeze=False)
        for ax, h in zip(axs[0], hl):
            hst_panel(ax, h, hrows[h], q)
            if len(hl) > 1:
                ax.text(0.97, 0.97, "comp " + ",".join(map(str, sorted(g[g.host == h].comp))), transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8))
        fig.tight_layout(); fig.savefig(OUT + f"galaxy_images/{q}_sys{s}_hst.png", dpi=90); plt.close(fig)
        # MUSE white light
        # cube-collapsed images first: the Resource images of HE0153, HE0238, PB6291, RXSJ02282 do not match the galaxy
        # catalogue astrometry (morph_hst/muse_white/README.md); 3C57 has none
        fn = MH + f"muse_white/{q}_white.fits" if os.path.exists(MH + f"muse_white/{q}_white.fits") else WL + f"{q}_white.fits"
        f = fits.open(fn); ext = "DATA" if "DATA" in f else 0
        im = f[ext].data; w = WCS(f[ext].header).celestial
        fig = plt.figure(figsize=(4.6, 4.6)); ax = fig.add_subplot(projection=w)
        ax.imshow(stretch(im, 2, 99.5), origin="lower", cmap="gray_r")
        pix = lambda ra, de: w.world_to_pixel(SkyCoord(ra, de, unit="deg"))
        qx, qy = pix(qtab.loc[q, "qra"], qtab.loc[q, "qdec"])
        ax.plot(qx, qy, marker="*", ms=13, mfc="none", mec="crimson", mew=1.3)
        rad = 1.2 / 0.2                                                                     # 1.2" in MUSE pixels
        mem = set()
        for h in hl:
            gid = grp[grp.id == h].grp
            if len(gid):
                mem |= set(grp[grp.grp == gid.iloc[0]].id)
        mem -= set(hl)
        for m in sorted(mem):
            if m in fg.index:
                x, y = pix(fg.loc[m, "RA"], fg.loc[m, "DEC"])
                ax.add_patch(Rectangle((x - rad, y - rad), 2 * rad, 2 * rad, fill=False, ec="deepskyblue", lw=1.3))
                ax.text(x, y + 1.6 * rad, m.split("_")[-1], color="deepskyblue", fontsize=7, ha="center")
        for h in hl:
            x, y = pix(fg.loc[h, "RA"], fg.loc[h, "DEC"])
            ax.add_patch(Circle((x, y), 1.3 * rad, fill=False, ec="gold", lw=2))
            ax.text(x, y - 2.4 * rad, h.split("_")[-1], color="darkgoldenrod", fontsize=8, ha="center", fontweight="bold")
        ax.set_title(f"{q} MUSE white light: host (yellow)" + (f", group members (cyan, {len(mem)})" if mem else ", isolated"), fontsize=8)
        ax.coords[0].set_axislabel("RA", fontsize=8); ax.coords[1].set_axislabel("Dec", fontsize=8)
        ax.coords[0].set_ticklabel(size=7); ax.coords[1].set_ticklabel(size=7)
        fig.savefig(OUT + f"galaxy_images/{q}_sys{s}_muse.png", dpi=90, bbox_inches="tight"); plt.close(fig)
        print(q, s, hl, "group members:", len(mem))


# ---------------------------------------------------------------------------------------------- index
if "index" in DO:
    t = cat[["sl", "sys", "comp", "zHI", "NHI", "dNHI", "n", "n_low", "n_high", "Z", "Z_low", "Z_high", "Z_flag",
             "CH", "CH_low", "CH_high", "CH_flag", "NH", "NH_low", "NH_high", "NH_flag",
             "CAL", "CAL_low", "CAL_high", "CAL_flag", "NAL", "NAL_low", "NAL_high", "NAL_flag"]].copy()
    t.to_csv(OUT + "catalogue.csv", index=False, float_format="%.4f")
    H = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>KS19 absorber models</title><style>body{font-family:system-ui,sans-serif;margin:16px;max-width:1300px;color:#1a1a1a}"
         "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;font-size:13px}"
         "img{width:300px;border:1px solid #eee}.row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}"
         ".gal{display:flex;flex-direction:column;gap:6px}h2{margin-top:36px}.lim{color:#a33}</style></head><body>",
         "<h1>MUSEQuBES CGM absorbers: KS19 photoionization models</h1>",
         "<p>One velocity plot per absorber system, with (right) the HST image of the host galaxy (smallest D/R<sub>vir</sub> "
         "within ±500 km/s; north up, east left; arrow: direction to the QSO; D = impact parameter) and the MUSE white-light "
         "image (host circled in yellow, other members of its group in cyan squares, QSO star); for each modelled component its corner plot and ionization-model fit "
         "(observed vs model columns, residuals below). Click an image for full size. Metallicity flags: detection, "
         "<span class='lim'>upper limit</span>. Data: <a href='catalogue.csv'>catalogue.csv</a>.</p>"]
    # ---- final results (figures added on request; manifest final_results/results.json)
    import json
    man = OUT + "final_results/results.json"
    if os.path.exists(man):
        res = json.load(open(man, encoding="utf-8"))
        H.append("<h2 id='final_results'>Final results</h2>")
        for e in res:
            links = " · ".join(f"<a href='final_results/{e[k]}'>{lbl}</a>" for k, lbl in [("pdf", "PDF"), ("data", "data")] if e.get(k))
            H.append(f"<h3>{e['title']}</h3><p><a href='final_results/{e['file']}'><img src='final_results/{e['file']}' "
                     f"style='width:100%;max-width:1250px'></a></p><p style='font-size:13px;max-width:1250px'>{e['caption']}"
                     f"{' &nbsp;[' + links + ']' if links else ''}</p>")
        H.append("<h2>Absorber systems</h2>")
    _h = pd.read_csv("/Users/dsayak/Work/PC/MUSEQuBES/Emission_Nebula/environment/group_stud/metal_coherence/minDRvir/"
                     "clouds_with_host_minDRvir.csv", index_col=0)
    MULTI = {k for k, v in _h.groupby(["sl", "sys"]).host.nunique().items() if v > 1}
    for (q, s), g in cat.groupby(["sl", "sys"], sort=True):
        H.append(f"<h2 id='{q}_sys{s}'>{q} &nbsp;sys {s}</h2>")
        vp = f"velocity_plots/{q}_sys{s}_vplot.png"
        gi = [f"galaxy_images/{q}_sys{s}_{k}.png" for k in ("hst", "muse")]
        gi = "".join(f"<a href='{x}'><img src='{x}' style='width:{w}px;border:none'></a>" for x, w in zip(gi, (430 if (q, s) in MULTI else 330, 330))
                     if os.path.exists(OUT + x))
        H.append(f"<div class='row'><a href='{vp}'><img src='{vp}' style='width:620px'></a><div class='gal'>{gi}</div></div>"
                 "<table><tr><th>comp</th><th>z, log N(HI)</th>"
                 "<th>log n, [α/H], [C/H], [N/H] (sampled); [C/α], [N/α] (derived)</th><th>corner</th><th>ionization model</th></tr>")
        for r in g.itertuples():
            zt = f"{r.Z:+.2f}" + (f" <sup>+{r.Z_high:.2f}</sup><sub>−{r.Z_low:.2f}</sub>")
            cls = " class='lim'" if r.Z_flag == 0 else ""
            H.append(f"<tr><td>{r.comp}</td><td>{r.zHI:.5f}<br>{r.NHI:.2f} ± {r.dNHI:.2f}</td>"
                     f"<td>log n = {r.n:+.2f}<br><span{cls}>[α/H] = {zt} ({FLAG[r.Z_flag]})</span><br>"
                     f"[C/H] = {r.CH:+.2f} ({FLAG[r.CH_flag]})<br>[N/H] = {r.NH:+.2f} ({FLAG[r.NH_flag]})<br>"
                     f"<i>[C/α] = {r.CAL:+.2f} ({FLAG[r.CAL_flag]}), [N/α] = {r.NAL:+.2f} ({FLAG[r.NAL_flag]})</i></td>"
                     f"<td><a href='corner/{tag(r)}_corner.png'><img src='corner/{tag(r)}_corner.png'></a></td>"
                     f"<td><a href='ionization_models/{tag(r)}_model.png'><img src='ionization_models/{tag(r)}_model.png'></a></td></tr>")
        H.append("</table>")
    H.append("</body></html>")
    open(OUT + "index.html", "w").write("\n".join(H))
    open(OUT + "README.md", "w").write(
        "# KS19 photoionization models of MUSEQuBES CGM absorbers\n\n"
        f"{len(cat)} components in {cat.groupby(['sl', 'sys']).ngroups} absorber systems "
        f"({cat.sl.nunique()} sightlines). Open `index.html` (static page; GitHub Pages ready).\n\n"
        "* `velocity_plots/` - normalised HST/COS spectra (STIS for transitions redward of COS) around the H I Lyman lines "
        "and every metal transition used in the models (detections and limits), with the Voigt-profile model: individual "
        "components (coloured dashed) only in the panels of their own ion, total model of all ions (red). Vertical dashed "
        "lines mark the modelled components.\n"
        "* `corner/` - posterior of the KS19 (Khaire & Srianand 2019 UVB) Cloudy photoionization model for each component: "
        "log n_H, [alpha/H], [C/H], [N/H] (independent flat priors, -3 to +1) and log N(HI); [C/alpha] and [N/alpha] are derived.\n"
        "* `ionization_models/` - observed column densities (points; open triangles = 3 sigma upper limits) vs the model "
        "(red, 16-84% over the posterior); open upward triangles = lower limits (saturated); bottom panel: residual log N_obs - log N_model, cropped to +-1 dex (hollow triangles at the edge = off scale; red = model violates a limit).\n"
        "* `galaxy_images/` - per system: HST cutout of the host (smallest D/Rvir within +-500 km/s; north up; arrow to the "
        "QSO; D, log M*, SFR, D/Rvir) and the MUSE white-light image with the host (yellow) and the other members of its "
        "group (cyan; group_galaxies2.dat). 3C57, HE0153, HE0238 (_rev cube), PB6291, RXSJ02282 white light collapsed from the MUSE cubes (the others do not match the catalogue astrometry).\n"
        "* `final_results/` - summary figures (listed in `final_results/results.json`, shown at the top of `index.html`).\n"
        "* `catalogue.csv` - posterior medians and 16-84% ranges, flags (0 upper limit, 1 detection, 2 lower limit, "
        "6 unconstrained).\n\nGenerated by `make_gallery.py`.\n")
    os.system(f"cp '{os.path.abspath(__file__)}' '{OUT}make_gallery.py'")
    print("index written")
