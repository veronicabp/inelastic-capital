from utils import *


def construct_event_study_panel(
    df: pd.DataFrame,
    group: str = "state_n",
    time: str = "year",
    treat_col: str = "inter_bra",
) -> pd.DataFrame:
    """
    Identify ALL change events in inter_bra for each state.
    A valid event at time t0 is when inter_bra changes from v0 (at t0-1)
    to v1 (at t0). We only keep windows where:
      - in [t0 - lags, ..., t0 - 1] inter_bra == v0
      - in [t0, ..., t0 + leads] inter_bra == v1
    If a later change truncates the post window, we drop the truncated part.

    Returns a *stacked* event-time panel with columns:
      - group, time, event_id, event_time (τ), treat_pre, treat_post, ...
    """
    df = (
        df[[group, time, treat_col, f"L{treat_col}"]]
        .drop_duplicates()
        .sort_values([group, time])
    )

    # A change happens where current != lag and lag is not NA
    df["is_change"] = (
        (df[treat_col] != df[f"L{treat_col}"]) & df[f"L{treat_col}"].notna()
    ).astype(int)

    events = []
    event_counter = 0

    # Iterate groups
    for g, gdf in df.groupby(group, sort=False):
        gdf = gdf.sort_values(time)

        # indices where a change occurs
        change_idxs = np.where(gdf["is_change"].values)[0]
        for idx in change_idxs:
            t0 = gdf[time].values[idx]  # event time (first period with new value)
            v1 = gdf[treat_col].values[idx]  # new value after change
            v0 = gdf[f"L{treat_col}"].values[idx]  # value before change

            pre_years = gdf[gdf[treat_col] == v0][time].values
            post_years = gdf[gdf[treat_col] == v1][time].values

            # If we pass checks, build stacked rows for this event window
            # τ runs from -lags,...,-1,0,...,+leads
            rows = gdf[gdf[time].isin(pre_years) | gdf[time].isin(post_years)][
                [group, time]
            ].copy()
            rows["event_time"] = rows["year"] - t0
            rows["event_id"] = event_counter
            rows["regidx_pre"] = v0
            rows["regidx_post"] = v1
            rows["event_year"] = t0
            rows[group] = g  # keep explicit
            events.append(rows.reset_index())

            event_counter += 1

    stacked = pd.concat(events, ignore_index=True).reset_index(drop=True)
    return stacked


def build_event_time_dummies(
    stacked: pd.DataFrame, leads: int, lags: int, ref: int = -1, prefix: str = "tau"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create τ dummies for τ in [-lags, ..., -1, 0, ..., +leads] excluding ref τ.
    Returns modified DataFrame and list of kept dummy column names.
    """
    tau_vals = list(range(-lags, leads + 1))
    tau_vals = [t for t in tau_vals if t != ref]

    tau_cols = []
    for t in tau_vals:
        if t >= 0:
            tval = str(t)
        else:
            tval = f"n{-t}"
        stacked[f"{prefix}_{tval}"] = (stacked["event_time"] == t).astype(int)
        tau_cols.append(f"{prefix}_{tval}")

    return stacked, tau_cols


def run_event_study(
    df: pd.DataFrame,
    outcome: str,
    leads: int = 4,
    lags: int = 4,
    ref: int = -1,
    controls: Optional[List[str]] = [],
    fes: Optional[List[str]] = ["county", "year"],
    cluster_by: Optional[str] = "state_n",
    title: Optional[str] = None,
) -> Dict[str, object]:
    """
    TWFE event study:
      y_it = Σ_{τ≠ref} β_τ 1[EventTime=τ] + γ_i + δ_t + X'θ + ε_it
    Cluster SEs by 'cluster_by' (default: state).
    """
    df = df[(df["event_time"] >= -lags) & (df["event_time"] <= leads)].copy()
    df, tau_cols = build_event_time_dummies(df, leads, lags, ref=ref)

    print(sorted(df.event_time.unique()))

    results_dict = {}
    for suffix in ["_b", "_pl"]:
        rhs_parts = tau_cols + controls
        formula = (
            f"{outcome}{suffix} ~ " + " + ".join(rhs_parts) + " | " + " + ".join(fes)
        )
        # print(formula)
        results = feols(formula, data=df, vcov="hetero")

        # Extract coefficients and CIs for τ dummies
        kept = [c for c in tau_cols if c in results.coef().index]
        betas = list(results.coef().loc[kept].to_numpy())
        se = list(results.se().loc[kept].to_numpy())

        # Add omitted period as 0
        betas = np.array(betas + [0])
        se = np.array(se + [0])

        results_dict[suffix] = {
            "betas": betas,
            "se": se,
        }

    # Parse τ from column names like 'tau_-4', 'tau_+3'
    def parse_tau(name: str) -> int:
        val = name.split("_")[-1]
        if val.startswith("n"):
            val = -int(val[1:])
        return int(val)

    taus = np.array([parse_tau(n) for n in kept] + [ref])
    order = np.argsort(taus)

    plt.figure()
    plt.axhline(0.0, linestyle="--")
    plt.axvline(0.0, linestyle="--")

    for suffix in ["_b", "_pl"]:
        plt.errorbar(
            taus[order],
            results_dict[suffix]["betas"][order],
            yerr=1.96 * results_dict[suffix]["se"][order],
            fmt="o-",
            capsize=3,
            label={"_b": "Bank", "_pl": "Placebo"}[suffix],
        )
    plt.xlabel("Event time (years)")
    plt.ylabel("β(τ)")
    plt.title(title or f"Event-study: {outcome} effect around deregulation change")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def fabra_imbs_event_study(data_folder):
    folder = os.path.join(
        data_folder, "raw", "original", "fabra_imbs_2015", "20121416_1data", "data"
    )
    hmda = pd.read_stata(os.path.join(folder, "hmda.dta"))
    hp_dereg = pd.read_stata(os.path.join(folder, "hp_dereg_controls.dta"))
    call = pd.read_stata(os.path.join(folder, "call.dta"))

    df = hmda.merge(
        hp_dereg[
            [c for c in hp_dereg.columns if c not in hmda.columns] + ["county", "year"]
        ],
        on=["county", "year"],
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        call[[c for c in call.columns if c not in df.columns] + ["county", "year"]],
        on=["county", "year"],
        how="left",
        validate="one_to_one",
    )

    # For various outcomes, construct level from changes
    df = df.sort_values(["state_n", "county", "year"])

    cols = ["inc", "pop", "hpi", "her_v"] + [
        f"{c}{s}"
        for c in ["nloans", "vloans", "nden", "lir", "nsold"]
        for s in ["_b", "_pl"]
    ]

    for outcome in cols:
        df[f"log_{outcome}"] = df[f"Dl_{outcome}"].cumsum()
        # df[f"log_{outcome}"] = df[f"log_{outcome}"] - df.groupby(["state_n", "county"])[
        #    f"log_{outcome}"
        # ].transform("first")

        df = get_lag(
            df,
            group_cols=["state_n", "county"],
            shift_col=f"log_{outcome}",
            shift_amt=1,
        )

    event_panel = construct_event_study_panel(
        df, group="state_n", time="year", treat_col="inter_bra"
    )
    event_panel = df.merge(event_panel, on=["state_n", "year"], how="inner")

    for outcome in ["log_nloans", "log_vloans", "log_nden", "log_lir", "log_nsold"]:
        run_event_study(
            event_panel,
            outcome,
            leads=4,
            lags=4,
            ref=-1,
            controls=[
                "log_inc",
                "log_pop",
                "log_hpi",
                "log_her_v",
                "L1_log_inc",
                "L1_log_pop",
                "L1_log_hpi",
                "L1_log_her_v",
                f"L1_{outcome}_b",
                f"L1_{outcome}_pl",
            ],
            fes=["county", "event_id"],
        )


# %% Replicate Fabra & Imbs tables


def fabra_imbs_replication(data_folder):
    folder = os.path.join(
        data_folder, "raw", "original", "fabra_imbs_2015", "20121416_1data", "data"
    )
    hmda = pd.read_stata(os.path.join(folder, "hmda.dta"))
    hp_dereg = pd.read_stata(os.path.join(folder, "hp_dereg_controls.dta"))
    call = pd.read_stata(os.path.join(folder, "call.dta"))

    df = hmda.merge(
        hp_dereg[
            [c for c in hp_dereg.columns if c not in hmda.columns] + ["county", "year"]
        ],
        on=["county", "year"],
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        call[[c for c in call.columns if c not in df.columns] + ["county", "year"]],
        on=["county", "year"],
        how="left",
        validate="one_to_one",
    )

    # %%
    controls = [
        "Dl_inc",
        "LDl_inc",
        "Dl_pop",
        "LDl_pop",
        "Dl_hpi",
        "LDl_hpi",
        "Dl_her_v",
        "LDl_her_v",
    ]

    for col in ["Dl_nloans_b", "Dl_vloans_b", "Dl_nden_b", "Dl_lir_b", "Dl_nsold_b"]:
        model = feols(
            f"{col} ~ Linter_bra + {'+'.join(controls)} + L{col} | year + county",
            data=df,
            vcov={"CRV1": "state_n"},
        )
        print(col)
        print("=" * 20)
        print(model.summary())
        print("\n\n")

    # %% IV regression

    # %%
