import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import joblib
    from vulnerability_networks.config import PATH_MODELS
    import pandas as pd
    import matplotlib.pyplot as plt
    import scienceplots

    return PATH_MODELS, joblib, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # MultiStage evaluation

    ## 1. Explore
    Based on 200 trials with wide search and low number of epochs (5-10), take the 20% of best trials and refine them:
    """
    )
    return


@app.cell
def _(PATH_MODELS, joblib):
    study_ge = joblib.load(PATH_MODELS/"explore/global_efficiency/optuna_study.pkl")
    study_nip = joblib.load(PATH_MODELS/"explore/independent_path/optuna_study.pkl")
    df_ge = study_ge.trials_dataframe().sort_values("value", ascending=False)
    df_nip = study_nip.trials_dataframe().sort_values("value", ascending=False)
    top_k = int(0.20*len(df_ge))
    df_ge = df_ge.head(top_k)
    df_nip = df_nip.head(top_k)
    return df_ge, df_nip


@app.cell
def _(df_ge):
    df_ge#.to_csv(PATH_MODELS/"explore/global_efficiency/top40_archs.csv", index=False)
    return


@app.cell
def _(df_nip):
    df_nip#.to_csv(PATH_MODELS/"explore/independent_path/top40_archs.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##2. Refining

    Based on those 20% of best trials, prune obsolete hyperparameters and do again a search with 50 epochs and smaller range of hyperparameters
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(PATH_MODELS, joblib):
    study_ge_ref = joblib.load(PATH_MODELS/"refine/global_efficiency/optuna_study.pkl")
    study_nip_ref = joblib.load(PATH_MODELS/"refine/independent_path/optuna_study.pkl")
    df_ge_ref = study_ge_ref.trials_dataframe().sort_values("value", ascending=False)
    df_nip_ref = study_nip_ref.trials_dataframe().sort_values("value", ascending=False)
    df_ge_ref = df_ge_ref.head(10)
    df_nip_ref = df_nip_ref.head(10)
    return df_ge_ref, df_nip_ref


@app.cell
def _(PATH_MODELS, df_ge_ref):
    df_ge_ref.to_csv(PATH_MODELS/"refine/global_efficiency/top10.csv", index=False)
    return


@app.cell
def _(df_ge_ref):
    df_ge_ref#.head(10)#.to_csv(PATH_MODELS/"refine/global_efficiency/top_10.csv", index=False)#.describe(include="all")
    return


@app.cell
def _():
    #df_nip_ref.to_csv(PATH_MODELS/"refine/independent_path/top10.csv", index=False)
    return


@app.cell
def _(df_nip_ref):
    df_nip_ref
    return


@app.cell
def _(mo):
    mo.md(r"""# Final Stage""")
    return


@app.cell
def _(PATH_MODELS, joblib):
    study_ge_val = joblib.load(PATH_MODELS/"validate/global_efficiency/optuna_study.pkl")
    study_nip_val = joblib.load(PATH_MODELS/"validate/independent_path/optuna_study.pkl")
    df_ge_val = study_ge_val.trials_dataframe().sort_values("value", ascending=False)
    df_nip_val = study_nip_val.trials_dataframe().sort_values("value", ascending=False)
    # df_ge_val = df_ge_val.head(10)
    # df_nip_val = df_nip_val.head(10
    return df_ge_val, df_nip_val


@app.cell
def _(df_ge_val):
    df_ge_val
    return


@app.cell
def _(df_nip_val):
    df_nip_val
    return


@app.cell
def _(mo):
    mo.md(r"""# BEST MODEL""")
    return


@app.cell
def _(PATH_MODELS, pd):
    path_ge_best = PATH_MODELS/"validate/global_efficiency/version_7"
    df_ge_best = pd.read_csv(path_ge_best/"metrics.csv")
    train_loss_rank = df_ge_best["train_loss_RANK"].dropna().values
    df_ge_best = df_ge_best.dropna(subset=["val_loss_RANK"])
    df_ge_best["train_loss_RANK"] = train_loss_rank
    df_ge_best["epoch"] += 1
    return df_ge_best, path_ge_best


@app.cell
def _(df_ge_best):
    df_ge_best
    return


@app.cell
def _():
    from matplotlib import rcParams
    return


@app.cell
def _(df_ge_best, path_ge_best, plt):
    plt.style.use(['science','ieee'])
    rc_fonts = {
        "font.size": 10,
        # 'figure.figsize': (4, 3),
    }
    plt.rcParams.update(rc_fonts)
    plt.plot(df_ge_best["epoch"], df_ge_best["train_loss_RANK"], label="train loss")
    plt.plot(df_ge_best["epoch"], df_ge_best["val_loss_RANK"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_ge_best/"plot_loss.pdf")
    plt.show()
    return


@app.cell
def _(PATH_MODELS, pd):
    path_ip_best = PATH_MODELS/"validate/independent_path/version_3"
    df_ip_best = pd.read_csv(path_ip_best/"metrics.csv")
    train_loss_rank_ip = df_ip_best["train_loss_RANK"].dropna().values
    df_ip_best = df_ip_best.dropna(subset=["val_loss_RANK"])
    df_ip_best["train_loss_RANK"] = train_loss_rank_ip
    df_ip_best["epoch"] += 1
    return df_ip_best, path_ip_best


@app.cell
def _(df_ip_best):
    df_ip_best
    return


@app.cell
def _(df_ip_best):
    df_ip_best2 = df_ip_best[["epoch", "train_loss_RANK", "val_loss_RANK"]][:20]
    return


@app.cell
def _(df_ip_best, path_ip_best, plt):
    plt.plot(df_ip_best["epoch"], df_ip_best["train_loss_RANK"], label="train loss")
    plt.plot(df_ip_best["epoch"], df_ip_best["val_loss_RANK"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_ip_best/"plot_loss.pdf")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
