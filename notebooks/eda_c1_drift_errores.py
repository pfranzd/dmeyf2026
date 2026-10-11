# %% [markdown]
# # EDA competencia_1: drift, importancias y errores de e032
#
# Informa los experimentos e033-e037 (ver `competencia_1/configs/exp/README.md`).
# Solo lee: el parquet base, el caché de features de e032 (fe_hash 6035dfc967f1) y los
# 24 modelos de la entrega definitiva. Para el análisis de errores entrena UN modelo corto
# en el fold B de e032 (train 202103-202104 -> valid 202106) con los params del mejor trial.
#
# Uso: `python notebooks/eda_c1_drift_errores.py` desde la raíz del repo (o celda a celda).

# %%
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.metricas import auc

RAIZ = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RUN = RAIZ / "work/competencia_1/runs/20261009-232755_e032_expansiva_auc_topk"
FEATURES = RAIZ / "work/competencia_1/features/6035dfc967f1/features.parquet"
BASE = RAIZ / "work/competencia_1/cache/base.parquet"
MODELOS = RAIZ / "competencia_1/definitiva/modelos"
SEED = 111623  # config/semillas.py: SEMILLA_PRIMARIA
FEATS = (RUN / "features.txt").read_text(encoding="utf-8").split()
FIJOS = {
    "objective": "binary",
    "max_bin": 31,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 8,
    "verbosity": -1,
    "feature_pre_filter": False,
    "seed": SEED,
}


# %% [markdown]
# ## 1. Importancia por familia en los 24 modelos de e032


# %%
def familia(f: str) -> str:
    reglas = [
        (r"^lag_", "3_lags"),
        (r"^deltapct_", "4_deltapct"),
        (r"^delta_", "4_deltas"),
        (r"^(avg|max|min|std)_\d+_", "5_ventanas"),
        (r"^ratioavg_", "5_ratioavg"),
        (r"^slope_", "6_tendencia"),
        (r"^(pr|d10)_", "2_rankings"),
        (r"^tc_", "1a_tc"),
        (r"^r_", "1c_ratios"),
        (r"^f_", "1e_flags"),
        (r"^[mc]_", "1d_agregados"),
        (r"^(meses_en_panel|antiguedad_panel)$", "7_historia"),
    ]
    for patron, nombre in reglas:
        if re.match(patron, f):
            return nombre
    return "0_originales"


def es_monto(f: str) -> bool:
    """Feature cuyo campo de origen es un monto nominal en pesos."""
    base = re.sub(
        r"^(lag_\d+_|delta_\d+_|deltapct_\d+_|(avg|max|min|std)_\d+_|ratioavg_\d+_|slope_\d+_|tc_)",
        "",
        f,
    )
    base = re.sub(r"^(Master|Visa)_", "", base)
    return base.startswith("m") and not f.startswith(
        ("pr_", "d10_", "deltapct_", "ratioavg_")
    )


imp = []
for p in sorted(MODELOS.glob("*.txt")):
    b = lgb.Booster(model_file=str(p))
    imp.append(
        pl.DataFrame(
            {"feature": b.feature_name(), "gain": b.feature_importance("gain")}
        )
    )
imp = pl.concat(imp).group_by("feature").agg(pl.col("gain").sum())
imp = imp.with_columns(
    pl.col("feature").map_elements(familia, return_dtype=pl.String).alias("familia"),
    pl.col("feature")
    .map_elements(es_monto, return_dtype=pl.Boolean)
    .alias("monto_nominal"),
    (pl.col("gain") / pl.col("gain").sum()).alias("share"),
)
print(
    imp.group_by("familia")
    .agg(pl.col("share").sum(), pl.len())
    .sort("share", descending=True)
)
print(
    "share de features sobre montos nominales:",
    round(imp.filter("monto_nominal")["share"].sum(), 3),
)
print(imp.sort("gain", descending=True).head(25).select("feature", "familia", "share"))

# %% [markdown]
# ## 2. Validación adversarial: ¿cuánto se distingue 202108 de los meses de train?
# Se excluyen las features que cuentan meses (antigüedad, fechas en días) porque separan
# los meses por construcción y no dicen nada del drift.

# %%
TRIVIALES = re.compile(
    r"antiguedad|meses_en_panel|fecha|Fvencimiento|fultimo_cierre|Finiciomora"
)
feats_adv = [f for f in FEATS if not TRIVIALES.search(f)]


def adversarial(mes_a: int, mes_b: int, feats: list[str]) -> tuple[float, pl.DataFrame]:
    df = (
        pl.scan_parquet(FEATURES)
        .filter(pl.col("foto_mes").is_in([mes_a, mes_b]))
        .select("numero_de_cliente", "foto_mes", *feats)
        .collect()
    )
    X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    y = (df["foto_mes"] == mes_b).to_numpy().astype(np.int8)
    rng = np.random.default_rng(SEED)
    test = rng.random(len(y)) < 0.3
    params = {
        **FIJOS,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.5,
        "min_data_in_leaf": 200,
    }
    b = lgb.train(
        params, lgb.Dataset(X[~test], y[~test], feature_name=feats), num_boost_round=200
    )
    a = auc(y[test].astype(bool), b.predict(X[test]))
    top = pl.DataFrame({"feature": feats, "gain": b.feature_importance("gain")})
    top = top.with_columns(pl.col("gain") / pl.col("gain").sum()).sort(
        "gain", descending=True
    )
    return a, top


for a_mes in (202106, 202104):
    a, top = adversarial(a_mes, 202108, feats_adv)
    print(f"\nadversarial {a_mes} vs 202108: AUC {a:.4f}")
    print(top.head(15))

# %% [markdown]
# ## 3. Errores en el fold B (train 202103-202104 con undersampling 0,25 -> valid 202106)

# %%
best = json.loads((RUN / "best_params.json").read_text(encoding="utf-8"))
params = {**FIJOS, **best["params"]}
n_iter = int(params.pop("num_iterations"))
tr = cargar_particion(
    FEATURES, FEATS, [202103, 202104], ["BAJA+1", "BAJA+2"], 0.25, SEED
)
va = cargar_particion(FEATURES, FEATS, [202106], ["BAJA+1", "BAJA+2"])
booster = lgb.train(
    params, lgb.Dataset(tr.X, tr.y, feature_name=FEATS), num_boost_round=n_iter
)
score = booster.predict(va.X)
print("AUC fold B:", round(auc(va.es_baja2, score), 4))

orden = np.argsort(-score)
top10k = np.zeros(len(score), dtype=bool)
top10k[orden[:10_000]] = True

seg_cols = [
    "numero_de_cliente",
    "clase_ternaria",
    "cpayroll_trx",
    "ctrx_quarter",
    "cliente_antiguedad",
    "mrentabilidad_annual",
    "ctarjeta_visa",
    "ctarjeta_master",
    "cproductos",
    "mcuentas_saldo",
    "cliente_edad",
    "mpayroll",
]
seg = (
    pl.scan_parquet(BASE)
    .filter(pl.col("foto_mes") == 202106)
    .select(seg_cols)
    .collect()
)
dfv = pl.DataFrame(
    {"numero_de_cliente": va.ids, "score": score, "top10k": top10k}
).join(seg, on="numero_de_cliente")
q_rent = dfv["mrentabilidad_annual"].quantile(0.9)
dfv = dfv.with_columns(
    pl.when(pl.col("ctrx_quarter").fill_null(0) == 0)
    .then(pl.lit("inactivo (ctrx_quarter=0)"))
    .when(pl.col("cpayroll_trx").fill_null(0) == 0)
    .then(pl.lit("sin payroll"))
    .otherwise(pl.lit("con payroll"))
    .alias("seg_actividad"),
    pl.when(pl.col("cliente_antiguedad") < 12)
    .then(pl.lit("<12 meses"))
    .when(pl.col("cliente_antiguedad") < 60)
    .then(pl.lit("1-5 años"))
    .otherwise(pl.lit(">5 años"))
    .alias("seg_antiguedad"),
    pl.when(
        (pl.col("ctarjeta_visa").fill_null(0) + pl.col("ctarjeta_master").fill_null(0))
        == 0
    )
    .then(pl.lit("sin TC"))
    .otherwise(pl.lit("con TC"))
    .alias("seg_tc"),
    pl.when(pl.col("mrentabilidad_annual") >= q_rent)
    .then(pl.lit("top10% rentab"))
    .otherwise(pl.lit("resto"))
    .alias("seg_rentab"),
    (pl.col("clase_ternaria") == "BAJA+2").alias("b2"),
)
for s in ("seg_actividad", "seg_antiguedad", "seg_tc", "seg_rentab"):
    t = (
        dfv.group_by(s)
        .agg(
            pl.len().alias("clientes"),
            pl.col("b2").sum().alias("baja2"),
            (pl.col("b2") & pl.col("top10k")).sum().alias("baja2_capturadas"),
            pl.col("top10k").sum().alias("en_top10k"),
        )
        .with_columns(
            (pl.col("baja2_capturadas") / pl.col("baja2")).round(3).alias("recall"),
            (pl.col("baja2_capturadas") / pl.col("en_top10k"))
            .round(4)
            .alias("precision"),
            ((pl.col("baja2") - pl.col("baja2_capturadas")) * 273_000 / 1e6)
            .round(1)
            .alias("M_no_capturados"),
        )
        .sort("clientes", descending=True)
    )
    print(t)

# %% [markdown]
# ## 4. ¿Cómo puntúa el modelo a BAJA+1 frente a BAJA+2?

# %%
print(
    dfv.group_by("clase_ternaria").agg(
        pl.col("score").mean().alias("score_medio"),
        pl.col("score").median().alias("score_mediana"),
        pl.col("top10k").mean().alias("frac_en_top10k"),
    )
)

# %% [markdown]
# ## 5. Aguinaldo y saltos de escala en montos

# %%
pay = (
    pl.scan_parquet(BASE)
    .filter(pl.col("mpayroll") > 0)
    .group_by("foto_mes")
    .agg(pl.len(), pl.col("mpayroll").median().alias("mediana"))
    .sort("foto_mes")
    .collect()
)
print(pay)
pv = (
    pl.scan_parquet(BASE)
    .filter(pl.col("foto_mes").is_in([202105, 202106, 202107]))
    .select("numero_de_cliente", "foto_mes", "mpayroll")
    .collect()
    .pivot(on="foto_mes", index="numero_de_cliente", values="mpayroll")
)
pv = pv.filter((pl.col("202105") > 0) & (pl.col("202106") > 0))
salto = pv.select(((pl.col("202106") / pl.col("202105")) > 1.3).mean()).item()
print(
    f"clientes con payroll en 202105 y 202106: {pv.height} | 202106/202105 > 1,3: {salto:.1%}"
)
