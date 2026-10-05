"""Semillerío: N modelos con distinta semilla -> N probas + promedio -> N x cortes CSV."""

import json
import re

import numpy as np
import polars as pl
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import final, salida, tracking
from competencia_1.pipeline.semillas import validar_semilla_curso
from config.semillas import SEMILLAS

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"
MANUAL = (
    "lgbm.manual={learning_rate: 0.1, num_iterations: 15, feature_fraction: 0.5, "
    "num_leaves: 8, min_data_in_leaf: 20}"
)
N = 3000


def _cfg(*ov):
    base = [MANUAL, "lgbm.fijos.num_threads=2", "dataset.undersampling=0.5"]
    return cfgmod.cargar_config(BASE_YAML, [*base, *ov])[0]


@pytest.fixture
def parquet(tmp_path):
    """202106 con target (train) y 202108 sin target (a predecir)."""
    rng = np.random.default_rng(2)
    filas = []
    for mes in (202106, 202108):
        clase = rng.choice(["CONTINUA", "BAJA+1", "BAJA+2"], N, p=[0.94, 0.03, 0.03])
        x = rng.normal(size=(N, 20)) + (clase == "BAJA+2")[:, None] * np.linspace(
            1.2, 0, 20
        )
        d = {f"x{i}": x[:, i] for i in range(20)}
        d.update(
            numero_de_cliente=np.arange(N, dtype=np.int64) + 48_000_000,
            foto_mes=np.full(N, mes, dtype=np.int32),
            clase_ternaria=clase if mes == 202106 else [None] * N,
        )
        filas.append(pl.DataFrame(d, schema_overrides={"clase_ternaria": pl.Utf8}))
    p = tmp_path / "features.parquet"
    pl.concat(filas).write_parquet(p)
    return p


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(tracking, "RUNS_CSV", tmp_path / "runs.csv")
    (tmp_path / "runs").mkdir()
    with tracking.Run("t_semillerio", {"x": 1}) as r:
        yield r


def _entrenar(cfg, parquet, run, monkeypatch):
    monkeypatch.setattr(final, "construir_features", lambda c: (parquet, "hash"))
    final.etapa_final(cfg, run)


def test_final_entrena_n_modelos_distintos_y_promedia(parquet, run, monkeypatch):
    cfg = _cfg("final.n_semillas=3")
    _entrenar(cfg, parquet, run, monkeypatch)
    probas = {
        p.stem: pl.read_parquet(p) for p in (run.dir / "probas").glob("*.parquet")
    }
    assert set(probas) == {f"s{s}" for s in SEMILLAS[:3]} | {"promedio"}
    assert all((run.dir / "modelos" / f"s{s}.txt").exists() for s in SEMILLAS[:3])
    # misma lista de clientes (Int64) y orden en todos los modelos
    ids = probas["promedio"]["numero_de_cliente"]
    assert ids.dtype == pl.Int64 and ids.n_unique() == N
    for df in probas.values():
        assert df["numero_de_cliente"].to_list() == ids.to_list()
    # los modelos difieren entre sí y el promedio es la media exacta
    p = [probas[f"s{s}"]["prob"].to_numpy() for s in SEMILLAS[:3]]
    assert not np.array_equal(p[0], p[1]) and not np.array_equal(p[1], p[2])
    assert np.allclose(probas["promedio"]["prob"].to_numpy(), np.mean(p, axis=0))
    assert (run.dir / "importancias.csv").exists()


def test_una_sola_semilla_no_genera_promedio(parquet, run, monkeypatch):
    _entrenar(_cfg("final.n_semillas=1"), parquet, run, monkeypatch)
    assert not (run.dir / "probas" / "promedio.parquet").exists()
    assert len(list((run.dir / "probas").glob("s*.parquet"))) == 1


def test_primer_modelo_no_cambia_al_agregar_semillas(
    parquet, run, monkeypatch, tmp_path
):
    _entrenar(_cfg("final.n_semillas=1"), parquet, run, monkeypatch)
    uno = pl.read_parquet(run.dir / "probas" / f"s{SEMILLAS[0]}.parquet")
    with tracking.Run("t_semillerio2", {"x": 1}) as r2:
        _entrenar(_cfg("final.n_semillas=2"), parquet, r2, monkeypatch)
        dos = pl.read_parquet(r2.dir / "probas" / f"s{SEMILLAS[0]}.parquet")
    assert uno["prob"].to_list() == dos["prob"].to_list()


def _preparar_salida(cfg, parquet, run, monkeypatch):
    _entrenar(cfg, parquet, run, monkeypatch)
    monkeypatch.setattr(salida, "CACHE_BASE", parquet)
    run.registrar(validacion={"envios_optimos": [100], "n_clientes_valid": N})


def test_salida_un_csv_por_modelo_y_corte(parquet, run, monkeypatch):
    cfg = _cfg(
        "final.n_semillas=3", "salida.envios=[100, 250]", "periodos.target=202108"
    )
    _preparar_salida(cfg, parquet, run, monkeypatch)
    salida.etapa_salida(cfg, run)
    csvs = sorted((run.dir / "submits").glob("*.csv"))
    assert len(csvs) == (3 + 1) * 2  # 3 semillas + promedio, 2 cortes
    nombres = {c.name for c in csvs}
    assert f"{run.run_id}_s{SEMILLAS[0]}_e100.csv" in nombres
    assert f"{run.run_id}_promedio_e250.csv" in nombres
    for c in csvs:
        lineas = c.read_text().splitlines()
        assert all(re.fullmatch(r"\d+", x) for x in lineas)
        assert len(lineas) == int(re.search(r"_e(\d+)\.csv", c.name).group(1))
    # el top-N del promedio es consistente con sus probabilidades
    prom = pl.read_parquet(run.dir / "probas" / "promedio.parquet")
    esperado = salida.seleccionar_top(prom, 100).to_list()
    leido = [
        int(x)
        for x in (run.dir / "submits" / f"{run.run_id}_promedio_e100.csv")
        .read_text()
        .split()
    ]
    assert leido == esperado
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["envios"] == [100, 250] and len(meta["archivos"]) >= 8
    sem = meta["semillerio"]["100"]
    assert sem["n_modelos"] == 3 and 0 < sem["jaccard_medio_pares"] <= 1
    assert 0 < sem["jaccard_medio_vs_promedio"] <= 1


def test_salida_sin_promedio_si_esta_apagado(parquet, run, monkeypatch):
    cfg = _cfg(
        "final.n_semillas=2",
        "final.promedio=false",
        "salida.envios=100",
        "periodos.target=202108",
    )
    _preparar_salida(cfg, parquet, run, monkeypatch)
    salida.etapa_salida(cfg, run)
    assert not list((run.dir / "submits").glob("*promedio*"))
    assert len(list((run.dir / "submits").glob("*.csv"))) == 2


def test_salida_rechaza_probas_que_no_cubren_el_target(parquet, run, monkeypatch):
    cfg = _cfg("final.n_semillas=1", "salida.envios=100", "periodos.target=202108")
    _preparar_salida(cfg, parquet, run, monkeypatch)
    f = next((run.dir / "probas").glob("s*.parquet"))
    pl.read_parquet(f).head(N - 1).write_parquet(f)  # falta un cliente
    with pytest.raises(ValueError, match="no cubren"):
        salida.etapa_salida(cfg, run)


def test_validacion_de_semillas_del_curso():
    for s in SEMILLAS:
        validar_semilla_curso(s)
    for malo in (111624, 99991, 1000003, 100003 * 2):  # compuesto / fuera de rango
        with pytest.raises(ValueError, match="primo"):
            validar_semilla_curso(malo)
    with pytest.raises(TypeError):
        validar_semilla_curso("111623")
    with pytest.raises(ValueError, match="repetidas"):
        _cfg("final.semillas=[111623, 111623]")
    with pytest.raises(ValueError, match="primo"):
        _cfg("final.semillas=[111623, 111624]")
    with pytest.raises(ValueError, match="semilla_maestra"):
        _cfg("semilla_maestra=111624")
    assert _cfg("final.semillas=[111623, 116239]").final.semillas == [111623, 116239]
