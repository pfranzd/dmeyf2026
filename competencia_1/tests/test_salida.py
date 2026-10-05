import polars as pl
import pytest

from competencia_1.pipeline.salida import (
    escribir_csv_ids,
    resolver_envios,
    seleccionar_top,
    validar_csv,
)

IDS_GRANDES = [48_000_000, 100_000_000, 999_999_999_999, 29_000_001, 12_345_678_901]


def _probas(ids, probs=None):
    probs = probs or [0.9 - 0.01 * i for i in range(len(ids))]
    return pl.DataFrame({"numero_de_cliente": ids, "prob": probs}).cast(
        {"numero_de_cliente": pl.Int64}
    )


def test_ids_grandes_sin_notacion_cientifica(tmp_path):
    p = escribir_csv_ids(_probas(IDS_GRANDES), 5, tmp_path / "s.csv")
    lineas = p.read_text().split("\n")
    assert lineas == [str(i) for i in IDS_GRANDES] + [""]
    assert not any(c in p.read_text() for c in "e.,\"' ")


def test_sin_header_y_orden_por_prob(tmp_path):
    ids = [3, 1, 2]
    p = escribir_csv_ids(_probas(ids, [0.1, 0.9, 0.5]), 2, tmp_path / "s.csv")
    assert p.read_text() == "1\n2\n"


def test_desempate_determinista():
    top = seleccionar_top(_probas([30, 10, 20], [0.5, 0.5, 0.5]), 2)
    assert top.to_list() == [10, 20]


def test_rechaza_ids_float_nulos_duplicados_y_prob_invalida(tmp_path):
    flotante = pl.DataFrame({"numero_de_cliente": [4.8e7, 1.0], "prob": [0.5, 0.4]})
    with pytest.raises(TypeError):
        escribir_csv_ids(flotante, 1, tmp_path / "a.csv")
    with pytest.raises(ValueError):
        escribir_csv_ids(_probas([1, 1]), 1, tmp_path / "b.csv")
    with pytest.raises(ValueError):
        escribir_csv_ids(_probas([1, 2], [0.5, 1.5]), 1, tmp_path / "c.csv")
    with pytest.raises(ValueError):
        escribir_csv_ids(_probas([1, 2]), 3, tmp_path / "d.csv")
    assert not list(tmp_path.glob("*.csv"))


@pytest.mark.parametrize(
    "contenido",
    ["4.8e+07\n", "48000000.0\n", "id\n1\n", '"1"\n', "-1\n", "1\r\n", "1\n1\n"],
)
def test_validar_csv_detecta_formatos_malos(tmp_path, contenido):
    p = tmp_path / "x.csv"
    p.write_bytes(contenido.encode())
    with pytest.raises(ValueError):
        validar_csv(p, len(contenido.splitlines()))


def test_validar_csv_pertenencia_y_borrado(tmp_path):
    with pytest.raises(ValueError, match="no pertenecen"):
        escribir_csv_ids(_probas([1, 2, 3]), 3, tmp_path / "e.csv", ids_validos={1, 2})
    assert not (tmp_path / "e.csv").exists()


def test_resolver_envios():
    assert resolver_envios(9000) == [9000]
    assert resolver_envios([11000, 9000, 9000]) == [9000, 11000]
    assert resolver_envios("auto", [9800, 10200, 10100], 164647, 164114) == [10000]
    with pytest.raises(ValueError):
        resolver_envios("auto")
