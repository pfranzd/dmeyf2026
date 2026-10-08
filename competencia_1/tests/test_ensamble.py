import polars as pl
import pytest

from competencia_1.pipeline.ensamble import coincidencias, combinar, normalizar_rank


def _probas(ids, probs):
    return pl.DataFrame({"numero_de_cliente": ids, "prob": probs}).cast(
        {"numero_de_cliente": pl.Int64}
    )


def test_rank_ignora_la_escala_de_probabilidad():
    # mismo orden, escalas muy distintas: el ensamble por rank conserva ese orden
    a = _probas([1, 2, 3, 4], [0.9, 0.5, 0.2, 0.1])
    b = _probas([1, 2, 3, 4], [0.009, 0.005, 0.002, 0.001])
    ens = combinar([a, b], "rank")
    orden = ens.sort("prob", descending=True)["numero_de_cliente"].to_list()
    assert orden == [1, 2, 3, 4]
    assert ens["prob"].is_between(0, 1).all()


def test_rank_promedia_ordenes_distintos():
    a = _probas([1, 2, 3], [0.9, 0.5, 0.1])  # 1 > 2 > 3
    b = _probas([1, 2, 3], [0.1, 0.5, 0.9])  # 3 > 2 > 1
    ens = combinar([a, b], "rank")
    assert ens["prob"].to_list() == pytest.approx([2 / 3, 2 / 3, 2 / 3])


def test_prob_ponderado():
    a = _probas([1, 2], [0.8, 0.2])
    b = _probas([1, 2], [0.2, 0.4])
    ens = combinar([a, b], "prob", pesos=[3, 1])
    assert ens["prob"].to_list() == pytest.approx([0.65, 0.25])


def test_no_depende_del_orden_de_filas():
    a = _probas([1, 2, 3], [0.9, 0.5, 0.1])
    b = _probas([3, 1, 2], [0.2, 0.7, 0.6])
    assert combinar([a, b]).equals(combinar([a, b.reverse()]))


def test_rechaza_clientes_distintos_y_un_solo_modelo():
    a = _probas([1, 2], [0.9, 0.1])
    with pytest.raises(ValueError, match="mismos clientes"):
        combinar([a, _probas([1, 3], [0.9, 0.1])])
    with pytest.raises(ValueError, match="al menos 2"):
        combinar([a])
    with pytest.raises(ValueError, match="pesos"):
        combinar([a, a], pesos=[1, 0])


def test_normalizar_rank_en_rango():
    r = normalizar_rank(_probas([1, 2, 3, 4], [0.3, 0.3, 0.9, 0.1]))
    assert r["prob"].max() == 1.0 and r["prob"].min() > 0


def test_coincidencias():
    a = _probas([1, 2, 3, 4], [0.9, 0.8, 0.2, 0.1])
    b = _probas([1, 2, 3, 4], [0.9, 0.1, 0.8, 0.2])
    res = coincidencias({"a": a, "b": b}, combinar([a, b]), [2])
    assert res["2"]["jaccard_medio_pares"] == pytest.approx(1 / 3)
