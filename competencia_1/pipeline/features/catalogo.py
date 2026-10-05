"""Catálogo de features: cada fragmento de SQL se registra al generarse.

Así la query y su documentación no pueden desincronizarse (patrón `_reg` de
exp/z402_feature_engineering/fe_sql.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass
class Catalogo:
    filas: list[dict] = field(default_factory=list)

    def reg(self, bloque: str, feature: str, expresion: str, origen: str = "") -> str:
        """Registra la feature y devuelve su línea de SELECT."""
        if feature in self.nombres():
            raise ValueError(f"feature duplicada en el catálogo: {feature}")
        self.filas.append(
            {
                "bloque": bloque,
                "feature": feature,
                "expresion": expresion,
                "origen": origen,
            }
        )
        return f"\n    , {expresion} as {feature}"

    def nombres(self) -> list[str]:
        return [f["feature"] for f in self.filas]

    def a_polars(self) -> pl.DataFrame:
        return pl.DataFrame(
            self.filas,
            schema={
                "bloque": pl.Utf8,
                "feature": pl.Utf8,
                "expresion": pl.Utf8,
                "origen": pl.Utf8,
            },
        )

    def guardar(self, path: Path) -> None:
        self.a_polars().write_csv(path)
