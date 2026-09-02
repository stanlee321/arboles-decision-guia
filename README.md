# Arboles de decision: guia introductoria reproducible

Mini-proyecto docente en espanol para preparar una exposicion introductoria
sobre arboles de decision. El repositorio conecta tres piezas que usan los
mismos datos y el mismo modelo:

1. una guia breve en LaTeX/PDF;
2. un Jupyter notebook ejecutable, con explicaciones, entradas y salidas;
3. scripts reutilizables que regeneran los datos, resultados y figuras.

> **Aviso:** el caso utiliza pacientes completamente simulados. Sirve para
> aprender ciencia de datos y no debe usarse para decisiones clinicas.

## Estructura

```text
.
├── data/
│   ├── raw/                 # entrada simulada e inmutable
│   └── processed/           # reservado para transformaciones futuras
├── docs/
│   ├── figures/             # figuras generadas por el pipeline
│   ├── generated/           # fragmentos LaTeX generados
│   └── guide/               # fuente principal del libro breve
├── notebooks/               # recorrido narrativo y ejecutable
├── output/
│   ├── pdf/                 # guia final
│   └── results/             # metricas, reglas y predicciones
├── scripts/                 # puntos de entrada reproducibles
├── src/arboles_decision/    # logica reutilizable
└── tests/                   # comprobaciones del flujo y de la matematica
```

## Instalacion

Con `conda`:

```bash
conda env create -f environment.yml
conda activate arboles-decision
```

O con un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Para compilar el PDF tambien se necesita
[Tectonic](https://tectonic-typesetting.github.io/). Para la revision visual
automatizada se recomienda Poppler (`pdftoppm` y `pdfinfo`).

## Reproducir todo

```bash
make all
```

Los pasos individuales son:

```bash
make data       # crea el CSV simulado con semilla fija
make results    # entrena, evalua y crea figuras/resultados
make notebook   # ejecuta el notebook y conserva sus salidas
make pdf        # compila la guia en output/pdf/
make slides     # compila la presentacion Beamer en output/pdf/
make test       # ejecuta las pruebas
```

Si Python o Jupyter tienen otro nombre, se pueden indicar explicitamente:

```bash
make all PYTHON="conda run -n arboles-decision python" \
         JUPYTER="conda run -n arboles-decision jupyter"
```

## Productos principales

- [Guia PDF](docs/guide/arboles-decision-guia.pdf): texto introductorio completo,
  almacenado junto a su fuente LaTeX.
- [Presentacion PDF](docs/slides/arboles-decision-presentacion.pdf): 11
  diapositivas almacenadas junto a su fuente LaTeX.
- `output/pdf/`: copias de distribucion de ambos documentos.
- `notebooks/01_arboles_decision.ipynb`: practica guiada con salidas guardadas.
- `output/results/metrics.json`: metricas sobre el conjunto de prueba.
- `output/results/tree_rules.txt`: reglas legibles aprendidas por el arbol.
- `output/results/predicciones_ejemplo.csv`: entradas y predicciones de ejemplo.

## Principios de reproducibilidad

- Los datos simulados y la division entrenamiento/prueba usan semillas fijas.
- El conjunto de prueba no participa en el entrenamiento.
- La guia, el notebook y los archivos de resultados llaman a la misma logica en
  `src/`.
- Los resultados generados incluyen versiones de Python y scikit-learn.
- Los datos reales nunca deben escribirse encima de `data/raw/`; cualquier
  limpieza futura debe producir un archivo nuevo en `data/processed/`.
