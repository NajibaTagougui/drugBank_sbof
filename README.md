# SBOF–DrugBank: Schrödinger Bridge Onsager Flow for Drug Repurposing

Reference implementation of the **SBOF** algorithm and the **SBOFNet** neural
parameterisation, evaluated on DrugBank 5.1.10.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)


## Repository layout

| Path | Contents |
|------|----------|
| `scripts/sbof_main.py` | Analytic SBOF, baselines, dataset loader, figures |
| `scripts/sbof_train.py` | SBOFNet neural trainer + evaluation |
| `scriptsenrich_smiles.py` | PubChem/RDKit SMILES enrichment utility |
| `data/drugbank_real.csv` | DrugBank 5.1.10 small-molecule descriptors |
| `data/top_drugs.csv` | Top-50 SBOF-ranked molecules |

Figures, training logs, and model checkpoints are **not** included in the repo.


## Quickstart


git clone https://github.com/NajibaTagougui/drugBank_sbof.git
cd rugBank_sbof
pip install -r requirements.txt

## Analytic SBOF (fast, CPU)

python scripts/sbof_main.py --csv data/drugbank_real.csv --kernel-n 3000

## Neural SBOFNet (GPU recommended)

python scripts/sbof_train.py --epochs 500

## Evaluate a saved checkpoint

python scripts/sbof_train.py --evaluate results/checkpoints/best_model.pt
