"""
enrich_smiles.py
================
Enriches drugbank_clean.csv with SMILES strings and RDKit descriptors
(logP, HBA, HBD, PSA, RTB, QED) fetched from PubChem's free REST API.

Run on any cloud environment with internet access — no API key required.
Works in Google Colab, AWS, GCP, Azure, Kaggle notebooks, etc.

Usage
-----
  # Colab / cloud terminal
  pip install pandas requests rdkit tqdm

  python enrich_smiles.py \
      --input  drugbank_clean.csv \
      --output drugbank_enriched.csv \
      --checkpoint smiles_checkpoint.csv

  # Or from within a Colab cell:
  # %run enrich_smiles.py --input drugbank_clean.csv --output drugbank_enriched.csv

Strategy
--------
1. Filter to small molecules (100–900 Da) — same filter as sbof_main.py.
2. Look up PubChem in batches of 100 using:
      Priority 1 → CAS number  (most reliable unique identifier)
      Priority 2 → drug name
3. For each CID found, fetch:
      IsomericSMILES, XLogP, HBondDonorCount, HBondAcceptorCount,
      TPSA, RotatableBondCount
4. For drugs where PubChem returns SMILES, also recompute
   all descriptors locally with RDKit for consistency.
5. For remaining drugs with no SMILES, fall back to the MW-conditional
   estimates from sbof_main.py so the output is always complete.
6. Save checkpoint every --save-every rows so progress survives crashes.

Output columns (added to original CSV)
---------------------------------------
  smiles          canonical SMILES (from PubChem, or '' if not found)
  smiles_source   'pubchem_cas' | 'pubchem_name' | 'not_found'
  pubchem_cid     PubChem CID (int or NaN)
  mw              molecular weight (Da)     — real DrugBank value preserved
  logp            XLogP from PubChem, or RDKit MolLogP, or estimated
  hba             H-bond acceptors (RDKit or PubChem)
  hbd             H-bond donors    (RDKit or PubChem)
  psa             topological PSA  (RDKit or PubChem)
  rtb             rotatable bonds  (RDKit or PubChem)
  qed             QED desirability (RDKit, requires SMILES)
  drug_score      composite Lipinski/QED score (same formula as sbof_main.py)
  descriptor_src  'rdkit' | 'pubchem' | 'estimated'
"""

import argparse
import json
import math
import time
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('enrich')

# ── PubChem constants ─────────────────────────────────────────────────────────
PUBCHEM_BASE   = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
PROPS_WANTED   = ('IsomericSMILES,XLogP,HBondDonorCount,'
                  'HBondAcceptorCount,TPSA,RotatableBondCount')
BATCH_SIZE     = 100    # PubChem allows ~100 names per POST request
RATE_LIMIT_RPS = 4      # stay under PubChem's 5 req/s anonymous limit
RETRY_TOTAL    = 4
RETRY_BACKOFF  = 1.5    # seconds


# ══════════════════════════════════════════════════════════════════════════════
# HTTP SESSION
# ══════════════════════════════════════════════════════════════════════════════

def make_session() -> requests.Session:
    """Requests session with retry + exponential back-off."""
    s   = requests.Session()
    ret = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET', 'POST'],
    )
    s.mount('https://', HTTPAdapter(max_retries=ret))
    s.headers.update({'User-Agent': 'sbof-enrich/1.0 (research)'})
    return s


SESSION = make_session()
_last_call: float = 0.0


def _throttle():
    """Sleep just enough to stay under RATE_LIMIT_RPS."""
    global _last_call
    gap     = 1.0 / RATE_LIMIT_RPS
    elapsed = time.perf_counter() - _last_call
    if elapsed < gap:
        time.sleep(gap - elapsed)
    _last_call = time.perf_counter()


# ══════════════════════════════════════════════════════════════════════════════
# PUBCHEM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _post_names(names: list[str]) -> dict[str, dict]:
    """
    POST a batch of names/CAS numbers to PubChem.
    Returns {input_name_lower: property_dict}.
    """
    _throttle()
    url  = f'{PUBCHEM_BASE}/compound/name/property/{PROPS_WANTED}/JSON'
    body = 'name=' + '%0A'.join(requests.utils.quote(n) for n in names)
    try:
        r = SESSION.post(url, data=body,
                         headers={'Content-Type': 'application/x-www-form-urlencoded'},
                         timeout=30)
        if r.status_code == 404:
            return {}   # none of the names found
        r.raise_for_status()
        props = r.json()['PropertyTable']['Properties']
        # PubChem returns them in the same order, but name matching is case-insensitive
        # We'll do a second call to get the mapping name→CID
        return {str(p['CID']): p for p in props}
    except Exception as exc:
        log.debug('POST failed for batch: %s', exc)
        return {}


def _name_to_cid_batch(names: list[str]) -> dict[str, int]:
    """
    Map a list of drug names / CAS strings to their PubChem CIDs.
    Returns {name: cid}.
    """
    _throttle()
    url  = f'{PUBCHEM_BASE}/compound/name/cids/JSON'
    body = 'name=' + '%0A'.join(requests.utils.quote(n) for n in names)
    try:
        r = SESSION.post(url, data=body,
                         headers={'Content-Type': 'application/x-www-form-urlencoded'},
                         timeout=30)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        data = r.json()
        # Response: {"IdentifierList":{"CID":[...]}}  or
        #           {"InformationList":{"Information":[{"Name":...,"CID":[...]}]}}
        info = data.get('InformationList', {}).get('Information', [])
        mapping = {}
        for entry in info:
            name = entry.get('Name', '')
            cids = entry.get('CID', [])
            if cids:
                mapping[name.lower()] = cids[0]
        return mapping
    except Exception as exc:
        log.debug('CID lookup failed: %s', exc)
        return {}


def _cid_to_props(cids: list[int]) -> dict[int, dict]:
    """Fetch SMILES + descriptor properties for a list of CIDs."""
    if not cids:
        return {}
    _throttle()
    cid_str = ','.join(str(c) for c in cids)
    url = f'{PUBCHEM_BASE}/compound/cid/{cid_str}/property/{PROPS_WANTED}/JSON'
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return {p['CID']: p for p in r.json()['PropertyTable']['Properties']}
    except Exception as exc:
        log.debug('CID→props failed: %s', exc)
        return {}


def lookup_batch(identifiers: list[str]) -> dict[str, dict | None]:
    """
    Look up a batch of identifiers (CAS or name) against PubChem.
    Returns {identifier: property_dict_or_None}.
    """
    # Step 1: name→CID mapping
    name_to_cid = _name_to_cid_batch(identifiers)
    cids        = list(set(name_to_cid.values()))

    # Step 2: CID→properties
    cid_props   = _cid_to_props(cids) if cids else {}

    # Step 3: assemble result keyed by original identifier
    result = {}
    for ident in identifiers:
        cid = name_to_cid.get(ident.lower())
        if cid and cid in cid_props:
            result[ident] = {**cid_props[cid], 'CID': cid}
        else:
            result[ident] = None
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RDKIT DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

def rdkit_descriptors(smiles: str) -> dict | None:
    """
    Compute MW, logP, HBA, HBD, PSA, RTB, QED from SMILES using RDKit.
    Returns None if SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, QED as RDKitQED
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'mw_rdkit': Descriptors.MolWt(mol),
            'logp':     Descriptors.MolLogP(mol),
            'hba':      Lipinski.NumHAcceptors(mol),
            'hbd':      Lipinski.NumHDonors(mol),
            'psa':      Descriptors.TPSA(mol),
            'rtb':      Descriptors.NumRotatableBonds(mol),
            'qed':      RDKitQED.qed(mol),
        }
    except ImportError:
        log.warning('RDKit not installed — SMILES found but descriptors not recomputed')
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MW-CONDITIONAL ESTIMATES (fallback, same model as sbof_main.py)
# ══════════════════════════════════════════════════════════════════════════════

_RNG = np.random.default_rng(42)
_QED_MU  = np.array([362.84, 2.58, 3.87, 1.74, 81.49, 5.29])
_QED_SD  = np.array([182.37, 1.99, 2.34, 1.49, 48.25, 3.44])


def estimate_descriptors(mw: float) -> dict:
    """Empirical MW-conditional estimates for logP, HBA, HBD, PSA, RTB, QED."""
    logp = float(np.clip(_RNG.normal(2.5 + 0.002*(mw-350), 1.7), -4, 8))
    hba  = int(np.clip(round(max(0, 0.017*mw + 0.5 + _RNG.normal(0, 1.5))), 0, 16))
    hbd  = int(np.clip(round(max(0, 0.006*mw + 0.3 + _RNG.normal(0, 1.2))), 0, 8))
    psa  = float(np.clip(max(0, 0.20*mw + 6   + _RNG.normal(0, 25.0)), 0, 250))
    rtb  = int(np.clip(round(max(0, 0.013*mw - 0.2 + _RNG.normal(0, 2.5))), 0, 20))
    props = np.array([[mw, logp, hba, hbd, psa, rtb]], dtype=float)
    d    = np.exp(-0.5 * ((props - _QED_MU) / _QED_SD)**2)
    qed  = float(np.clip(d.prod(axis=1)[0]**(1/6), 0.01, 0.99))
    return {'logp': logp, 'hba': hba, 'hbd': hbd,
            'psa': psa, 'rtb': rtb, 'qed': qed}


def drug_score(mw, logp, qed, rtb, hbd) -> float:
    return float(np.clip(
        0.30*np.clip(1-mw/500, 0, 1) +
        0.25*np.clip(1-abs(logp-2.5)/4, 0, 1) +
        0.20*qed +
        0.15*np.clip(1-rtb/10, 0, 1) +
        0.10*np.clip(1-hbd/5,  0, 1), 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def enrich(input_path: str, output_path: str, checkpoint_path: str,
           mw_min: float, mw_max: float, save_every: int):

    input_path      = Path(input_path)
    output_path     = Path(output_path)
    checkpoint_path = Path(checkpoint_path)

    # ── Load and filter CSV ───────────────────────────────────────────────────
    log.info('Reading %s …', input_path)
    raw = pd.read_csv(input_path, low_memory=False)
    log.info('Raw rows: %d', len(raw))

    df = raw[
        (raw['type'] == 'small molecule') &
        (raw['average-mass'].notna()) &
        (raw['average-mass'].between(mw_min, mw_max))
    ].copy().reset_index(drop=True)
    log.info('Filtered to %d small molecules (%.0f–%.0f Da)', len(df), mw_min, mw_max)

    # ── Resume from checkpoint if it exists ───────────────────────────────────
    already_done: dict[str, dict] = {}
    if checkpoint_path.exists():
        ck = pd.read_csv(checkpoint_path)
        for _, row in ck.iterrows():
            already_done[row['drugbank_id']] = row.to_dict()
        log.info('Resumed from checkpoint: %d already done', len(already_done))

    # ── Output accumulator ────────────────────────────────────────────────────
    results: list[dict] = list(already_done.values())
    todo    = df[~df['drugbank-id'].isin(already_done)].copy()
    log.info('Remaining to process: %d', len(todo))

    # ── Batch loop ────────────────────────────────────────────────────────────
    total_batches = math.ceil(len(todo) / BATCH_SIZE)
    found_smiles  = sum(1 for r in results if r.get('smiles'))
    not_found     = len(results) - found_smiles

    for batch_idx in range(total_batches):
        batch = todo.iloc[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]

        # ── Pass 1: try CAS numbers ───────────────────────────────────────────
        cas_identifiers = {row['cas-number']: row for _, row in batch.iterrows()
                           if pd.notna(row.get('cas-number', None))
                           and str(row.get('cas-number', '')).strip()}
        cas_results = {}
        if cas_identifiers:
            cas_results = lookup_batch(list(cas_identifiers.keys()))

        # ── Pass 2: name fallback for CAS misses ─────────────────────────────
        cas_failed   = [r for _, r in batch.iterrows()
                        if cas_results.get(str(r.get('cas-number', ''))) is None]
        name_results = {}
        if cas_failed:
            name_map = {str(r['name']): r for r in cas_failed}
            name_results = lookup_batch(list(name_map.keys()))

        # ── Assemble results for this batch ───────────────────────────────────
        for _, row in batch.iterrows():
            db_id  = row['drugbank-id']
            mw_val = float(row['average-mass'])
            cas    = str(row.get('cas-number', '') or '')
            name   = str(row.get('name', '') or '')

            # Choose best PubChem result
            pc_data = None; src = 'not_found'
            if cas and cas_results.get(cas):
                pc_data = cas_results[cas]; src = 'pubchem_cas'
            elif name_results.get(name):
                pc_data = name_results[name]; src = 'pubchem_name'

            smiles   = ''
            pc_cid   = None
            desc_src = 'estimated'
            desc     = estimate_descriptors(mw_val)    # fallback

            if pc_data:
                smiles = pc_data.get('IsomericSMILES', '') or ''
                pc_cid = pc_data.get('CID')

                # Try RDKit first (most accurate)
                rdk = rdkit_descriptors(smiles) if smiles else None
                if rdk:
                    desc     = {k: rdk[k] for k in ('logp','hba','hbd','psa','rtb','qed')}
                    desc_src = 'rdkit'
                elif smiles:
                    # Use PubChem values directly
                    try:
                        desc = {
                            'logp': float(pc_data.get('XLogP', desc['logp'])),
                            'hba':  int(pc_data.get('HBondAcceptorCount', desc['hba'])),
                            'hbd':  int(pc_data.get('HBondDonorCount', desc['hbd'])),
                            'psa':  float(pc_data.get('TPSA', desc['psa'])),
                            'rtb':  int(pc_data.get('RotatableBondCount', desc['rtb'])),
                            'qed':  desc['qed'],   # QED needs SMILES → RDKit
                        }
                        desc_src = 'pubchem'
                    except Exception:
                        pass

            record = {
                'drugbank_id':   db_id,
                'name':          name,
                'drug_groups':   str(row.get('groups', '') or ''),
                'cas_number':    cas,
                'pubchem_cid':   pc_cid,
                'smiles':        smiles,
                'smiles_source': src,
                'mw':            mw_val,
                'logp':          round(desc['logp'], 4),
                'hba':           desc['hba'],
                'hbd':           desc['hbd'],
                'psa':           round(desc['psa'], 2),
                'rtb':           desc['rtb'],
                'qed':           round(desc['qed'], 4),
                'drug_score':    round(drug_score(mw_val, desc['logp'],
                                                  desc['qed'], desc['rtb'],
                                                  desc['hbd']), 4),
                'is_approved':   'approved' in str(row.get('groups', '') or '').lower(),
                'descriptor_src': desc_src,
                'state':         str(row.get('state', '') or ''),
            }
            results.append(record)

            if smiles:
                found_smiles += 1
            else:
                not_found += 1

        # ── Progress ──────────────────────────────────────────────────────────
        done_so_far = len(results)
        pct = 100*done_so_far/(len(df))
        log.info('Batch %3d/%d | done=%d/%d (%.1f%%) | SMILES found=%d | missing=%d',
                 batch_idx+1, total_batches, done_so_far, len(df),
                 pct, found_smiles, not_found)

        # ── Checkpoint save ───────────────────────────────────────────────────
        if done_so_far % save_every < BATCH_SIZE:
            ck_df = pd.DataFrame(results)
            ck_df.to_csv(checkpoint_path, index=False)
            log.info('  → Checkpoint saved (%d rows)', len(ck_df))

    # ── Final save ────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    smiles_count = (out_df['smiles'] != '').sum()
    rdkit_count  = (out_df['descriptor_src'] == 'rdkit').sum()
    pc_count     = (out_df['descriptor_src'] == 'pubchem').sum()
    est_count    = (out_df['descriptor_src'] == 'estimated').sum()

    log.info('='*60)
    log.info('ENRICHMENT COMPLETE')
    log.info('='*60)
    log.info('  Total rows          : %d', len(out_df))
    log.info('  SMILES found        : %d (%.1f%%)',
             smiles_count, 100*smiles_count/len(out_df))
    log.info('  Descriptors source:')
    log.info('    RDKit (from SMILES): %d', rdkit_count)
    log.info('    PubChem (no RDKit) : %d', pc_count)
    log.info('    Estimated (MW-cond): %d', est_count)
    log.info('  Output              : %s', output_path.resolve())

    # Print SMILES coverage by approval status
    appr = out_df[out_df['is_approved']]
    log.info('\n  Approved drugs: %d total, %d with SMILES (%.1f%%)',
             len(appr),
             (appr['smiles']!='').sum(),
             100*(appr['smiles']!='').mean())

    return out_df


# ══════════════════════════════════════════════════════════════════════════════
# COLAB HELPER
# ══════════════════════════════════════════════════════════════════════════════

def colab_setup():
    """
    Run this in a Colab cell before calling enrich().
    Installs dependencies and mounts Google Drive if you want to save output there.
    """
    import subprocess
    packages = ['rdkit', 'requests', 'tqdm', 'pandas', 'numpy']
    for pkg in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
    log.info('Dependencies installed.')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Enrich drugbank_clean.csv with SMILES + RDKit descriptors via PubChem.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Basic usage
  python enrich_smiles.py --input drugbank_clean.csv --output drugbank_enriched.csv

  # Custom MW filter and larger batches
  python enrich_smiles.py --input drugbank_clean.csv --output out.csv \\
      --mw-min 150 --mw-max 800 --save-every 500

  # In Google Colab (paste into a cell):
  #   !python enrich_smiles.py --input /content/drugbank_clean.csv \\
  #       --output /content/drugbank_enriched.csv

  # Resume after interruption (checkpoint is read automatically):
  python enrich_smiles.py --input drugbank_clean.csv --output drugbank_enriched.csv
        """)

    p.add_argument('--input',      required=True,
                   help='Path to drugbank_clean.csv')
    p.add_argument('--output',     default='drugbank_enriched.csv',
                   help='Output CSV with SMILES + descriptors (default: drugbank_enriched.csv)')
    p.add_argument('--checkpoint', default='smiles_checkpoint.csv',
                   help='Checkpoint file for resuming interrupted runs')
    p.add_argument('--mw-min',     type=float, default=100.0,
                   help='Minimum MW filter in Da (default: 100)')
    p.add_argument('--mw-max',     type=float, default=900.0,
                   help='Maximum MW filter in Da (default: 900)')
    p.add_argument('--save-every', type=int,   default=200,
                   help='Save checkpoint every N molecules (default: 200)')

    args = p.parse_args()

    enrich(
        input_path      = args.input,
        output_path     = args.output,
        checkpoint_path = args.checkpoint,
        mw_min          = args.mw_min,
        mw_max          = args.mw_max,
        save_every      = args.save_every,
    )
