import pandas as pd
import numpy as np
from pathlib import Path
import pyxlsb
import time
from datetime import datetime


def load_data(
    used_market_path,
    portfolio_path,
    used_market_csv_cache=None
):
    """
    Charge used_market (xlsb avec cache CSV) et portfolio (xlsx).

    Params
    ------
    used_market_path    : Path ou str vers le fichier .xlsb
    portfolio_path      : Path ou str vers le fichier .xlsx
    used_market_csv_cache : Path ou str vers le CSV cache (optionnel).
                           Si None, placé à côté du xlsb avec le même nom.

    Returns
    -------
    mkt : pd.DataFrame — données marché occasion
    pf  : pd.DataFrame — portefeuille à pricer
    """
    import pandas as pd
    from pathlib import Path

    used_market_path = Path(used_market_path)
    portfolio_path = Path(portfolio_path)

    if used_market_csv_cache is None:
        used_market_csv_cache = used_market_path.with_suffix('.csv')
    used_market_csv_cache = Path(used_market_csv_cache)

    # ── used_market ───────────────────────────────────────────────────────────
    if used_market_csv_cache.exists():
        print(f'[used_market] Lecture cache CSV : {used_market_csv_cache}')
        mkt = pd.read_csv(used_market_csv_cache, low_memory=False)
    else:
        print('[used_market] Lecture xlsb (peut prendre ~1 min)...')
        import pyxlsb
        mkt = pd.read_excel(used_market_path, engine='pyxlsb')
        mkt.to_csv(used_market_csv_cache, index=False)
        print(f'[used_market] Cache CSV sauvegardé : {used_market_csv_cache}')

    print(f'[used_market] {mkt.shape[0]:,} lignes x {mkt.shape[1]} colonnes')

    # ── portfolio ─────────────────────────────────────────────────────────────
    pf = pd.read_excel(portfolio_path)
    print(f'[portfolio]   {pf.shape[0]:,} lignes x {pf.shape[1]} colonnes')

    return mkt, pf


# Nettoyage de used_market

def clean_used_market(mkt: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et normalise le DataFrame used_market.

    Paramètres
    ----------
    mkt : pd.DataFrame
        DataFrame brut du marché occasion.

    Retourne
    --------
    pd.DataFrame
        DataFrame nettoyé avec colonnes renommées, types corrigés
        et lignes sans prix de vente supprimées.
    """
    RENAME_MKT = {
        'brand'                     : 'brand',
        'MODEL'                     : 'model',
        'FUEL TYPE'                 : 'fuel_type',
        'RANGE TYPE'                : 'range_type',
        'ENGINE Power (HP)'         : 'engine_hp',
        'gearbox'                   : 'gearbox',
        'BODY TYPE'                 : 'body_type',
        'MODEL SEGMENT'             : 'model_segment',
        'mileage'                   : 'mileage',
        'age'                       : 'age_months',
        'prix de vente'             : 'sale_price',
        "prix catalogue d'origine"  : 'list_price',
        'date de vente'             : 'sale_date',
        'productionYear'            : 'production_year',
        'modelyear'                 : 'model_year',
    }

    # Renommage des colonnes existantes
    existing_rename = {k: v for k, v in RENAME_MKT.items() if k in mkt.columns}
    mkt = mkt.rename(columns=existing_rename)

    # Conversion de la date Excel (serial date) en datetime
    if 'sale_date' in mkt.columns:
        mkt['sale_date'] = pd.to_datetime(
            mkt['sale_date'].astype(float), unit='D', origin='1899-12-30'
        )

    # Conversion des colonnes numériques
    numeric_cols = ['mileage', 'age_months', 'sale_price', 'list_price', 'engine_hp']
    for col in numeric_cols:
        if col in mkt.columns:
            mkt[col] = pd.to_numeric(mkt[col], errors='coerce')

    # Suppression des lignes sans prix de vente
    if 'sale_price' in mkt.columns:
        n_before = len(mkt)
        mkt = mkt.dropna(subset=['sale_price'])
        print(f'Lignes supprimées (sale_price NaN) : {n_before - len(mkt)}')

    print(f'used_market après nettoyage : {mkt.shape}')
    display(mkt.dtypes.to_frame('dtype').T)

    return mkt


# Nettoyage de portfolio
def clean_portfolio(pf: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et normalise le DataFrame portfolio.

    Paramètres
    ----------
    pf : pd.DataFrame
        DataFrame brut du portfolio.

    Retourne
    --------
    pd.DataFrame
        DataFrame nettoyé avec colonnes renommées et types corrigés.
    """
    RENAME_PF = {
        'model'                             : 'model',
        'fuel_type'                         : 'fuel_type',
        'range_type'                        : 'range_type',
        'production_year'                   : 'production_year',
        'current_contract_planned_end_date' : 'end_date',
        'contract_mileage'                  : 'contract_mileage',
        "prix catalogue d'origine"          : 'list_price',
        'contract_duration'                 : 'contract_duration',
        'remaining_contract_duration'       : 'remaining_duration',
        'initial_car_age'                   : 'initial_age',
        'initial_mileage'                   : 'initial_mileage',
    }

    # Renommage des colonnes existantes
    existing_pf = {k: v for k, v in RENAME_PF.items() if k in pf.columns}
    pf = pf.rename(columns=existing_pf)

    # Conversion de la date de fin de contrat
    if 'end_date' in pf.columns:
        pf['end_date'] = pd.to_datetime(pf['end_date'], errors='coerce')

    # Conversion des colonnes numériques
    numeric_cols = [
        'contract_mileage', 'list_price', 'contract_duration',
        'remaining_duration', 'initial_age', 'initial_mileage',
    ]
    for col in numeric_cols:
        if col in pf.columns:
            pf[col] = pd.to_numeric(pf[col], errors='coerce')

    print(f'portfolio après nettoyage : {pf.shape}')
    display(pf.dtypes.to_frame('dtype').T)

    return pf


def prepare_features(
    mkt: pd.DataFrame,
    pf_enriched: pd.DataFrame,
    num_features: list,
    cat_features: list,
    all_features: list,
    p_low: float = 0.005,
    p_high: float = 0.995,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtre les outliers sur la cible, impute les NaN numériques et catégorielles
    dans used_market et portfolio.

    Paramètres
    ----------
    mkt          : DataFrame used_market nettoyé.
    pf_enriched  : DataFrame portfolio enrichi.
    num_features : Liste des features numériques.
    cat_features : Liste des features catégorielles.
    all_features : Liste complète des features (num + cat).
    p_low        : Quantile bas pour le filtre outliers (défaut 0.5%).
    p_high       : Quantile haut pour le filtre outliers (défaut 99.5%).

    Retourne
    --------
    mkt_train   : DataFrame used_market prêt pour l'entraînement.
    pf_enriched : DataFrame portfolio imputé.
    """
    # Filtre outliers extrêmes sur la cible
    q_low  = mkt['sale_price'].quantile(p_low)
    q_high = mkt['sale_price'].quantile(p_high)
    mkt_train = mkt[(mkt['sale_price'] >= q_low) & (mkt['sale_price'] <= q_high)].copy()
    print(f'Lignes après filtre outliers : {len(mkt_train):,}  (retiré {len(mkt) - len(mkt_train):,})')

    # Imputation NaN numériques par médiane par fuel_type (used_market)
    for col in num_features:
        if col in mkt_train.columns:
            n_nan = mkt_train[col].isna().sum()
            if n_nan > 0:
                fill = mkt_train.groupby('fuel_type')[col].transform('median')
                mkt_train[col] = mkt_train[col].fillna(fill).fillna(mkt_train[col].median())
                print(f'  Imputation {col}: {n_nan} NaN')

    # Catégorielles : remplace NaN par 'UNKNOWN' (used_market)
    for col in cat_features:
        if col in mkt_train.columns:
            mkt_train[col] = mkt_train[col].fillna('UNKNOWN')

    # Même traitement pour le portfolio
    for col in num_features:
        if col in pf_enriched.columns:
            fill = pf_enriched.groupby('fuel_type')[col].transform('median')
            pf_enriched[col] = pf_enriched[col].fillna(fill).fillna(mkt_train[col].median())
    for col in cat_features:
        if col in pf_enriched.columns:
            pf_enriched[col] = pf_enriched[col].fillna('UNKNOWN')

    # Vérification NaN résiduels
    nan_check = mkt_train[all_features].isna().sum()
    remaining = nan_check[nan_check > 0]
    if len(remaining):
        print('\nNaN résiduels :')
        display(remaining.to_frame('nb_NaN'))
    else:
        print('\nAucun NaN résiduel dans mkt_train.')

    return mkt_train, pf_enriched