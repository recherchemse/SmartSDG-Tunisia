import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split as sk_split
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR    = '/kaggle/input/datasets/fedyou/smartsdg/SmartSDG-Tunisia/Brut_Data'
CDI_PATH    = '/kaggle/input/datasets/fedyou/smartsdg/SmartSDG-Tunisia/CDI'

NUM_TILES   = 90

# Hyper-paramètres
SEQUENCE_LENGTH = 3

# N-HiTS specific hyper-parameters
# Stacks hiérarchiques : chaque stack opère à une résolution temporelle différente
NHITS_STACKS        = 3          # Nombre de stacks (niveaux hiérarchiques)
NHITS_BLOCKS        = 1          # Blocs par stack
NHITS_HIDDEN_SIZE   = 256        # Taille des couches MLP internes
NHITS_NUM_LAYERS    = 2          # Couches MLP par bloc
# Pooling sizes : chaque stack sous-échantillonne l'entrée à une résolution différente
# Ex : [1, 2, 3] → stack 0 voit toute la séquence, stack 1 la sous-échantillonne par 2, etc.
NHITS_POOL_SIZES    = [1, 2, 3]  # doit avoir len = NHITS_STACKS
# Nombre de coefficients d'interpolation par stack (expressivité de la reconstruction)
NHITS_N_FREQ_DOWN   = [1, 2, 3]  # doit avoir len = NHITS_STACKS

DROPOUT         = 0.3
EPOCHS          = 20
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
PATIENCE        = 10
RANDOM_SEED     = 42

PATCH_SIZE      = 8

# Split temporel
TRAIN_RATIO = 0.64
VAL_RATIO   = 0.30

DURATION_MONTHS = 311

torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


# ============================================================
# UTILITAIRES DATES
# ============================================================
def get_last_n_months(n_months=60):
    end_date   = datetime(2025, 12, 1)
    start_date = end_date - relativedelta(months=n_months - 1)
    dates, current = [], start_date
    while current <= end_date:
        dates.append(current)
        current += relativedelta(months=1)
    return sorted(dates)


# ============================================================
# CHARGEMENT DES FICHIERS BRUTS
# ============================================================
def _fmt(date):
    return date.strftime('%Y'), date.strftime('%m'), date.strftime('%Y_%m')


REF_SHAPE     = None
REF_TRANSFORM = None
REF_CRS       = None


def _resample_to_ref(src, ref_shape, ref_transform, ref_crs):
    from rasterio.warp import reproject, Resampling
    data_out = np.empty(ref_shape, dtype=np.float32)
    reproject(
        source         = rasterio.band(src, 1),
        destination    = data_out,
        src_transform  = src.transform,
        src_crs        = src.crs,
        dst_transform  = ref_transform,
        dst_crs        = ref_crs,
        resampling     = Resampling.bilinear
    )
    return data_out


def load_cdi_raster(date):
    _, _, ym = _fmt(date)
    path = os.path.join(CDI_PATH, f'CDI_Tunisia_{ym}.tif')
    if not os.path.exists(path):
        return None, None
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if REF_SHAPE is not None and arr.shape != REF_SHAPE:
                arr = _resample_to_ref(src, REF_SHAPE, REF_TRANSFORM, REF_CRS)
            return arr, src.transform
    except Exception:
        return None, None


def load_monthly_indicators(date):
    global REF_SHAPE, REF_TRANSFORM, REF_CRS
    _, _, ym = _fmt(date)

    paths = {
        'SPI':  os.path.join(BASE_DIR, 'SPI',  f'Tunisia_SPI_{ym}.tif'),
        'SPEI': os.path.join(BASE_DIR, 'SPEI', f'Tunisia_SPEI_{ym}.tif'),
        'SM':   os.path.join(BASE_DIR, 'SM',   f'GLDAS_SoilMoisture_Tunisia_{ym}.tif'),
        'NDVI': os.path.join(BASE_DIR, 'NDVI', f'NDVI_Tunisia_{ym}.tif'),
        'LST':  os.path.join(BASE_DIR, 'LST',  f'LST_MonthlyMean_{ym}.tif'),
    }

    indicators = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            return None, None, False
        try:
            with rasterio.open(path) as src:
                if REF_SHAPE is None:
                    REF_SHAPE     = (src.height, src.width)
                    REF_TRANSFORM = src.transform
                    REF_CRS       = src.crs
                arr = src.read(1).astype(np.float32)
                if arr.shape != REF_SHAPE:
                    arr = _resample_to_ref(src, REF_SHAPE, REF_TRANSFORM, REF_CRS)
                indicators[name] = arr
        except Exception:
            return None, None, False

    return indicators, REF_TRANSFORM, True


# ============================================================
# COLLECTE TEMPORELLE
# ============================================================
def collect_temporal_data(n_months=DURATION_MONTHS):
    print(f"\n📅 Collecte des {n_months} derniers mois...")
    all_dates = get_last_n_months(n_months)
    print(f"   Période : {all_dates[0]:%Y-%m} → {all_dates[-1]:%Y-%m}")

    temporal_data    = []
    raster_transform = None

    for date in tqdm(all_dates, desc="Chargement"):
        indicators, transform, ok = load_monthly_indicators(date)
        if not ok:
            continue
        cdi_real, _ = load_cdi_raster(date)
        if cdi_real is None:
            continue

        temporal_data.append({'date': date, 'indicators': indicators, 'cdi_real': cdi_real})
        if raster_transform is None:
            raster_transform = transform

    print(f"   ✅ {len(temporal_data)} mois disponibles\n")
    return temporal_data, raster_transform


# ============================================================
# GÉNÉRATION DES TUILES
# ============================================================
def generate_tiles(raster_shape, num_tiles=NUM_TILES):
    H, W      = raster_shape
    tile_size = max(1, H // num_tiles)
    tiles     = []
    for r in range(0, H, tile_size):
        r_end = min(r + tile_size, H)
        tiles.append((r, r_end, 0, W))
    if len(tiles) > num_tiles:
        last_r0 = tiles[num_tiles - 1][0]
        tiles   = tiles[:num_tiles - 1] + [(last_r0, H, 0, W)]
    print(f"   Lignes par tuile : ~{tile_size}  |  Tuiles effectives : {len(tiles)}")
    return tiles


# ============================================================
# CLÉS DES INDICATEURS
# ============================================================
INDICATOR_KEYS = ['SPI', 'SPEI', 'SM', 'NDVI', 'LST']
NUM_FEATURES   = len(INDICATOR_KEYS)


# ============================================================
# SÉQUENCES SPATIO-TEMPORELLES POUR UNE TUILE (N-HiTS)
# Chaque patch est aplati en vecteur : T x (C * pH * pW)
# ============================================================
def prepare_sequences_tile_nhits(temporal_data, r0, r1, c0, c1, patch_size=PATCH_SIZE):
    temporal_data = sorted(temporal_data, key=lambda x: x['date'])
    num_months    = len(temporal_data)

    H_tile = r1 - r0
    W_tile = c1 - c0

    sequences, targets, seq_dates, patch_origins = [], [], [], []

    for target_idx in range(SEQUENCE_LENGTH, num_months):
        target_month    = temporal_data[target_idx]
        sequence_months = temporal_data[target_idx - SEQUENCE_LENGTH:target_idx]

        seq_stack = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES, H_tile, W_tile), dtype=np.float32)
        valid_global = True
        for t, md in enumerate(sequence_months):
            for c_idx, k in enumerate(INDICATOR_KEYS):
                patch_data = md['indicators'][k][r0:r1, c0:c1]
                if np.any(np.isnan(patch_data)) or np.any(~np.isfinite(patch_data)) or np.any(np.abs(patch_data) > 1e6):
                    valid_global = False
                    break
                seq_stack[t, c_idx] = patch_data
            if not valid_global:
                break
        if not valid_global:
            continue

        tgt_patch = target_month['cdi_real'][r0:r1, c0:c1].astype(np.int64)
        if np.any(np.isnan(target_month['cdi_real'][r0:r1, c0:c1])):
            continue
        tgt_patch = np.clip(tgt_patch, 0, 5)

        for pr in range(0, H_tile - patch_size + 1, patch_size):
            for pc in range(0, W_tile - patch_size + 1, patch_size):
                # seq : (T, C, pH, pW) → aplatir en (T, C*pH*pW) pour N-HiTS
                sub_seq = seq_stack[:, :, pr:pr+patch_size, pc:pc+patch_size]
                sub_seq_flat = sub_seq.reshape(SEQUENCE_LENGTH, -1)  # (T, C*pH*pW)
                sub_tgt = tgt_patch[pr:pr+patch_size, pc:pc+patch_size]

                sequences.append(sub_seq_flat)
                targets.append(sub_tgt)
                seq_dates.append(target_month['date'])
                patch_origins.append((r0 + pr, c0 + pc))

    if len(sequences) == 0:
        return None, None, None, None

    return (np.array(sequences,     dtype=np.float32),   # (N, T, C*pH*pW)
            np.array(targets,       dtype=np.int64),      # (N, pH, pW)
            np.array(seq_dates),
            np.array(patch_origins))


# ============================================================
# SPLIT TEMPOREL ORDONNÉ
# ============================================================
def split_data_ordered(sequences, targets, seq_dates, patch_origins):
    unique_dates = np.array(sorted(set(seq_dates)))
    n_dates      = len(unique_dates)
    n_test_dates = int(n_dates * (1 - TRAIN_RATIO - VAL_RATIO))
    if n_test_dates == 0:
        n_test_dates = 1
    split_date = unique_dates[-n_test_dates]

    test_mask = seq_dates >= split_date
    tv_mask   = seq_dates <  split_date

    X_tv, y_tv  = sequences[tv_mask],   targets[tv_mask]
    X_te, y_te  = sequences[test_mask], targets[test_mask]
    origins_te  = patch_origins[test_mask]
    dates_te    = seq_dates[test_mask]

    if len(X_tv) == 0 or len(X_te) == 0:
        return None

    val_ratio_in_tv = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    tr_idx, va_idx  = sk_split(
        np.arange(len(X_tv)),
        test_size=val_ratio_in_tv,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    return (X_tv[tr_idx], X_tv[va_idx], X_te,
            y_tv[tr_idx], y_tv[va_idx], y_te,
            origins_te,
            dates_te)


# ============================================================
# NORMALISATION (par canal, sur l'ensemble train)
# ============================================================
def normalize_features_nhits(X_train, X_val, X_test):
    # X : (N, T, F)  où F = C*pH*pW
    N, T, F = X_train.shape
    scaler   = RobustScaler()
    X_tr_s   = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N, T, F)
    nv        = X_val.shape[0]
    X_va_s   = scaler.transform(X_val.reshape(-1, F)).reshape(nv, T, F)
    nte       = X_test.shape[0]
    X_te_s   = scaler.transform(X_test.reshape(-1, F)).reshape(nte, T, F)
    return X_tr_s, X_va_s, X_te_s


# ============================================================
# DATASET
# ============================================================
class CDIDatasetNHiTS(Dataset):
    def __init__(self, X, y):
        # N-HiTS attend (N, T, F) — même format que GRU
        self.X = torch.FloatTensor(X)   # (N, T, F)
        self.y = torch.LongTensor(y)    # (N, pH, pW)

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ============================================================
# ARCHITECTURE N-HiTS
# Reference: Challu et al. (2023), "N-HiTS: Neural Hierarchical
# Interpolation for Time Series Forecasting", AAAI 2023.
#
# Principe :
#   • Plusieurs stacks empilés en séquence (doubly residual stacking).
#   • Chaque stack opère sur une résolution temporelle différente
#     grâce à un MaxPool1d de taille pool_size (multi-rate sampling).
#   • Chaque bloc produit deux projections linéaires :
#       – backcast  : reconstruit la portion de l'input expliquée
#       – forecast  : prédit la sortie à sa résolution propre
#   • Les backcasts sont soustraits de l'entrée (résidu) avant d'être
#     passés au stack suivant.
#   • Les forecasts de tous les stacks sont interpolés (upsampling
#     linéaire) puis sommés pour former la prédiction finale.
#
# Adaptation pour la classification :
#   La sortie finale (somme des forecasts interpolés) est projetée
#   vers num_classes * pH * pW via une tête linéaire.
# ============================================================

class NHiTSBlock(nn.Module):
    """
    Bloc élémentaire N-HiTS.

    Args:
        input_size   : T_pooled * F  (séquence sous-échantillonnée aplatie)
        hidden_size  : taille des couches MLP internes
        num_layers   : profondeur du MLP interne
        backcast_len : T  (longueur originale de l'input)
        forecast_len : n_freq_down  (nb de coefficients de forecast)
        dropout      : taux de dropout
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 backcast_len, forecast_len, dropout=0.0):
        super().__init__()
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len

        # MLP partagé (θ_b et θ_f partagent le même tronc)
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            in_dim  = hidden_size
        self.mlp = nn.Sequential(*layers)

        # Têtes de projection séparées
        self.fc_backcast = nn.Linear(hidden_size, backcast_len)
        self.fc_forecast = nn.Linear(hidden_size, forecast_len)

    def forward(self, x):
        # x : (B, input_size)
        h          = self.mlp(x)                          # (B, hidden_size)
        backcast   = self.fc_backcast(h)                  # (B, T)
        forecast   = self.fc_forecast(h)                  # (B, n_freq_down)
        return backcast, forecast


class NHiTSStack(nn.Module):
    """
    Stack N-HiTS : agrège num_blocks blocs séquentiels avec résidus.

    Args:
        seq_len      : longueur de la séquence d'entrée (T)
        feature_size : dimension des features (F)
        pool_size    : facteur de sous-échantillonnage MaxPool
        n_freq_down  : nb de coefficients de forecast produits
        hidden_size  : taille interne du MLP
        num_layers   : profondeur MLP
        num_blocks   : nb de blocs dans ce stack
        dropout      : taux de dropout
    """
    def __init__(self, seq_len, feature_size, pool_size, n_freq_down,
                 hidden_size, num_layers, num_blocks, dropout=0.0):
        super().__init__()
        self.pool_size    = pool_size
        self.n_freq_down  = n_freq_down
        self.seq_len      = seq_len
        self.feature_size = feature_size

        # Longueur après pooling — calculée dynamiquement via un forward test
        # pour éviter tout mismatch dû au ceil_mode ou à la division entière
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size,
                                 ceil_mode=True)
        with torch.no_grad():
            dummy        = torch.zeros(1, feature_size, seq_len)
            pooled_out   = self.pool(dummy)             # (1, feature_size, T_pooled)
            self.pooled_len = pooled_out.shape[-1]

        input_size = self.pooled_len * feature_size

        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size   = input_size,
                hidden_size  = hidden_size,
                num_layers   = num_layers,
                backcast_len = seq_len,        # le backcast couvre l'input complet
                forecast_len = n_freq_down,
                dropout      = dropout
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x : (B, T, F)  — résidu de l'entrée courant
        Returns:
            residual : (B, T, F)  — entrée mise à jour (x - backcast)
            forecast : (B, n_freq_down)  — prédiction à basse résolution
        """
        B, T, F = x.shape

        # 1. Sous-échantillonnage multi-rate par MaxPool sur la dim temporelle
        #    pool attend (B, C, L) → on traite F comme C
        x_pooled = self.pool(x.permute(0, 2, 1))   # (B, F, T_pooled)
        x_flat   = x_pooled.permute(0, 2, 1).reshape(B, -1)  # (B, T_pooled * F)

        # 2. Accumulation des backcasts et forecasts sur les blocs
        stack_forecast = torch.zeros(B, self.n_freq_down, device=x.device)
        residual       = x  # résidu courant dans ce stack

        for block in self.blocks:
            # Le bloc reçoit l'entrée poolée (figée), pas le résidu mis à jour
            # (comme dans l'implémentation officielle N-BEATS / N-HiTS)
            backcast, forecast = block(x_flat)       # (B, T), (B, n_freq_down)

            # Soustraction du backcast sur la dim temporelle
            # backcast : (B, T) → (B, T, 1) broadcast sur F
            residual = residual - backcast.unsqueeze(-1).expand_as(residual) / F

            stack_forecast = stack_forecast + forecast

        return residual, stack_forecast


class CDINHiTSClassifier(nn.Module):
    """
    Modèle N-HiTS adapté à la classification de patches CDI.

    Architecture :
        • NHITS_STACKS stacks en doubly residual stacking
        • Chaque stack opère à une résolution temporelle différente
          (pool_sizes croissants → vision de plus en plus globale)
        • Les forecasts (basse résolution) de chaque stack sont
          interpolés (upsampling linéaire) vers une dimension commune
          puis sommés → embedding de sortie
        • Une tête MLP projette cet embedding vers
          num_classes * patch_size * patch_size

    Args:
        seq_len      : T = SEQUENCE_LENGTH
        feature_size : F = NUM_FEATURES * PATCH_SIZE * PATCH_SIZE
        stacks       : nombre de stacks hiérarchiques
        blocks       : blocs par stack
        pool_sizes   : liste de pool_size par stack
        n_freq_downs : liste de n_freq_down par stack
        hidden_size  : taille interne MLP
        num_layers   : couches MLP par bloc
        num_classes  : 6 classes CDI
        dropout      : taux de dropout
        patch_size   : taille du patch spatial
    """
    def __init__(self, seq_len, feature_size,
                 stacks=NHITS_STACKS, blocks=NHITS_BLOCKS,
                 pool_sizes=None, n_freq_downs=None,
                 hidden_size=NHITS_HIDDEN_SIZE, num_layers=NHITS_NUM_LAYERS,
                 num_classes=6, dropout=DROPOUT, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size  = patch_size
        self.num_classes = num_classes
        self.seq_len     = seq_len

        if pool_sizes   is None: pool_sizes   = NHITS_POOL_SIZES
        if n_freq_downs is None: n_freq_downs = NHITS_N_FREQ_DOWN
        assert len(pool_sizes) == stacks == len(n_freq_downs), \
            "pool_sizes et n_freq_downs doivent avoir len = stacks"

        self.stacks_modules = nn.ModuleList([
            NHiTSStack(
                seq_len      = seq_len,
                feature_size = feature_size,
                pool_size    = pool_sizes[i],
                n_freq_down  = n_freq_downs[i],
                hidden_size  = hidden_size,
                num_layers   = num_layers,
                num_blocks   = blocks,
                dropout      = dropout
            )
            for i in range(stacks)
        ])

        # Dimension totale après concaténation des forecasts interpolés
        # Chaque stack produit n_freq_down coefficients interpolés → seq_len
        # On concatène les contributions : stacks * seq_len
        total_forecast_dim = stacks * seq_len

        self.dropout_layer = nn.Dropout(dropout)

        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(total_forecast_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes * patch_size * patch_size)
        )

    def forward(self, x):
        """
        Args:
            x : (B, T, F)
        Returns:
            logits : (B, num_classes, pH, pW)
        """
        B, T, F = x.shape
        residual         = x
        all_forecasts    = []

        for stack in self.stacks_modules:
            residual, forecast = stack(residual)   # forecast : (B, n_freq_down)

            # Interpolation linéaire : upsampling du forecast vers la longueur T
            # pour obtenir une contribution à résolution complète
            forecast_up = F_interp(forecast, size=T)   # (B, T)
            all_forecasts.append(forecast_up)

        # Concaténation de toutes les contributions interpolées
        combined = torch.cat(all_forecasts, dim=-1)  # (B, stacks * T)
        combined = self.dropout_layer(combined)

        logits = self.classifier(combined)            # (B, num_classes * pH * pW)
        logits = logits.view(B, self.num_classes, self.patch_size, self.patch_size)
        return logits


def F_interp(x, size):
    """
    Interpolation linéaire 1D d'un tenseur (B, L) vers (B, size).
    Utilise F.interpolate qui attend (B, C, L).
    """
    return F.interpolate(
        x.unsqueeze(1),          # (B, 1, L)
        size=size,
        mode='linear',
        align_corners=True
    ).squeeze(1)                 # (B, size)


# ============================================================
# ENTRAÎNEMENT
# ============================================================
def train_model(train_loader, val_loader, model, criterion, optimizer):
    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(EPOCHS):
        model.train()
        for bX, by in train_loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bX)
            loss   = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for bX, by in val_loader:
                logits = model(bX.to(DEVICE))
                pred   = logits.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy().flatten())
                val_true.extend(by.numpy().flatten())

        val_acc = accuracy_score(val_true, val_preds)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tile.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        if epoch % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return best_val_acc


# ============================================================
# TRAITEMENT D'UNE TUILE
# ============================================================
CLASS_NAMES = ['Normal', 'Légère', 'Modérée', 'Sévère', 'Extrême', 'Except.']


def process_tile(temporal_data, tile_idx, r0, r1, c0, c1):
    seqs, tgts, dates, origins = prepare_sequences_tile_nhits(
        temporal_data, r0, r1, c0, c1, patch_size=PATCH_SIZE)

    if seqs is None or len(seqs) < 10:
        return None

    split = split_data_ordered(seqs, tgts, dates, origins)
    if split is None:
        return None

    X_tr, X_va, X_te, y_tr, y_va, y_te, origins_te, dates_te = split
    del seqs; gc.collect()

    X_tr_s, X_va_s, X_te_s = normalize_features_nhits(X_tr, X_va, X_te)
    del X_tr, X_va, X_te; gc.collect()

    feature_size = X_tr_s.shape[2]  # C * pH * pW

    tr_loader = DataLoader(CDIDatasetNHiTS(X_tr_s, y_tr), BATCH_SIZE, shuffle=False, pin_memory=True)
    va_loader = DataLoader(CDIDatasetNHiTS(X_va_s, y_va), BATCH_SIZE, shuffle=False, pin_memory=True)
    te_loader = DataLoader(CDIDatasetNHiTS(X_te_s, y_te), BATCH_SIZE, shuffle=False, pin_memory=True)
    del X_tr_s, X_va_s, X_te_s; gc.collect()

    model = CDINHiTSClassifier(
        seq_len      = SEQUENCE_LENGTH,
        feature_size = feature_size,
        stacks       = NHITS_STACKS,
        blocks       = NHITS_BLOCKS,
        pool_sizes   = NHITS_POOL_SIZES,
        n_freq_downs = NHITS_N_FREQ_DOWN,
        hidden_size  = NHITS_HIDDEN_SIZE,
        num_layers   = NHITS_NUM_LAYERS,
        num_classes  = 6,
        dropout      = DROPOUT,
        patch_size   = PATCH_SIZE
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = train_model(tr_loader, va_loader, model, criterion, optimizer)

    model.load_state_dict(torch.load('best_tile.pth'))
    model.eval()
    preds_flat, trues_flat = [], []

    with torch.no_grad():
        for bX, by in te_loader:
            logits = model(bX.to(DEVICE))
            pred   = logits.argmax(dim=1)
            preds_flat.extend(pred.cpu().numpy().flatten())
            trues_flat.extend(by.numpy().flatten())

    preds_flat = np.array(preds_flat)
    trues_flat = np.array(trues_flat)

    acc  = accuracy_score(trues_flat, preds_flat)
    kap  = cohen_kappa_score(trues_flat, preds_flat)
    f1m  = f1_score(trues_flat, preds_flat, average='macro',    zero_division=0)
    f1w  = f1_score(trues_flat, preds_flat, average='weighted', zero_division=0)

    print(f"   🔲 Tuile {tile_idx:03d} [{r0}:{r1}, {c0}:{c1}] "
          f"| {len(trues_flat):,} pixels test "
          f"| Acc={acc:.4f}  Kappa={kap:.4f}  F1m={f1m:.4f}  F1w={f1w:.4f}")

    pred_maps_by_date = {}
    true_maps_by_date = {}

    H_tile = r1 - r0
    W_tile = c1 - c0

    patch_counter = 0
    model.eval()
    with torch.no_grad():
        for bX, by in te_loader:
            logits     = model(bX.to(DEVICE))
            batch_pred = logits.argmax(dim=1).cpu().numpy()
            batch_true = by.numpy()
            B = batch_pred.shape[0]
            for b in range(B):
                global_idx = patch_counter + b
                if global_idx >= len(origins_te):
                    break
                r_orig, c_orig = origins_te[global_idx]
                date_key       = dates_te[global_idx]

                if date_key not in pred_maps_by_date:
                    pred_maps_by_date[date_key] = np.full((H_tile, W_tile), np.nan, dtype=np.float32)
                    true_maps_by_date[date_key] = np.full((H_tile, W_tile), np.nan, dtype=np.float32)

                pr_local = r_orig - r0
                pc_local = c_orig - c0
                pred_maps_by_date[date_key][pr_local:pr_local+PATCH_SIZE,
                                            pc_local:pc_local+PATCH_SIZE] = batch_pred[b]
                true_maps_by_date[date_key][pr_local:pr_local+PATCH_SIZE,
                                            pc_local:pc_local+PATCH_SIZE] = batch_true[b]
            patch_counter += B

    del model, tr_loader, va_loader, te_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'tile_idx':           tile_idx,
        'bounds':             (r0, r1, c0, c1),
        'n_test':             len(trues_flat),
        'accuracy':           acc,
        'kappa':              kap,
        'f1_macro':           f1m,
        'f1_weighted':        f1w,
        'val_acc':            best_val_acc,
        'pred_maps_by_date':  pred_maps_by_date,
        'true_maps_by_date':  true_maps_by_date,
    }


# ============================================================
# RÉSUMÉ GLOBAL
# ============================================================
def print_global_summary(results, full_shape, output_dir='.'):
    if not results:
        print("Aucun résultat disponible.")
        return

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('pred_maps_by_date', 'true_maps_by_date')}
                       for r in results])
    total_samples = df['n_test'].sum()

    mean_acc = df['accuracy'].mean()
    mean_kap = df['kappa'].mean()
    mean_f1m = df['f1_macro'].mean()
    mean_f1w = df['f1_weighted'].mean()

    w    = df['n_test'].values / total_samples
    wacc = np.average(df['accuracy'].values,    weights=w)
    wkap = np.average(df['kappa'].values,       weights=w)
    wf1m = np.average(df['f1_macro'].values,    weights=w)
    wf1w = np.average(df['f1_weighted'].values, weights=w)

    print("\n" + "="*70)
    print(f"RESUME GLOBAL (N-HiTS) — {len(df)} tuile(s)  |  {total_samples:,} pixels test")
    print("="*70)
    print(f"  Métrique               Moyenne simple    Moyenne pondérée")
    print(f"  {'-'*58}")
    print(f"  Accuracy               {mean_acc:>16.4f}  {wacc:>18.4f}")
    print(f"  Cohen Kappa            {mean_kap:>16.4f}  {wkap:>18.4f}")
    print(f"  F1 Macro               {mean_f1m:>16.4f}  {wf1m:>18.4f}")
    print(f"  F1 Weighted            {mean_f1w:>16.4f}  {wf1w:>18.4f}")
    print("="*70)

    df.to_csv(os.path.join(output_dir, 'tile_results_nhits.csv'), index=False)
    print("\nRésultats détaillés → tile_results_nhits.csv")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    OUTPUT_DIR = '/kaggle/working/cdi_rasters_nhits'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("🚀 CDI PREDICTION — N-HiTS (raster Tunisie complet)")
    print(f"   Nombre de tuiles souhaité : {NUM_TILES}")
    print(f"   Taille des patchs         : {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"   Stacks N-HiTS             : {NHITS_STACKS}")
    print(f"   Blocs par stack           : {NHITS_BLOCKS}")
    print(f"   Pool sizes (par stack)    : {NHITS_POOL_SIZES}")
    print(f"   N-freq-down (par stack)   : {NHITS_N_FREQ_DOWN}")
    print(f"   Hidden size MLP           : {NHITS_HIDDEN_SIZE}")
    print(f"   Couches MLP par bloc      : {NHITS_NUM_LAYERS}")
    print("="*70)

    temporal_data, raster_transform = collect_temporal_data(DURATION_MONTHS)

    full_shape = list(temporal_data[0]['indicators'].values())[0].shape
    print(f"\n   Raster shape : {full_shape}")

    tiles = generate_tiles(full_shape, NUM_TILES)
    print(f"   Nombre de tuiles : {len(tiles)}")

    print("\n🔄 Traitement des tuiles (N-HiTS)...\n")
    all_results = []

    for tile_idx, (r0, r1, c0, c1) in enumerate(tiles):
        result = process_tile(temporal_data, tile_idx, r0, r1, c0, c1)
        if result is not None:
            all_results.append(result)

    print_global_summary(all_results, full_shape, output_dir=OUTPUT_DIR)

    print("\n✅ TERMINÉ")
