import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
import torch.nn as nn
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
SEQUENCE_LENGTH = 3       # lookback horizon (L in the TiDE paper)
HIDDEN_SIZE     = 256     # dimension of the dense encoder/decoder hidden layers
ENCODER_LAYERS  = 2       # number of residual blocks in encoder
DECODER_LAYERS  = 2       # number of residual blocks in decoder
TEMPORAL_DIM    = 64      # dimension of the temporal decoder projection
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
# SÉQUENCES SPATIO-TEMPORELLES POUR UNE TUILE (TiDE)
#
# TiDE ingère une fenêtre de L pas de temps (lookback) aplatie :
#   x_hist : (N, L * F_flat)   — covariables historiques encodées
#
# Ici F_flat = C * pH * pW (features spatiales aplaties par pas de temps).
# ============================================================
def prepare_sequences_tile_tide(temporal_data, r0, r1, c0, c1, patch_size=PATCH_SIZE):
    temporal_data = sorted(temporal_data, key=lambda x: x['date'])
    num_months    = len(temporal_data)

    H_tile = r1 - r0
    W_tile = c1 - c0

    sequences, targets, seq_dates, patch_origins = [], [], [], []

    for target_idx in range(SEQUENCE_LENGTH, num_months):
        target_month    = temporal_data[target_idx]
        sequence_months = temporal_data[target_idx - SEQUENCE_LENGTH:target_idx]

        # seq_stack : (L, C, pH, pW)
        seq_stack    = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES, H_tile, W_tile), dtype=np.float32)
        valid_global = True
        for t, md in enumerate(sequence_months):
            for c_idx, k in enumerate(INDICATOR_KEYS):
                patch_data = md['indicators'][k][r0:r1, c0:c1]
                if (np.any(np.isnan(patch_data))
                        or np.any(~np.isfinite(patch_data))
                        or np.any(np.abs(patch_data) > 1e6)):
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
                # Extraction du patch local
                sub_seq = seq_stack[:, :, pr:pr+patch_size, pc:pc+patch_size]
                # Aplatissage : (L, C*pH*pW)  puis flatten global → (L * C*pH*pW,)
                sub_seq_flat = sub_seq.reshape(SEQUENCE_LENGTH, -1)   # (L, F_flat)
                sub_tgt      = tgt_patch[pr:pr+patch_size, pc:pc+patch_size]

                sequences.append(sub_seq_flat)
                targets.append(sub_tgt)
                seq_dates.append(target_month['date'])
                patch_origins.append((r0 + pr, c0 + pc))

    if len(sequences) == 0:
        return None, None, None, None

    return (np.array(sequences,  dtype=np.float32),   # (N, L, F_flat)
            np.array(targets,    dtype=np.int64),      # (N, pH, pW)
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
        test_size   = val_ratio_in_tv,
        random_state= RANDOM_SEED,
        shuffle     = True
    )

    return (X_tv[tr_idx], X_tv[va_idx], X_te,
            y_tv[tr_idx], y_tv[va_idx], y_te,
            origins_te,
            dates_te)


# ============================================================
# NORMALISATION (RobustScaler par feature, sur train)
# ============================================================
def normalize_features_tide(X_train, X_val, X_test):
    # X : (N, L, F_flat)
    N,  L, F = X_train.shape
    nv, _,  _ = X_val.shape
    nt, _,  _ = X_test.shape

    scaler  = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N,  L, F)
    X_va_s  = scaler.transform(X_val.reshape(-1,  F)).reshape(nv, L, F)
    X_te_s  = scaler.transform(X_test.reshape(-1, F)).reshape(nt, L, F)
    return X_tr_s, X_va_s, X_te_s


# ============================================================
# DATASET
# ============================================================
class CDIDatasetTiDE(Dataset):
    """
    X : (N, L, F_flat)   — historical covariates (lookback window)
    y : (N, pH, pW)      — CDI target map for the next month
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)   # (N, L, F_flat)
        self.y = torch.LongTensor(y)    # (N, pH, pW)

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ============================================================
# BLOCS DE BASE TiDE
# ============================================================
class ResidualBlock(nn.Module):
    """
    Dense Residual Block as defined in the TiDE paper (Figure 2).
    LayerNorm → Linear → activation → Dropout → Linear → add skip → LayerNorm
    A projection layer aligns the skip connection when in_dim ≠ out_dim.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1   = nn.Linear(in_dim, out_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Skip projection if dimensions differ
        self.skip  = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x + residual
        x = self.norm2(x)
        return x


# ============================================================
# MODÈLE TiDE  (adapté à la classification CDI par patch)
# ============================================================
class TiDEClassifier(nn.Module):
    """
    TiDE: Time-series Dense Encoder (Das et al., 2023)
    "Long-term Forecasting with TiDE: Time-series Dense Encoder"
    https://arxiv.org/abs/2304.08424

    Adaptation pour notre problème de classification spatiale :
    - Entrée   : (B, L, F_flat)  — L pas de temps × features spatiales aplaties
    - Sortie   : (B, num_classes, pH, pW)  — logits par pixel du patch

    Architecture :
      1. Flatten lookback → (B, L * F_flat)
      2. Encoder  : stack de ResidualBlock → embedding (B, hidden)
      3. Decoder  : stack de ResidualBlock → projection (B, temporal_dim)
      4. Tête     : Linear → (B, num_classes * pH * pW) puis reshape
    """

    def __init__(
        self,
        lookback_len   : int,
        feature_dim    : int,       # F_flat = C * pH * pW
        hidden_size    : int   = HIDDEN_SIZE,
        encoder_layers : int   = ENCODER_LAYERS,
        decoder_layers : int   = DECODER_LAYERS,
        temporal_dim   : int   = TEMPORAL_DIM,
        num_classes    : int   = 6,
        dropout        : float = DROPOUT,
        patch_size     : int   = PATCH_SIZE,
    ):
        super().__init__()
        self.patch_size   = patch_size
        self.num_classes  = num_classes
        input_flat_dim    = lookback_len * feature_dim   # L * F_flat

        # ── Encoder ──────────────────────────────────────────────
        enc_blocks = []
        in_d = input_flat_dim
        for i in range(encoder_layers):
            out_d = hidden_size
            enc_blocks.append(ResidualBlock(in_d, out_d, dropout))
            in_d = out_d
        self.encoder = nn.Sequential(*enc_blocks)

        # ── Decoder ──────────────────────────────────────────────
        dec_blocks = []
        in_d = hidden_size
        for i in range(decoder_layers):
            out_d = temporal_dim if i == decoder_layers - 1 else hidden_size
            dec_blocks.append(ResidualBlock(in_d, out_d, dropout))
            in_d = out_d
        self.decoder = nn.Sequential(*dec_blocks)

        # ── Tête de classification spatiale ──────────────────────
        self.head = nn.Linear(temporal_dim, num_classes * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, F_flat)
        returns logits : (B, num_classes, pH, pW)
        """
        B = x.shape[0]

        # 1. Flatten temporel → (B, L * F_flat)
        x_flat = x.reshape(B, -1)

        # 2. Encoder → (B, hidden_size)
        enc_out = self.encoder(x_flat)

        # 3. Decoder → (B, temporal_dim)
        dec_out = self.decoder(enc_out)

        # 4. Tête → (B, num_classes * pH * pW)
        logits = self.head(dec_out)

        # 5. Reshape → (B, num_classes, pH, pW)
        logits = logits.view(B, self.num_classes, self.patch_size, self.patch_size)
        return logits


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
            logits = model(bX)          # (B, num_classes, pH, pW)
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
    seqs, tgts, dates, origins = prepare_sequences_tile_tide(
        temporal_data, r0, r1, c0, c1, patch_size=PATCH_SIZE)

    if seqs is None or len(seqs) < 10:
        return None

    split = split_data_ordered(seqs, tgts, dates, origins)
    if split is None:
        return None

    X_tr, X_va, X_te, y_tr, y_va, y_te, origins_te, dates_te = split
    del seqs; gc.collect()

    X_tr_s, X_va_s, X_te_s = normalize_features_tide(X_tr, X_va, X_te)
    del X_tr, X_va, X_te; gc.collect()

    L, F_flat = X_tr_s.shape[1], X_tr_s.shape[2]   # lookback, feature_flat

    tr_loader = DataLoader(CDIDatasetTiDE(X_tr_s, y_tr), BATCH_SIZE, shuffle=False, pin_memory=True)
    va_loader = DataLoader(CDIDatasetTiDE(X_va_s, y_va), BATCH_SIZE, shuffle=False, pin_memory=True)
    te_loader = DataLoader(CDIDatasetTiDE(X_te_s, y_te), BATCH_SIZE, shuffle=False, pin_memory=True)
    del X_tr_s, X_va_s, X_te_s; gc.collect()

    model = TiDEClassifier(
        lookback_len   = L,
        feature_dim    = F_flat,
        hidden_size    = HIDDEN_SIZE,
        encoder_layers = ENCODER_LAYERS,
        decoder_layers = DECODER_LAYERS,
        temporal_dim   = TEMPORAL_DIM,
        num_classes    = 6,
        dropout        = DROPOUT,
        patch_size     = PATCH_SIZE,
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
    print(f"RESUME GLOBAL (TiDE) — {len(df)} tuile(s)  |  {total_samples:,} pixels test")
    print("="*70)
    print(f"  Métrique               Moyenne simple    Moyenne pondérée")
    print(f"  {'-'*58}")
    print(f"  Accuracy               {mean_acc:>16.4f}  {wacc:>18.4f}")
    print(f"  Cohen Kappa            {mean_kap:>16.4f}  {wkap:>18.4f}")
    print(f"  F1 Macro               {mean_f1m:>16.4f}  {wf1m:>18.4f}")
    print(f"  F1 Weighted            {mean_f1w:>16.4f}  {wf1w:>18.4f}")
    print("="*70)

    df.to_csv(os.path.join(output_dir, 'tile_results_tide.csv'), index=False)
    print("\nRésultats détaillés → tile_results_tide.csv")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    OUTPUT_DIR = '/kaggle/working/cdi_rasters_tide'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("🚀 CDI PREDICTION — TiDE (raster Tunisie complet)")
    print(f"   Nombre de tuiles souhaité : {NUM_TILES}")
    print(f"   Taille des patchs         : {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"   Hidden size TiDE          : {HIDDEN_SIZE}")
    print(f"   Encoder layers            : {ENCODER_LAYERS}")
    print(f"   Decoder layers            : {DECODER_LAYERS}")
    print(f"   Temporal dim              : {TEMPORAL_DIM}")
    print("="*70)

    temporal_data, raster_transform = collect_temporal_data(DURATION_MONTHS)

    full_shape = list(temporal_data[0]['indicators'].values())[0].shape
    print(f"\n   Raster shape : {full_shape}")

    tiles = generate_tiles(full_shape, NUM_TILES)
    print(f"   Nombre de tuiles : {len(tiles)}")

    print("\n🔄 Traitement des tuiles (TiDE)...\n")
    all_results = []

    for tile_idx, (r0, r1, c0, c1) in enumerate(tiles):
        result = process_tile(temporal_data, tile_idx, r0, r1, c0, c1)
        if result is not None:
            all_results.append(result)

    print_global_summary(all_results, full_shape, output_dir=OUTPUT_DIR)

    print("\n✅ TERMINÉ")
