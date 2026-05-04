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
SEQUENCE_LENGTH = 3
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
DROPOUT         = 0.3
EPOCHS          = 20
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
PATIENCE        = 10
RANDOM_SEED     = 42

KERNEL_SIZE     = 3
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
# SÉQUENCES SPATIO-TEMPORELLES POUR UNE TUILE (ConvLSTM)
# ============================================================
def prepare_sequences_tile_convlstm(temporal_data, r0, r1, c0, c1, patch_size=PATCH_SIZE):
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
                sub_seq = seq_stack[:, :, pr:pr+patch_size, pc:pc+patch_size]
                sub_tgt = tgt_patch[pr:pr+patch_size, pc:pc+patch_size]

                sequences.append(sub_seq)
                targets.append(sub_tgt)
                seq_dates.append(target_month['date'])
                patch_origins.append((r0 + pr, c0 + pc))

    if len(sequences) == 0:
        return None, None, None, None

    return (np.array(sequences,     dtype=np.float32),
            np.array(targets,       dtype=np.int64),
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
def normalize_features_spatial(X_train, X_val, X_test):
    N, T, C, pH, pW = X_train.shape
    scalers = []
    X_tr_s  = X_train.copy()
    X_va_s  = X_val.copy()
    X_te_s  = X_test.copy()

    for c_idx in range(C):
        flat = X_train[:, :, c_idx, :, :].reshape(-1, 1)
        sc   = RobustScaler()
        sc.fit(flat)
        scalers.append(sc)

        X_tr_s[:, :, c_idx, :, :] = sc.transform(
            X_train[:, :, c_idx, :, :].reshape(-1, 1)).reshape(N, T, pH, pW)
        nv = X_val.shape[0]
        X_va_s[:, :, c_idx, :, :] = sc.transform(
            X_val[:, :, c_idx, :, :].reshape(-1, 1)).reshape(nv, T, pH, pW)
        nte = X_test.shape[0]
        X_te_s[:, :, c_idx, :, :] = sc.transform(
            X_test[:, :, c_idx, :, :].reshape(-1, 1)).reshape(nte, T, pH, pW)

    return X_tr_s, X_va_s, X_te_s


# ============================================================
# DATASET
# ============================================================
class CDIDatasetConvLSTM(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ============================================================
# CELLULE CONVLSTM
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


# ============================================================
# MODÈLE CONVLSTM EMPILÉ + TÊTE DE CLASSIFICATION PIXEL
# ============================================================
class CDIConvLSTMClassifier(nn.Module):
    def __init__(self, in_channels=NUM_FEATURES, hidden_channels=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=6,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        self.num_layers      = num_layers
        self.hidden_channels = hidden_channels

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(c_in, hidden_channels, kernel_size))

        self.dropout = nn.Dropout2d(dropout)

        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        states = [cell.init_hidden(B, (H, W), x.device) for cell in self.cells]

        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                h, c = states[l]
                h, c = cell(inp, h, c)
                states[l] = (h, c)
                inp = h

        h_last = states[-1][0]
        h_last = self.dropout(h_last)
        return self.classifier(h_last)


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
    seqs, tgts, dates, origins = prepare_sequences_tile_convlstm(
        temporal_data, r0, r1, c0, c1, patch_size=PATCH_SIZE)

    if seqs is None or len(seqs) < 10:
        return None

    split = split_data_ordered(seqs, tgts, dates, origins)
    if split is None:
        return None

    X_tr, X_va, X_te, y_tr, y_va, y_te, origins_te, dates_te = split
    del seqs; gc.collect()

    X_tr_s, X_va_s, X_te_s = normalize_features_spatial(X_tr, X_va, X_te)
    del X_tr, X_va, X_te; gc.collect()

    tr_loader = DataLoader(CDIDatasetConvLSTM(X_tr_s, y_tr), BATCH_SIZE, shuffle=False, pin_memory=True)
    va_loader = DataLoader(CDIDatasetConvLSTM(X_va_s, y_va), BATCH_SIZE, shuffle=False, pin_memory=True)
    te_loader = DataLoader(CDIDatasetConvLSTM(X_te_s, y_te), BATCH_SIZE, shuffle=False, pin_memory=True)
    del X_tr_s, X_va_s, X_te_s; gc.collect()

    model = CDIConvLSTMClassifier(
        in_channels     = NUM_FEATURES,
        hidden_channels = HIDDEN_SIZE,
        num_layers      = NUM_LAYERS,
        num_classes     = 6,
        kernel_size     = KERNEL_SIZE,
        dropout         = DROPOUT
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
def print_global_summary(results, output_dir='.'):
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
    print(f"RESUME GLOBAL (ConvLSTM) — {len(df)} tuile(s)  |  {total_samples:,} pixels test")
    print("="*70)
    print(f"  Métrique               Moyenne simple    Moyenne pondérée")
    print(f"  {'-'*58}")
    print(f"  Accuracy               {mean_acc:>16.4f}  {wacc:>18.4f}")
    print(f"  Cohen Kappa            {mean_kap:>16.4f}  {wkap:>18.4f}")
    print(f"  F1 Macro               {mean_f1m:>16.4f}  {wf1m:>18.4f}")
    print(f"  F1 Weighted            {mean_f1w:>16.4f}  {wf1w:>18.4f}")
    print("="*70)

    df.to_csv(os.path.join(output_dir, 'tile_results_convlstm.csv'), index=False)
    print("\nRésultats détaillés → tile_results_convlstm.csv")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    OUTPUT_DIR = '/kaggle/working/cdi_rasters_convlstm'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("🚀 CDI PREDICTION — ConvLSTM (raster Tunisie complet)")
    print(f"   Nombre de tuiles souhaité : {NUM_TILES}")
    print(f"   Taille des patchs         : {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"   Filtres ConvLSTM          : {HIDDEN_SIZE}")
    print(f"   Couches ConvLSTM          : {NUM_LAYERS}")
    print(f"   Kernel size               : {KERNEL_SIZE}")
    print("="*70)

    temporal_data, raster_transform = collect_temporal_data(DURATION_MONTHS)

    full_shape = list(temporal_data[0]['indicators'].values())[0].shape
    print(f"\n   Raster shape : {full_shape}")

    tiles = generate_tiles(full_shape, NUM_TILES)
    print(f"   Nombre de tuiles : {len(tiles)}")

    print("\n🔄 Traitement des tuiles (ConvLSTM)...\n")
    all_results = []

    for tile_idx, (r0, r1, c0, c1) in enumerate(tiles):
        result = process_tile(temporal_data, tile_idx, r0, r1, c0, c1)
        if result is not None:
            all_results.append(result)

    print_global_summary(all_results, output_dir=OUTPUT_DIR)

    print("\n✅ TERMINÉ")
