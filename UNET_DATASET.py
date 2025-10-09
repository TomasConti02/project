import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = 256
BASE_DIR = "/content/drive/MyDrive/branch/VOCdevkit/VOC2012"
IMAGES_DIR = os.path.join(BASE_DIR, "JPEGImages")
MASKS_DIR = os.path.join(BASE_DIR, "SegmentationClass")

print("🎯 Configurazione:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR} → Esiste: {os.path.exists(IMAGES_DIR)}")
print(f"MASKS_DIR: {MASKS_DIR} → Esiste: {os.path.exists(MASKS_DIR)}")

def get_valid_image_mask_pairs():
    """Trova tutte le coppie valide immagine-maschera"""
    print("🔍 Ricerca coppie immagine-maschera valide...")
    
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]
    
    print(f"   Trovate {len(image_files)} immagini e {len(mask_files)} maschere")
    
    valid_pairs = []
    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        mask_file = image_id + '.png'
        mask_path = os.path.join(MASKS_DIR, mask_file)
        
        if os.path.exists(mask_path):
            valid_pairs.append({
                'id': image_id,
                'image_path': os.path.join(IMAGES_DIR, image_file),
                'mask_path': mask_path
            })
    
    print(f"   ✅ Coppie valide trovate: {len(valid_pairs)}")
    return valid_pairs

def load_and_preprocess_image(image_path):
    """Carica e preprocessa un'immagine"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

def load_and_preprocess_mask_corrected(mask_path):
    """Carica e preprocessa una maschera - versione corretta"""
    mask = cv2.imread(mask_path)
    if mask is None:
        raise ValueError(f"Impossibile caricare la maschera: {mask_path}")
    
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_resized = cv2.resize(mask_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Strategia corretta: tutto ciò che non è nero [0,0,0] è foreground
    binary_mask = np.any(mask_resized != [0, 0, 0], axis=-1).astype(np.float32)
    
    # Aggiungi dimensione canale
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    return binary_mask

def create_dataset(pairs, split_name):
    """Crea il dataset per un determinato split"""
    print(f"\n📦 Creazione dataset {split_name}...")
    
    X, y = [], []
    failed_pairs = []
    
    for pair in tqdm(pairs):
        try:
            img = load_and_preprocess_image(pair['image_path'])
            mask = load_and_preprocess_mask_corrected(pair['mask_path'])
            
            X.append(img)
            y.append(mask)
            
        except Exception as e:
            failed_pairs.append((pair['id'], str(e)))
            continue
    
    if failed_pairs:
        print(f"   ⚠️  {len(failed_pairs)} coppie fallite:")
        for failed_id, error in failed_pairs[:3]:
            print(f"     - {failed_id}: {error}")
    
    if len(X) == 0:
        raise ValueError(f"Nessuna immagine processata per {split_name}!")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"   ✅ {split_name} completato: {len(X)} immagini")
    return X, y

def analyze_dataset(X, y, split_name):
    """Analizza le statistiche del dataset"""
    print(f"\n📊 ANALISI {split_name.upper()}:")
    print(f"   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    print(f"   Range X: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   Range y: [{y.min():.3f}, {y.max():.3f}]")
    
    total_pixels = y.shape[0] * y.shape[1] * y.shape[2]
    foreground_pixels = np.sum(y > 0.5)
    background_pixels = np.sum(y <= 0.5)
    
    print(f"   Pixel foreground: {foreground_pixels} ({foreground_pixels/total_pixels*100:.2f}%)")
    print(f"   Pixel background: {background_pixels} ({background_pixels/total_pixels*100:.2f}%)")
    
    non_empty_masks = np.sum(np.sum(y, axis=(1,2,3)) > 0)
    print(f"   Maschere non vuote: {non_empty_masks}/{len(y)} ({non_empty_masks/len(y)*100:.1f}%)")
    
    foreground_ratios = [np.sum(mask) / mask.size for mask in y]
    print(f"   Ratio foreground - Media: {np.mean(foreground_ratios):.4f}")
    print(f"   Ratio foreground - Min: {np.min(foreground_ratios):.4f}")
    print(f"   Ratio foreground - Max: {np.max(foreground_ratios):.4f}")

def validate_dataset_quality(X_train, y_train, X_val, y_val):
    """Valida la qualità del dataset"""
    print("\n" + "=" * 60)
    print("🔍 VALIDAZIONE DATASET")
    print("=" * 60)
    
    # 1. Controllo consistenza dimensioni
    print("✅ 1. Controllo consistenza dimensioni:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    assert X_train.shape[0] == y_train.shape[0], "Train: X e y hanno lunghezze diverse"
    assert X_val.shape[0] == y_val.shape[0], "Val: X e y hanno lunghezze diverse"
    print("   ✅ Dimensioni consistenti")
    
    # 2. Controllo range valori
    print("\n✅ 2. Controllo range valori:")
    print(f"   X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    assert X_train.min() >= 0 and X_train.max() <= 1.0, "X_train fuori range [0,1]"
    assert y_train.min() == 0.0 and y_train.max() == 1.0, "y_train non binario"
    print("   ✅ Range valori corretto")
    
    # 3. Controllo maschere non vuote
    print("\n✅ 3. Controllo maschere non vuote:")
    train_non_empty = np.sum(np.sum(y_train, axis=(1,2,3)) > 0)
    val_non_empty = np.sum(np.sum(y_val, axis=(1,2,3)) > 0)
    
    print(f"   Train: {train_non_empty}/{len(y_train)} ({train_non_empty/len(y_train)*100:.1f}%)")
    print(f"   Val: {val_non_empty}/{len(y_val)} ({val_non_empty/len(y_val)*100:.1f}%)")
    
    assert train_non_empty > 0, "Nessuna maschera non vuota in train"
    assert val_non_empty > 0, "Nessuna maschera non vuota in val"
    print("   ✅ Maschere non vuote presenti")
    
    # 4. Controllo bilanciamento
    print("\n✅ 4. Controllo bilanciamento:")
    train_foreground_ratio = np.sum(y_train > 0.5) / y_train.size
    val_foreground_ratio = np.sum(y_val > 0.5) / y_val.size
    
    print(f"   Train foreground ratio: {train_foreground_ratio:.4f}")
    print(f"   Val foreground ratio: {val_foreground_ratio:.4f}")
    print(f"   Differenza: {abs(train_foreground_ratio - val_foreground_ratio):.4f}")
    
    if abs(train_foreground_ratio - val_foreground_ratio) < 0.05:
        print("   ✅ Bilanciamento accettabile")
    else:
        print("   ⚠️  Attenzione: possibile sbilanciamento")
    
    # 5. Controllo memoria
    print("\n✅ 5. Utilizzo memoria:")
    train_memory = X_train.nbytes + y_train.nbytes
    val_memory = X_val.nbytes + y_val.nbytes
    total_memory = train_memory + val_memory
    
    print(f"   Train: {train_memory / 1024 / 1024:.1f} MB")
    print(f"   Val: {val_memory / 1024 / 1024:.1f} MB")
    print(f"   Totale: {total_memory / 1024 / 1024:.1f} MB")
    
    return True

def visualize_samples(X, y, n_samples=3, title="Campioni"):
    """Visualizza alcuni campioni del dataset"""
    print(f"\n👀 Visualizzazione {n_samples} {title}...")
    
    if len(X) == 0:
        print("   ❌ Nessun campione da visualizzare!")
        return
        
    indices = random.sample(range(len(X)), min(n_samples, len(X)))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    for i, idx in enumerate(indices):
        # Immagine originale
        axes[i, 0].imshow(X[idx])
        axes[i, 0].set_title(f'Immagine {idx}')
        axes[i, 0].axis('off')
        
        # Maschera
        mask_display = y[idx].squeeze()
        axes[i, 1].imshow(mask_display, cmap='gray')
        axes[i, 1].set_title(f'Maschera {idx}\nForeground: {np.sum(mask_display > 0.5):.0f} px')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(X[idx])
        axes[i, 2].imshow(mask_display, cmap='Reds', alpha=0.5)
        axes[i, 2].set_title(f'Overlay {idx}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Funzione principale - VALIDAZIONE SENZA SALVATAGGIO"""
    print("🚀 VALIDAZIONE DATASET U-NET")
    print("=" * 60)
    
    # Trova coppie valide
    valid_pairs = get_valid_image_mask_pairs()
    
    if len(valid_pairs) == 0:
        print("❌ Nessuna coppia valida trovata!")
        return
    
    # Split train/val
    train_pairs, val_pairs = train_test_split(
        valid_pairs, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\n🎯 SPLIT DATASET:")
    print(f"   Train: {len(train_pairs)} immagini")
    print(f"   Val: {len(val_pairs)} immagini")
    
    # Crea dataset train
    X_train, y_train = create_dataset(train_pairs, "train")
    
    # Crea dataset val
    X_val, y_val = create_dataset(val_pairs, "val")
    
    # Analisi dettagliata
    analyze_dataset(X_train, y_train, "train")
    analyze_dataset(X_val, y_val, "val")
    
    # Validazione completa
    validation_passed = validate_dataset_quality(X_train, y_train, X_val, y_val)
    
    # Visualizza campioni
    visualize_samples(X_train, y_train, 3, "TRAIN")
    visualize_samples(X_val, y_val, 3, "VALIDATION")
    
    print("\n" + "=" * 60)
    if validation_passed:
        print("🎉 DATASET VALIDATO CON SUCCESSO!")
        print("💡 Il dataset è pronto per il training U-Net")
        print(f"📊 Dimensionalità finali:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}") 
        print(f"   X_val: {X_val.shape}")
        print(f"   y_val: {y_val.shape}")
    else:
        print("❌ PROBLEMI NELLA VALIDAZIONE")
    
    return X_train, y_train, X_val, y_val

# Esegui
if __name__ == "__main__":
    X_train, y_train, X_val, y_val = main()
