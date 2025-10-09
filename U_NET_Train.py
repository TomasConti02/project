import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURAZIONE ===
IMG_SIZE = 256
BASE_DIR = "/content/drive/MyDrive/branch/VOCdevkit/VOC2012"
IMAGES_DIR = os.path.join(BASE_DIR, "JPEGImages")
MASKS_DIR = os.path.join(BASE_DIR, "SegmentationClass")

print("🎯 U-NET - SEGMENTAZIONE RAMI (SENZA SALVATAGGIO)")
print("=" * 60)

# === FUNZIONI DATASET ===
def get_valid_image_mask_pairs():
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
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

def load_and_preprocess_mask_corrected(mask_path):
    mask = cv2.imread(mask_path)
    if mask is None:
        raise ValueError(f"Impossibile caricare la maschera: {mask_path}")

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_resized = cv2.resize(mask_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    binary_mask = np.any(mask_resized != [0, 0, 0], axis=-1).astype(np.float32)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    return binary_mask

def create_dataset(pairs, split_name):
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
        print(f"   ⚠️  {len(failed_pairs)} coppie fallite")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"   ✅ {split_name} completato: {len(X)} immagini")
    return X, y

# === ARCHITETTURA U-NET ===
def build_unet(input_shape=(256, 256, 3)):
    """Costruisce architettura U-Net per segmentazione binaria"""
    print("🧠 Costruzione U-Net...")

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (Contraction Path)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder (Expansion Path)
    u5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs, outputs, name='U-Net')

    print("✅ U-Net costruita con successo!")
    return model

# === LOSS FUNCTIONS ===
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# === TRAINING SENZA SALVATAGGIO ===
def train_unet_no_save(X_train, y_train, X_val, y_val, epochs=30):
    """Addestra il modello SENZA salvataggio"""
    print("\n🏋️‍♂️ INIZIO TRAINING U-NET (SENZA SALVATAGGIO)...")

    # Build model
    model = build_unet()

    # Compila il modello
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=['accuracy', 'Precision', 'Recall', dice_coefficient]
    )

    print("📋 Configurazione training:")
    print("   - Optimizer: Adam (lr=1e-4)")
    print("   - Loss: Combined (BCE + Dice)")
    print("   - Metrics: Accuracy, Precision, Recall, Dice")
    print("   - ⚠️  NESSUN SALVATAGGIO SU DISCO")

    # Callbacks SENZA ModelCheckpoint
    callbacks = [
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Training SENZA data augmentation per semplicità
    print("🚀 Avvio training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Training completato! Modello mantenuto in memoria.")
    return model, history

# === TEST DI VALIDAZIONE COMPLETI ===
def comprehensive_validation(model, X_val, y_val):
    """Test di validazione completi con metriche avanzate"""
    print("\n🔬 TEST DI VALIDAZIONE COMPLETI")
    print("=" * 50)

    # 1. Valutazione base
    print("📊 VALUTAZIONE BASE...")
    val_loss, val_accuracy, val_precision, val_recall, val_dice = model.evaluate(
        X_val, y_val, verbose=0
    )

    print("🎯 METRICHE PRINCIPALI:")
    print(f"   • Loss: {val_loss:.4f}")
    print(f"   • Accuracy: {val_accuracy:.4f}")
    print(f"   • Precision: {val_precision:.4f}")
    print(f"   • Recall: {val_recall:.4f}")
    print(f"   • Dice: {val_dice:.4f}")

    # 2. Calcolo F1-Score
    f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    print(f"   • F1-Score: {f1:.4f}")

    # 3. Metriche avanzate
    print("\n📈 METRICHE AVANZATE...")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(np.float32)

    # Flatten per metriche sklearn
    y_true_flat = y_val.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # Classification Report
    print("📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true_flat, y_pred_flat,
                              target_names=['Background', 'Foreground'],
                              digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    print("🎯 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0,0]:>8} ({(cm[0,0]/len(y_true_flat)*100):.2f}%)")
    print(f"   False Positives: {cm[0,1]:>8} ({(cm[0,1]/len(y_true_flat)*100):.2f}%)")
    print(f"   False Negatives: {cm[1,0]:>8} ({(cm[1,0]/len(y_true_flat)*100):.2f}%)")
    print(f"   True Positives:  {cm[1,1]:>8} ({(cm[1,1]/len(y_true_flat)*100):.2f}%)")

    # 4. Statistiche aggiuntive
    total_pixels = len(y_true_flat)
    foreground_true = np.sum(y_true_flat)
    foreground_pred = np.sum(y_pred_flat)

    print(f"\n📐 STATISTICHE PIXEL:")
    print(f"   • Totali: {total_pixels}")
    print(f"   • Foreground veri: {foreground_true} ({foreground_true/total_pixels*100:.2f}%)")
    print(f"   • Foreground predetti: {foreground_pred} ({foreground_pred/total_pixels*100:.2f}%)")

    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'dice': val_dice,
        'f1': f1,
        'confusion_matrix': cm
    }

def analyze_performance(history, val_metrics):
    """Analizza le performance del training"""
    print("\n📊 ANALISI PERFORMANCE")
    print("=" * 40)

    # Metriche finali training
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_dice = history.history['dice_coefficient'][-1]
    final_val_dice = history.history['val_dice_coefficient'][-1]

    print("🎯 CONFRONTO FINALE:")
    print(f"   Training Loss: {final_train_loss:.4f}")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    print(f"   Training Dice: {final_train_dice:.4f}")
    print(f"   Validation Dice: {final_val_dice:.4f}")

    # Analisi overfitting
    loss_gap = final_train_loss - final_val_loss
    dice_gap = final_train_dice - final_val_dice

    print(f"\n⚠️  ANALISI OVERFITTING:")
    print(f"   Gap Loss: {loss_gap:.4f} {'✅' if abs(loss_gap) < 0.05 else '❌'}")
    print(f"   Gap Dice: {dice_gap:.4f} {'✅' if abs(dice_gap) < 0.05 else '❌'}")

    if abs(loss_gap) < 0.05 and abs(dice_gap) < 0.05:
        print("   🎉 MODELLO BEN BILANCIATO - Overfitting minimo")
    else:
        print("   ⚠️  POSSIBILE OVERFITTING - Considera regolarizzazione")

def visualize_validation_results(model, X_val, y_val, num_samples=5):
    """Visualizza i risultati della validazione"""
    print(f"\n👀 VISUALIZZAZIONE RISULTATI VALIDAZIONE ({num_samples} campioni)...")

    indices = random.sample(range(len(X_val)), min(num_samples, len(X_val)))
    X_samples = X_val[indices]
    y_true_samples = y_val[indices]

    y_pred_samples = model.predict(X_samples, verbose=0)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))

    for i in range(num_samples):
        # Input
        axes[i, 0].imshow(X_samples[i])
        axes[i, 0].set_title('Immagine Input')
        axes[i, 0].axis('off')

        # Ground Truth
        axes[i, 1].imshow(y_true_samples[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Maschera Vera')
        axes[i, 1].axis('off')

        # Predizione
        axes[i, 2].imshow(y_pred_samples[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Predizione')
        axes[i, 2].axis('off')

        # Overlay
        axes[i, 3].imshow(X_samples[i])
        axes[i, 3].imshow(y_pred_samples[i].squeeze(), cmap='Reds', alpha=0.6)
        axes[i, 3].set_title('Overlay Predizione')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

# === SCRIPT PRINCIPALE ===
def main():
    """Script principale - Training e validazione completi SENZA salvataggio"""
    print("🚀 U-NET - TRAINING E VALIDAZIONE COMPLETI")
    print("=" * 60)

    # 1. Caricamento dati
    print("\n1️⃣  FASE 1: CARICAMENTO DATI")
    valid_pairs = get_valid_image_mask_pairs()

    if len(valid_pairs) == 0:
        print("❌ Nessuna coppia valida trovata!")
        return

    # Split dataset
    train_pairs, val_pairs = train_test_split(
        valid_pairs, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"🎯 Split dataset:")
    print(f"   Train: {len(train_pairs)} immagini")
    print(f"   Validation: {len(val_pairs)} immagini")

    # Creazione dataset
    X_train, y_train = create_dataset(train_pairs, "train")
    X_val, y_val = create_dataset(val_pairs, "validation")

    # 2. Training SENZA salvataggio
    print("\n2️⃣  FASE 2: TRAINING (SENZA SALVATAGGIO)")
    model, history = train_unet_no_save(X_train, y_train, X_val, y_val, epochs=30)

    # 3. Test di validazione completi
    print("\n3️⃣  FASE 3: TEST DI VALIDAZIONE COMPLETI")
    val_metrics = comprehensive_validation(model, X_val, y_val)

    # 4. Analisi performance
    analyze_performance(history, val_metrics)

    # 5. Visualizzazioni
    print("\n4️⃣  FASE 4: VISUALIZZAZIONI")
    visualize_validation_results(model, X_val, y_val, num_samples=5)

    # 6. Riepilogo finale
    print("\n" + "=" * 60)
    print("🎉 TRAINING E VALIDAZIONE COMPLETATI!")
    print("📊 RIEPILOGO FINALE:")
    print(f"   • Modello mantenuto in memoria")
    print(f"   • Dice Coefficient: {val_metrics['dice']:.4f}")
    print(f"   • Precision: {val_metrics['precision']:.4f}")
    print(f"   • Recall: {val_metrics['recall']:.4f}")
    print(f"   • F1-Score: {val_metrics['f1']:.4f}")
    print(f"   • Pronto per l'uso!")

    return model, history, val_metrics

# === ESECUZIONE ===
if __name__ == "__main__":
    # Disabilita salvataggi automatici
    import warnings
    warnings.filterwarnings('ignore')

    print(f"✅ TensorFlow: {tf.__version__}")
    print(f"✅ GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("⚠️  MODALITÀ SENZA SALVATAGGIO ATTIVA")

    # Esegui training e validazione
    model, history, val_metrics = main()

    print("\n" + "=" * 60)
    print("🚀 MODELLO PRONTO PER L'USO!")
    print("""
    ISTRUZIONI PER L'USO:

    # Predizione su nuova immagine
    def segmenta_rami(image_path):
        img = load_and_preprocess_image(image_path)
        prediction = model.predict(np.expand_dims(img, axis=0))
        return prediction[0]

    # Esempio:
    # maschera = segmenta_rami("percorso/immagine.jpg")
    # plt.imshow(maschera.squeeze(), cmap='gray')
    # plt.show()
    """)
