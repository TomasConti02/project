import cv2
import os
import glob
import numpy as np

def resize_images(input_dir, output_dir=None, size=(256, 256)):
    """
    Ridimensiona tutte le immagini in una directory a 256x256 pixel
    e crea maschere nere su sfondo bianco
    
    Args:
        input_dir (str): Directory di input contenente le immagini
        output_dir (str): Directory di output (opzionale)
        size (tuple): Dimensione desiderata (width, height)
    """
    
    # Se non viene specificata una directory di output, crea una sottodirectory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "resized_256x256")
    
    # Crea la directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Crea sottodirectory per le maschere
    masks_dir = os.path.join(output_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    
    # Trova tutti i file immagine comuni
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, extension)))
        image_files.extend(glob.glob(os.path.join(input_dir, extension.upper())))
    
    print(f"Trovate {len(image_files)} immagini da elaborare")
    
    # Contatori per statistiche
    success_count = 0
    error_count = 0
    
    # Elabora ogni immagine
    for image_path in image_files:
        try:
            # Leggi l'immagine
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Errore: impossibile leggere {image_path}")
                error_count += 1
                continue
            
            # Ridimensiona l'immagine originale
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            # Prepara il nome del file di output per l'immagine ridimensionata
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_256x256{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salva l'immagine ridimensionata
            cv2.imwrite(output_path, resized_img)
            
            # CREAZIONE DELLA MASCHERA INVERTITA (NERA SU BIANCO)
            # Converti in scala di grigi
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            # Metodo 1: Threshold automatico (Otsu)
            _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Metodo 2: Threshold adattivo (per immagini con illuminazione non uniforme)
            mask_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
            
            # Scegli il metodo che produce risultati migliori
            # Conta i pixel bianchi per decidere quale maschera usare
            otsu_white_pixels = np.sum(mask_otsu == 255)
            adaptive_white_pixels = np.sum(mask_adaptive == 255)
            
            # Preferisci la maschera con più pixel bianchi (presumendo che l'oggetto sia chiaro)
            if otsu_white_pixels > adaptive_white_pixels and otsu_white_pixels > 0:
                final_mask = mask_otsu
                method_used = "otsu"
            else:
                final_mask = mask_adaptive
                method_used = "adaptive"
            
            # Applica operazioni morfologiche per pulire la maschera
            kernel = np.ones((3,3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)  # Rimuove rumore
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel) # Riempie buchi
            
            # INVERTI LA MASCHERA: nero su bianco
            inverted_mask = cv2.bitwise_not(final_mask)
            
            # Salva la maschera invertita
            mask_filename = f"{name}_mask{ext}"
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, inverted_mask)
            
            print(f"Elaborata: {filename} -> {output_filename} | Maschera: {mask_filename} ({method_used}, invertita)")
            success_count += 1
            
        except Exception as e:
            print(f"Errore nell'elaborazione di {image_path}: {str(e)}")
            error_count += 1
    
    # Stampa le statistiche finali
    print(f"\nElaborazione completata!")
    print(f"Immagini elaborate con successo: {success_count}")
    print(f"Immagini con errori: {error_count}")
    print(f"Directory immagini: {output_dir}")
    print(f"Directory maschere: {masks_dir}")

# Versione alternativa con più controllo sulla creazione della maschera invertita
def create_advanced_inverted_mask(image):
    """
    Crea una maschera più avanzata invertita (nero su bianco)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Prova diversi metodi di thresholding
    methods = []
    
    # 1. Otsu
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.append(('otsu', otsu_mask))
    
    # 2. Adaptive
    adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    methods.append(('adaptive', adaptive_mask))
    
    # 3. Canny edge detection + fill
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    methods.append(('canny', dilated_edges))
    
    # Seleziona la maschera con l'area più grande (presumendo che l'oggetto sia il componente principale)
    best_mask = None
    best_area = 0
    best_method = ''
    
    for method_name, mask in methods:
        # Trova i contorni
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Prendi il contorno più grande
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > best_area:
                best_area = area
                best_method = method_name
                # Crea una maschera pulita dal contorno più grande
                best_mask = np.zeros_like(mask)
                cv2.fillPoly(best_mask, [largest_contour], 255)
    
    # INVERTI LA MASCHERA SELEZIONATA
    if best_mask is not None:
        inverted_mask = cv2.bitwise_not(best_mask)
        return inverted_mask, best_method
    else:
        return None, 'none'

def resize_images_with_advanced_inverted_masks(input_dir, output_dir=None, size=(256, 256)):
    """
    Versione con maschere invertite più avanzate (nero su bianco)
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "resized_256x256_advanced")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    masks_dir = os.path.join(output_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, extension)))
        image_files.extend(glob.glob(os.path.join(input_dir, extension.upper())))
    
    print(f"Trovate {len(image_files)} immagini da elaborare (versione avanzata invertita)")
    
    success_count = 0
    error_count = 0
    
    for image_path in image_files:
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Errore: impossibile leggere {image_path}")
                error_count += 1
                continue
            
            # Ridimensiona l'immagine
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            # Salva immagine ridimensionata
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_256x256{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, resized_img)
            
            # Crea maschera avanzata invertita
            mask, method_used = create_advanced_inverted_mask(resized_img)
            
            if mask is not None:
                mask_filename = f"{name}_mask{ext}"
                mask_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
                print(f"Elaborata: {filename} -> Maschera invertita ({method_used})")
            else:
                print(f"Avviso: nessuna maschera valida per {filename}")
                # Crea una maschera completamente bianca (nessun oggetto nero)
                mask = np.ones((size[1], size[0]), dtype=np.uint8) * 255
                mask_filename = f"{name}_mask{ext}"
                mask_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
            
            success_count += 1
            
        except Exception as e:
            print(f"Errore nell'elaborazione di {image_path}: {str(e)}")
            error_count += 1
    
    print(f"\nElaborazione avanzata completata!")
    print(f"Immagini elaborate con successo: {success_count}")
    print(f"Immagini con errori: {error_count}")

# Versione semplificata per maschere invertite
def create_simple_inverted_mask(image):
    """
    Crea una maschera invertita semplice (nero su bianco)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Usa Otsu per il thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inverti la maschera
    inverted_mask = cv2.bitwise_not(mask)
    
    return inverted_mask

# Esempio di utilizzo
if __name__ == "__main__":
    # Specifica la directory di input
    input_directory = "/content/drive/MyDrive/branch/tue_immagini" # Sostituisci con il percorso della tua directory
    
    # Verifica che la directory esista
    if not os.path.exists(input_directory):
        print(f"Errore: la directory {input_directory} non esiste!")
    else:
        # Esegui il ridimensionamento con maschere invertite base
        print("=== ELABORAZIONE BASE (MASCHERE INVERTITE) ===")
        resize_images(input_directory)
        
        # Esegui il ridimensionamento con maschere invertite avanzate (opzionale)
        print("\n=== ELABORAZIONE AVANZATA (MASCHERE INVERTITE) ===")
        resize_images_with_advanced_inverted_masks(input_directory)
