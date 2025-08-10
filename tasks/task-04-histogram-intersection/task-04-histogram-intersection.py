import numpy as np
import cv

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    hist1, _ = np.histogram(img1, bins=256, range=(0, 256))
    hist2, _ = np.histogram(img2, bins=256, range=(0, 256))

    hist1 = hist1.astype(np.float32) / hist1.sum()
    hist2 = hist2.astype(np.float32) / hist2.sum()

    intersection = np.sum(np.minimum(hist1, hist2))
    return float(intersection)

if __name__ == "__main__":
    # Cria imagem 1 (gradiente horizontal)
    height, width = 256, 256
    img1 = np.tile(np.arange(width, dtype=np.uint8), (height, 1))

    # Cria imagem 2 (gradiente + ruído)
    noise = np.random.randint(-20, 21, (height, width), dtype=np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Salva para visualização
    cv.imwrite("synthetic_img1.png", img1)
    cv.imwrite("synthetic_img2.png", img2)

    # Mostra as imagens
    cv.imshow("Imagem 1 - Gradiente", img1)
    cv.imshow("Imagem 2 - Gradiente + Ruido", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Calcula similaridade
    score = compute_histogram_intersection(img1, img2)
    print(f"Histogram Intersection Score: {score:.4f}")
