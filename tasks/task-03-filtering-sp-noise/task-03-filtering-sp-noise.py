import cv
import numpy as np

# Função para adicionar ruído sal e pimenta
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = image.copy()
    height, width = image.shape

    # Salt (branco)
    salt = np.random.rand(height, width) < salt_prob
    noisy[salt] = 255

    # Pepper (preto)
    pepper = np.random.rand(height, width) < pepper_prob
    noisy[pepper] = 0

    return noisy

# Função para remover o ruído
def remove_salt_and_pepper_noise(image):
    return cv.medianBlur(image, 3)

if __name__ == "__main__":
    # Lê a imagem
    img = cv.imread("imagem.png", cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Não foi possível carregar 'imagem.png'")

    # Mostra original
    cv.imshow("Original", img)
    cv.waitKey(0)

    # Adiciona ruído
    noisy_img = add_salt_and_pepper_noise(img)
    cv.imshow("Com ruido", noisy_img)
    cv.waitKey(0)

    # Remove o ruído
    denoised_img = remove_salt_and_pepper_noise(noisy_img)
    cv.imshow("Sem ruido", denoised_img)
    cv.waitKey(0)

    cv.destroyAllWindows()
