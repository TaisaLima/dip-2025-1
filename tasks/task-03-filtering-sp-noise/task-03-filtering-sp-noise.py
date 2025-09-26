import cv2 as cv
import numpy as np

def remove_salt_and_pepper_noise(image):
    return cv.medianBlur(image, 3)

if __name__ == "__main__":
   
    img = cv.imread("head.png", cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Não foi possível carregar 'imagem.png'")

    denoised_img = remove_salt_and_pepper_noise(img)
    cv.imshow("Sem ruido", denoised_img)
    cv.waitKey(0)

    cv.destroyAllWindows()
