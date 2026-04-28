import cv2
import numpy as np
from pathlib import Path
# 1) Görüntüyü oku

img_path = Path(__file__).resolve().parent / "ornek.jpg"
img = cv2.imread(str(img_path))
# Kontrol: dosya okunamadıysa img None olur
if img is None:

    raise FileNotFoundError(f"Goruntu bulunamadi: {img_path}") #Dosya yolunu kontrol edin

# 2) Boyut bilgisi al
height, width, channels = img.shape
print(f"Yukseklik: {height}, Genislik: {width}, Kanal: {channels}")

# 3) BGR -> Gray donusumu
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4) Yeniden boyutlandir
resized = cv2.resize(img, (640, 480))

# 5) ROI (Region of Interest) kirp
roi = img[100:300, 200:500]  # y1:y2, x1:x2

# 6) Ekranda goster
cv2.imshow("Orijinal", img)
cv2.imshow("Gray", gray)
cv2.imshow("Resized", resized)
cv2.imshow("ROI", roi)

# 7) Kaydet
cv2.imwrite("gray_output.jpg", gray)
cv2.imwrite("roi_output.jpg", roi)

# 8) Tus bekle ve pencereleri kapat
cv2.waitKey(0)
cv2.destroyAllWindows()