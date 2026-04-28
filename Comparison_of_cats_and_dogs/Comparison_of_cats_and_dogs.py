import cv2

# 1. Adım: Tam dosya yolunu başına 'r' koyarak yazın. 
# DİKKAT: Dosya adınız 'etraflarinda' mı yoksa 'etraflarini' mi, klasörden kontrol edip buraya doğrusunu yazın!
dosya_yolu = r'C:\Users\Oguzhan\Desktop\Python_imageProcessing\Comparison_of_cats_and_dogs\mutlu-insanlar-etraflarini-dogru-insanlarla-kusatirlar-2.jpg'

img = cv2.imread(dosya_yolu,0)

# 2. Adım: Fotoğrafın başarıyla okunup okunmadığını kontrol edin
if img is None:
    print("HATA: Fotoğraf bulunamadı! Lütfen dosya yolunu, ismini ve uzantısını kontrol edin.")
else:
    # 3. Adım: Eğer fotoğraf bulunduysa ekranda göster
    cv2.imshow("ilk", img)
    
    # Pencerenin hemen kapanmaması için klavyeden bir tuşa basılmasını bekle
    cv2.waitKey(0) 
    
    # Tuşa basıldıktan sonra tüm pencereleri temizle
    cv2.destroyAllWindows()