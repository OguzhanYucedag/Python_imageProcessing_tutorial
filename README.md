# Python Image Processing Tutorial

Bu projede farkli Python surumleri oldugu icin `venv` kullanmak gerekir.

## 1) Ortami kur (bir kere)

PowerShell:

```powershell
cd "c:/Users/Oguzhan/Documents/GitHub/Python_imageProcessing_tutorial"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Model egit

```powershell
.\.venv\Scripts\Activate.ps1
python "Comparison_of_cats_and_dogs/Comparison_of_cats_and_dogs.py" train --epochs 12 --batch-size 16 --lr 0.001
```

## 3) Tahmin yap (ekranda degerleri gosterir)

```powershell
.\.venv\Scripts\Activate.ps1
python "Comparison_of_cats_and_dogs/Comparison_of_cats_and_dogs.py" predict --image "Comparison_of_cats_and_dogs/veri_seti/cat/Image_24.jpg" --threshold 70
```

## Notlar

- Sadece `python script.py` calistirmayin; script `train` veya `predict` komutu ister.
- Eger pencere acmadan sadece terminal ciktisi istiyorsaniz `--no-window` ekleyin.
- Ortamin dogru oldugunu kontrol etmek icin `python --version` komutunda `3.13.x` gorunmeli.
