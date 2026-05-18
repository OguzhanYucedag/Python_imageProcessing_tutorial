import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATASET_DIR = Path(__file__).resolve().parent / "veri_seti"
MODEL_PATH = Path(__file__).resolve().parent / "cat_dog_cnn.pt"
META_PATH = Path(__file__).resolve().parent / "cat_dog_cnn_meta.json"
CLASS_NAMES = ["cat", "dog"]
IMAGE_SIZE = 128
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_dataloaders(batch_size=16, val_ratio=0.2):
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=train_transform)
    if len(full_dataset) < 100:
        raise RuntimeError("Model egitimi icin cok az veri var. En az 10 goruntu gerekir.")

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Validation setinde augmentation istemedigimiz icin transformu degistiriyoruz.
    val_dataset.dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
    

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return (correct / total) * 100.0 if total else 0.0


def train_model(epochs=12, batch_size=16, learning_rate=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanilan cihaz: {device}")
    print("[1/4] Veri yukleniyor...")
    train_loader, val_loader = build_dataloaders(batch_size=batch_size)

    print("[2/4] Model olusturuluyor...")
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("[3/4] Egitim basliyor...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: %{val_acc:.2f}")

    print("[4/4] Model kaydediliyor...")
    torch.save(model.state_dict(), str(MODEL_PATH))
    final_acc = evaluate(model, val_loader, device)
    meta = {
        "classes": CLASS_NAMES,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_accuracy_percent": round(final_acc, 2),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nModel kaydedildi: {MODEL_PATH.name}")
    print(f"Meta kaydedildi : {META_PATH.name}")
    print(f"Validation Dogrulugu: %{final_acc:.2f}")


def prepare_image(image_path: Path):
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def show_prediction_window(image_path: Path, classes, probs, predicted_class, confidence, threshold):
    image = cv2.imread(str(image_path))
    if image is None:
        print("[UYARI] Goruntu ekranda gosterilemedi.")
        return

    max_width = 900
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (max_width, new_height))

    panel = np.full((110, image.shape[1], 3), 30, dtype=np.uint8)
    cat_prob = probs[classes.index("cat")] * 100.0 if "cat" in classes else probs[0] * 100.0
    dog_prob = probs[classes.index("dog")] * 100.0 if "dog" in classes else probs[1] * 100.0

    line1 = f"Cat: %{cat_prob:.2f}   Dog: %{dog_prob:.2f}"
    if confidence < threshold:
        line2 = f"Sonuc: Emin degil (Esik: %{threshold:.0f})"
    else:
        line2 = f"Sonuc: {predicted_class} (Guven: %{confidence:.2f})"

    cv2.putText(panel, line1, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, line2, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)

    combined = np.vstack((panel, image))
    cv2.imshow("Tahmin Sonucu - Cikis icin bir tusa basin", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_image(image_path: Path, threshold=70.0, show_window=True):
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model bulunamadi. Once egitim icin: python Comparison_of_cats_and_dogs.py train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    classes = meta.get("classes", CLASS_NAMES)

    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()

    image_tensor = prepare_image(image_path).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_index = int(probs.argmax())
    confidence = float(probs[pred_index] * 100.0)
    predicted_class = classes[pred_index]
    print(f"Cat olasiligi: %{probs[classes.index('cat')] * 100.0:.2f}" if "cat" in classes else f"{classes[0]} olasiligi: %{probs[0] * 100.0:.2f}")
    print(f"Dog olasiligi: %{probs[classes.index('dog')] * 100.0:.2f}" if "dog" in classes else f"{classes[1]} olasiligi: %{probs[1] * 100.0:.2f}")

    if confidence < threshold:
        print(f"Sonuc : Emin degil (en yuksek guven: %{confidence:.2f})")
    else:
        print(f"Tahmin: {predicted_class}")
        print(f"Guven : %{confidence:.2f}")

    if show_window:
        show_prediction_window(image_path, classes, probs, predicted_class, confidence, threshold)


def get_random_image_from_class(class_name: str):
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Sinif klasoru bulunamadi: {class_dir}")

    candidates = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()]
    if not candidates:
        raise RuntimeError(f"{class_name} klasorunde gecerli goruntu bulunamadi.")
    return random.choice(candidates)


def interactive_random_predict(threshold=70.0, show_window=True):
    user_choice = input("Hangi siniftan rastgele resim secilsin? (cat/dog): ").strip().lower()
    if user_choice not in CLASS_NAMES:
        print("Gecersiz giris. Lutfen sadece 'cat' veya 'dog' yazin.")
        return

    selected_image = get_random_image_from_class(user_choice)
    print(f"Secilen dosya: {selected_image}")
    predict_image(selected_image, threshold=threshold, show_window=show_window)


def main():
    parser = argparse.ArgumentParser(description="Kedi-Kopek siniflandirma (PyTorch CNN)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="CNN modeli egit")
    train_parser.add_argument("--epochs", type=int, default=12, help="Egitim epoch sayisi")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch boyutu")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Ogrenme orani")

    predict_parser = subparsers.add_parser("predict", help="Tek bir goruntuyu tahmin et")
    predict_parser.add_argument("--image", required=True, type=str, help="Tahmin edilecek goruntu yolu")
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Bu yuzdenin altinda model sadece 'Emin degil' der",
    )
    predict_parser.add_argument(
        "--no-window",
        action="store_true",
        help="Ekranda goruntu penceresi acma",
    )
    random_parser = subparsers.add_parser(
        "random-predict",
        help="Terminalden cat/dog alip veri setinden rastgele resim sec ve tahmin et",
    )
    random_parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Bu yuzdenin altinda model sadece 'Emin degil' der",
    )
    random_parser.add_argument(
        "--no-window",
        action="store_true",
        help="Ekranda goruntu penceresi acma",
    )

    args = parser.parse_args()
    if args.command == "train":
        train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.command == "predict":
        predict_image(Path(args.image), threshold=args.threshold, show_window=not args.no_window)
    elif args.command == "random-predict":
        interactive_random_predict(threshold=args.threshold, show_window=not args.no_window)


if __name__ == "__main__":
    main()
