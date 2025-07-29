import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleTransformer

# Preparar datos
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer().to(device)
model.load_state_dict(torch.load("modelo_entrenado.pth"))
model.eval()

# Evaluación
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nPrecisión en test: {100 * correct / total:.2f}%")


# Tomar un batch del loader
examples = iter(test_loader)
images, labels = next(examples)

# Pasar las imagenes por el modelo
images_gpu = images.to(device)
outputs = model(images_gpu)
_, preds = torch.max(outputs, 1)

# Mostrar imagenes con sus etiquetas y predicciones
plt.figure(figsize=(12, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    title_color = "green" if preds[i].item() == labels[i].item() else "red"
    plt.title(f"Real: {labels[i].item()}\nPrediccion: {preds[i].item()}", color=title_color)
    plt.axis('off')
plt.tight_layout()
plt.show()