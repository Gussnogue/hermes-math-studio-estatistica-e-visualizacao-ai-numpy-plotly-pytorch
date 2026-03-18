import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def pytorch_demo():
    st.header("🧠 Módulo PyTorch - Redes Neurais")
    st.markdown("Treine uma rede neural simples em datasets clássicos ou dados próprios.")

    # Escolha do dataset
    dataset_choice = st.selectbox("Dataset", ["MNIST", "CIFAR-10", "Upload CSV (regressão)"])

    # Parâmetros de treinamento
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Épocas", min_value=1, max_value=100, value=3)
    with col2:
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64)
    with col3:
        lr = st.number_input("Taxa de aprendizado", value=0.001, format="%.4f")

    if st.button("Iniciar Treinamento"):
        with st.spinner("Treinando... (pode levar alguns minutos)"):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            st.write(f"Dispositivo: {device}")

            if dataset_choice == "MNIST":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
                testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
                num_classes = 10
                input_size = 28*28

            elif dataset_choice == "CIFAR-10":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ])
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
                num_classes = 10
                input_size = 3*32*32

            else:  # Upload CSV
                uploaded = st.file_uploader("Carregue CSV (última coluna é target)", type="csv")
                if uploaded is None:
                    st.warning("Aguardando upload...")
                    return
                df = pd.read_csv(uploaded)
                st.write(df.head())
                # Assume que a última coluna é target
                X = df.iloc[:, :-1].values.astype(np.float32)
                y = df.iloc[:, -1].values.astype(np.float32)
                # Normalizar
                mean, std = X.mean(axis=0), X.std(axis=0)
                std[std==0] = 1
                X = (X - mean) / std
                # Converter para tensores
                X_t = torch.tensor(X)
                y_t = torch.tensor(y).view(-1, 1)
                # Criar dataset
                trainset = torch.utils.data.TensorDataset(X_t, y_t)
                testset = trainset  # simplificação
                num_classes = 1
                input_size = X.shape[1]

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

            # Definir rede
            class Net(nn.Module):
                def __init__(self, input_size, num_classes):
                    super().__init__()
                    if num_classes == 1:
                        self.fc = nn.Sequential(
                            nn.Linear(input_size, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1)
                        )
                    else:
                        self.fc = nn.Sequential(
                            nn.Linear(input_size, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, num_classes)
                        )

                def forward(self, x):
                    return self.fc(x)

            net = Net(input_size, num_classes).to(device)

            if num_classes == 1:
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            # Treinamento
            loss_history = []
            for epoch in range(epochs):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.view(inputs.size(0), -1).to(device)
                    labels = labels.to(device)
                    if num_classes == 1:
                        labels = labels.float().view(-1,1)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                avg_loss = running_loss / len(trainloader)
                loss_history.append(avg_loss)
                st.write(f"Época {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Avaliação
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    inputs = inputs.view(inputs.size(0), -1).to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    if num_classes == 1:
                        predicted = outputs
                        # Não calcula acurácia para regressão
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            if num_classes > 1:
                st.success(f"Acurácia no teste: {100*correct/total:.2f}%")

            # Plot da perda
            fig, ax = plt.subplots()
            ax.plot(loss_history)
            ax.set_xlabel("Época")
            ax.set_ylabel("Perda")
            ax.set_title("Curva de Treinamento")
            st.pyplot(fig)

            # Mostrar algumas predições para MNIST
            if dataset_choice == "MNIST":
                dataiter = iter(testloader)
                images, labels = next(dataiter)
                images_flat = images.view(images.size(0), -1).to(device)
                outputs = net(images_flat)
                _, predicted = torch.max(outputs, 1)
                fig, axes = plt.subplots(3,5, figsize=(10,6))
                for idx, ax in enumerate(axes.flat):
                    ax.imshow(images[idx].squeeze(), cmap='gray')
                    ax.set_title(f"P:{predicted[idx].cpu()} / R:{labels[idx]}")
                    ax.axis('off')
                st.pyplot(fig)


                