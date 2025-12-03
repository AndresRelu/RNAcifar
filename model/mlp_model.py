import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron mejorado para clasificación de imágenes CIFAR-10 (4 clases)
    
    Arquitectura optimizada con regularización:
    - Dataset: 16,000 muestras de entrenamiento
    - Input: 3072 features (32x32x3 aplanado)
    - Hidden Layer 1: 512 neuronas + BatchNorm + ReLU + Dropout(0.4)
    - Hidden Layer 2: 256 neuronas + BatchNorm + ReLU + Dropout(0.4)
    - Hidden Layer 3: 128 neuronas + BatchNorm + ReLU + Dropout(0.4)
    - Output: 4 clases (airplane, automobile, ship, truck)
    
    Parámetros totales: ~500k
    """
    
    def __init__(self, input_size=3072, hidden_sizes=[512, 256, 128], num_classes=4, dropout_rate=0.4):
        super(MLP, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Primera capa
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Segunda capa
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Tercera capa
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Capa de salida
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        
    def forward(self, x):
        # Aplanar la imagen de (batch, 3, 32, 32) a (batch, 3072)
        x = x.view(x.size(0), -1)
        
        # Primera capa oculta
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Segunda capa oculta
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Tercera capa oculta
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Capa de salida
        x = self.fc4(x)
        
        return x
    
    def get_model_info(self):
        """Retorna información del modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': f'MLP: 3072 -> {self.hidden_sizes[0]} -> {self.hidden_sizes[1]} -> {self.hidden_sizes[2]} -> 4',
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate
        }