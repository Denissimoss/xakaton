import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Загрузка данных из файлов
movies = pd.read_csv('datasets/cosmetic_val_target.tsv', sep='\t')
ratings = pd.read_csv('datasets/cosmetic_val.tsv', sep='\t')

# Удаление ненужных столбцов
ratings.drop(['device_id', 'local_date', 'price'], axis=1, inplace=True)

# Создание user-item матрицы
user_item_matrix = ratings.pivot(index='receipt_id', columns='item_id', values='quantity')
user_item_matrix.fillna(0, inplace=True)

# Преобразование разреженной матрицы
user_item_matrix_encoded = user_item_matrix.astype(float).values

# Разделение данных на обучающий и тестовый наборы
X_train, X_test = train_test_split(user_item_matrix_encoded, test_size=0.2, random_state=42)

# Определение нейронной сети для рекомендаций
class RecommendationModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)
    
    def forward(self, x):
        embedded = self.embedding(x.long())  # Преобразование в Long
        output = torch.relu(self.fc1(embedded))
        output = self.fc2(output)
        return output

# Инициализация модели
input_dim = user_item_matrix_encoded.shape[1]
embedding_dim = 32
model = RecommendationModel(input_dim, embedding_dim)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Преобразование данных в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Обучение модели
epochs = 100
batch_size = 64  # Размер пакета
num_batches = len(X_test) // batch_size

for epoch in range(epochs):
    optimizer.zero_grad()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_X = X_test[start_idx:end_idx]
        batch_predicted = model(batch_X)
        
        if i == 0:
            all_predicted = batch_predicted
        else:
            all_predicted = torch.cat((all_predicted, batch_predicted))
    
    loss = criterion(all_predicted, X_test[:len(all_predicted)])  # Сравниваем с целевыми данными тестового набора
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Вывод результатов рекомендации
recommendations = []
for i in range(len(X_test)):
    receipt_id = i
    recommended_item_id = model(X_test[i].unsqueeze(0)).detach().numpy()
    recommendations.append((receipt_id, recommended_item_id))

# Сохранение результатов в файл
result_df = pd.DataFrame(recommendations, columns=['receipt_id', 'recommended_item_id'])
result_df.to_csv('555.tsv', sep='\t', index=False)
