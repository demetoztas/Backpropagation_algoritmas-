import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


train_file_path = 'carpim_tablosu_teksayilar.csv'
data_train = pd.read_csv(train_file_path)

test_file_path = 'carpim_tablosu_test.csv'
data_test = pd.read_csv(test_file_path)


X_train = data_train[['input1', 'input2']].values
Y_train = data_train['output1'].values.reshape(-1, 1)

X_test = data_test[['input1', 'input2']].values
Y_test = data_test['output1'].values.reshape(-1, 1)


scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
Y_train = scaler_Y.fit_transform(Y_train)

X_test = scaler_X.transform(X_test)
Y_test = scaler_Y.transform(Y_test)

input_size = 2        
hidden_size = 8       
output_size = 1       
learning_rate = 0.5 

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(A):
    return A * (1 - A)



def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2  
    return Z1, A1, Z2, A2


def backward_propagation(X, Y, Z1, A1, Z2, A2):
    global W1, b1, W2, b2

    m = X.shape[0]
    error = A2 - Y  

    
    dW2 = np.dot(A1.T, error) / m
    db2 = np.sum(error, axis=0) / m

    dA1 = np.dot(error, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m

    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


def train_model(X_train, Y_train, epochs):
    losses = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X_train)
        backward_propagation(X_train, Y_train, Z1, A1, Z2, A2)
        

        loss = np.mean((A2 - Y_train) ** 2)
        losses.append(loss)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    return losses


print("Model egitiliyor...")
train_model(X_train, Y_train, epochs=25000)

print("Model test ediliyor...")
_, _, _, predictions = forward_propagation(X_test)


predictions = scaler_Y.inverse_transform(predictions)
Y_test = scaler_Y.inverse_transform(Y_test)


for i in range(len(X_test)):
    print(f"Giris: {scaler_X.inverse_transform([X_test[i]])[0]}, Gercek: {Y_test[i][0]:.2f}, Tahmin: {predictions[i][0]:.2f}")
