# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
 

### STEP 1: Define the problem
Classify handwritten digits (0–9) using the MNIST dataset.


### STEP 2:Import libraries and dataset
Import required libraries such as TensorFlow/Keras, NumPy, and Matplotlib. Load the MNIST dataset using keras.datasets.mnist.load_data().



### STEP 3:Preprocess the data
Normalize the image pixel values (scale from 0-255 to 0-1). Reshape the images to match CNN input shape.



### STEP 4:Build the CNN model
Initialize a Sequential model. Add convolutional layers with activation (ReLU), followed by pooling layers. Flatten the output and add Dense layers. Use a softmax layer for classification.




### STEP 5:Compile and train the model
Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and metrics (accuracy). Train the model using training data and validate using validation split or test data.




### STEP 6:Evaluate and visualize results
Evaluate the model on test data and print accuracy. Plot training/validation loss and accuracy curves. Optionally, display a confusion matrix or sample predictions.



## PROGRAM

### Name: LOKESH R

### Register Number: 212222240055

```python
class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

import time
start_time = time.time()
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []
for i in range(epochs):

    trn_corr = 0
    tst_corr = 0
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)


        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b%600 == 0:
            print(f'epoch: {i}  batch: {b} loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')

```

### OUTPUT


![Screenshot 2025-04-24 091055](https://github.com/user-attachments/assets/74d07cc1-4654-4f2f-998f-2c818ca7cf36)


![Screenshot 2025-04-24 091045](https://github.com/user-attachments/assets/cf5b6dab-f171-4213-a6f5-a06c83d031f6)


![Screenshot 2025-04-24 091032](https://github.com/user-attachments/assets/f92a006f-7557-4055-be75-bb54cd07ee39)


![Screenshot 2025-04-24 091026](https://github.com/user-attachments/assets/d4cb8a43-e1ca-462d-b39a-ff9bc464ac39)



## RESULT
Thus the program for image classification using Convulutional Nueral Networks is implemented successfully
