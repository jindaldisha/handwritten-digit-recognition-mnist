# Handwritten Digit Recognition MNIST

We'll solve an image classification problem. The dataset used is the `MNIST Handwritten Digits Database`. It consists of `28px by 28px` grayscale images of handwritten digits (0 to 9) and labels for each image indicating which digit it represents. We will use a feed forward neural netowrk to do that. This model will be able to capture the non-linear relation between the inputs and targets.


The dataset consists of tuple consisting of a 28x28 image and its label. Originally the image are an object of class `PIL.Image.Image`, which is a part of the Python imaging library Pillow. We need to convert the images into tensors as PyTorch doesn't work with images. We can do this by specifying a transform while creating our dataset.

```Python
#Download training and test dataset (it is going to download inside 'data' directory and creates a PyTorch Dataset)
dataset = MNIST(root='data/', 
                download = True, 
                train = True, 
                transform = transforms.ToTensor())
test_dataset = MNIST(root='data/', 
                     train=False, 
                     transform = transforms.ToTensor())
```

The training dataset has `60,000` images that we'll use to train and evaluate our model. And the testing dataset has `10,000` images that we'll use for testing purposes.

The images have been converted to a `1x28x28` tensors. The first dimension tracks the color channels. The second and third dimension represents pixels along the height and width of the image respectively. Since the dataset we're using has only grayscale image, there is only one color channel.

The pixel values range from `0` to `1`, representing black and white respectively. And the value in between different shades of grey.

A few sample images of our dataset:

![beforepreprocessing](https://user-images.githubusercontent.com/53920732/129560476-32438702-f327-4572-91c0-37afea2387bf.png)

## Training and Validation Datasets


While building real-world machine learning models, it is quite common to split the dataset into three parts:

1. **Training set** - used to train the model, i.e., compute the loss and adjust the model's weights using gradient descent.
2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc.), and pick the best version of the model.
3. **Test set** - used to compare different models or approaches and report the model's final accuracy.

In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images. 

Since there's no predefined validation set, we must manually split the 60,000 images into training and validation datasets. Let's set aside 10,000 randomly chosen images for validation. We can do this using the `random_spilt` method from PyTorch.

It is essential to choose a random sample for creating a validation set. Training data is often sorted by target labels. If we create a validation set using consecutive images, it would consist of similar images. Such training and validation datasets would make it impossible to train a useful model.

```Python
#Spliting dataset into training and validation sets
validation_size = 1000
train_size = len(dataset) - validation_size

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
len(train_dataset), len(validation_dataset)
```

When working with large datasets, its not possible to train the entire dataset at once as it may not fit into the memory and even if it does, the entire process will be very slow. And therefore what we do instead is take the dataset and break it into batches and train our model batch by batch.

We'll also create a `DataLoader`, which can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.

We can set the `shuffle = True` in `DataLoader`. This helps randomize the input to optimization algorithm, leading to a faster reduction in loss. This also helps in generalization i.e. it helps in improving the performace of the model on data it has never seen before.

```Python
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size, num_workers=2, pin_memory=True) #validation_dataset is already shuffled
```
A batch of images look as follows:

![batch](https://user-images.githubusercontent.com/53920732/129560629-d081d13c-01fa-494a-9508-2ab5b307654c.png)


## Hidden Layers, Activation Functions and Non-Linearity

We'll create a neural network with two layers: a hidden layer and an output layer. There will also be an activation function between the two layers.

Firstly, we'll flatten the `1x28x28` images into vectors of size `28*28 = 784`, so that they can be passed into an nn.Linear layer.

The `nn.Linear` layer will server as our hidden layer. We can calculate intermediate outputs for the batch of images by passing input through it.
```Python
layer1 = nn.Linear(input_size, hidden_size)
#Intermediate outputs through layer1
layer1_outputs = layer1(inputs)
```
```Python
print('Input Shape: ', inputs.shape)
print('Layer1 Weights Transform Shape:', layer1.weight.t().shape)
print('Layer1 Biases Shape: ', layer1.bias.shape)
print('Layer1 Output Shape: ', layer1_outputs.shape)
```
```
Input Shape:  torch.Size([128, 784])
Layer1 Weights Transform Shape: torch.Size([784, 32])
Layer1 Biases Shape:  torch.Size([32])
Layer1 Output Shape:  torch.Size([128, 32])
```

Since we use the formula `y = x @ w.t() + b`, and the hidden output size is `32`, we get an output shape of `(128,32)`. The image vectors of size `784` are transformed into intermediate output vectors of length `32` by performing a matrix multiplication of inputs matrix with the transposed weights matrix of layer1 and adding the bias.

Also the `layer1_outputs` and `inputs` have a linear relation. Each element in `layer_outputs` is a weighted sum of elements from `inputs`. Thus, even as we train the model and modify the weights, `layer1` can only capture linear relation between `inputs` and `outputs`.


We'll use the Rectified Linear Unit (ReLU) function as an activation function for the outputs to add non-linearity to our model. It has the formula `relu(x) = max(0, x)`. We replace the negative values in a given tensor with `0`.

```Python
relu_outputs = F.relu(layer1_outputs)
```
After appling `ReLU` activation function to our inputs, our new outputs `relu_outputs` and `inputs` no longer have a linear relation. `ReLU` is called an activation function because for each input, certain outputs are activated (non-zero values) and others are turned off (zero values).


Next step is create an output layer that will convert the current vector output `relu_outputs` of length `hidden_size` into vectors of length 10, since that is the number of target labels that we have.

```Python
#Define output layer
layer2 = nn.Linear(hidden_size, output_size)
layer2_outputs = layer2(relu_outputs)
print('Layer2 Outputs Shape: ', layer2_outputs.shape)
```

Now we can use outputs from the output layer and calculate loss and then use gradient descent to adjust the weights of layer1 and layer2.

```Python
#Loss function
loss = F.cross_entropy(layer2_outputs, labels)
print('Loss: ', loss.item())
```

In summary, our model transforms `inputs` into `layer2_outputs` by first applying a linear transformation in `layer 1` followed by the non linear activation function `relu` and then followed by another linear transformation in `layer 2`. Since the outputs and inputs do not have a linear relation because of the non-linear activation function, as we train the model, we can now capture the non-linear relationship between the images and their labels. 

## Model

We'll create a neural network with one hidden layer.

* We'll use two `nn.Linear` objects. Each of these is called a _layer_ in the network. 

* The first layer (also known as the hidden layer) will transform the input matrix of shape `batch_size x input_size` into an intermediate output matrix of shape `batch_size x hidden_size`. The parameter `hidden_size` can be configured manually (e.g., 32 or 64).

* We'll then apply a non-linear *activation function* to the intermediate outputs. The activation function transforms individual elements of the matrix.

* The result of the activation function, which is also of size `batch_size x hidden_size`, is passed into the second layer (also known as the output layer).  The second layer transforms it into a matrix of size `batch_size x output_size`. We can use this output to compute the loss and adjust weights using gradient descent.

- In `__init__` constructor, we instantiate weights and biases using nn.Linear for hidden and output layer.
- Our images are of the shape 1x28x28, but we need them to be vectors of size 784, i.e., we need to flatten them. Inside `forward` method, we flatten the input image tensors and pass it to `self.layer1`.
- Then we pass the outputs from layer1 to the relu activation function.
- The outputs from the activation function are then passed on to the output layer.
- In `training_step` method, we first generate predictions and then using these predictions and true labels, we calculate the loss.
- In `validation_step` method, we first generate predictions and then using these predictions and true labels, we calculate loss and accuracy.
- In `validation_epoch_end` method, we take the mean of accuracy and loss for an epoch.
- In `epoch_end` method, we simply print the average accuracy and loss for an epoch that we calculated in `validation_epoch_end`.

```Python
class Model(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    #Hidden Layer
    self.layer1 = nn.Linear(input_size, hidden_size)
    #Output Layer
    self.layer2 = nn.Linear(hidden_size, output_size)
  
  def forward(self, xb):
    #Flatten the image tensors
    xb = xb.view(xb.size(0), -1)
    #Hidden layer outputs
    output = self.layer1(xb)
    #Activation Function
    output = F.relu(output)
    #Output Layer
    output = self.layer2(output)
    return output
  
  def training_step(self, batch):
    images, labels = batch
    #generate predictions
    output = self(images)
    #calculate loss
    loss = F.cross_entropy(output, labels)
    return loss

  def validation_step(self, batch):
    images, labels = batch
    #generate predictions
    output = self(images)
    #calculate loss
    loss = F.cross_entropy(output, labels)
    #calculate accuracy
    acc = accuracy(output, labels)
    return {'val_loss': loss, 'val_acc': acc}
  
  def validation_epoch_end(self, outputs):
    batch_loss = [x['val_loss'] for x in outputs]
    #Mean Loss
    epoch_loss = torch.stack(batch_loss).mean()
    batch_acc = [x['val_acc'] for x in outputs]
    #Mean Accuracy
    epoch_acc = torch.stack(batch_acc).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
  
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
```

```Python
#accuracy
def accuracy(outputs, labels):
  probs, preds = torch.max(outputs, dim=1)
  num_of_equals = torch.sum(preds == labels).item()
  num_of_elements = len(preds)
  acc = torch.tensor(num_of_equals/num_of_elements)
  return acc
```
## Training the Model

So far we've created:
- Data loaders
- The model
- Loss Function

Now the next step is to train the model. We have a training phase and a validation phase for every epoch.

Steps (for every epoch):
- Training Phase (for every batch)
  - Load the batch
    - Generate predictions
    - Calculate loss
    - Compute gradients
    - Update weights
    - Reset gradients
- Validation Phase (for every batch)
  - Load the batch
    - Generate predictions
    - Calculate loss
    - Calculate metrics (accuracy, etc.)
- Calculate  average validation loss and metrics

- Log epoch, loss and metrics for inspection


The fit function records the validation loss and metric from each epoch. It returns a history of the training, useful for debugging & visualization.

Configurations like batch size, learning rate, etc. (called hyperparameters), need to picked in advance while training machine learning models. Choosing the right hyperparameters is critical for training a reasonably accurate model within a reasonable amount of time.

```Python
#evalute function returns the average loss and everage accuracy for validation dataset.
def evaluate(model, validation_loader):
  outputs = [model.validation_step(batch) for batch in validation_loader]
  return model.validation_epoch_end(outputs)
```

```Python
#function to fit the model to data
def fit(epochs, lr, model, train_loade, validation_loader, opt_func=torch.optim.SGD):
  #define optimizer
  optimizer = opt_func(model.parameters(), lr)
  history = [] #record result every epoch

  for epoch in range(epochs):
    #Training phase
    for batch in train_loader:
      #calculate loss
      loss = model.training_step(batch)
      #compute gradients
      loss.backward()
      #update gradients
      optimizer.step()
      #reset gradients to zero
      optimizer.zero_grad()
    
    #Validation phase
    result = evaluate(model, validation_loader)
    model.epoch_end(epoch, result)
    history.append(result)
  return history

```

**Summary:**

- When we call the fit function to fit the model to the data, for each epoch, we are getting batches from training dataloader `train_loader` and validation data loader `validation_loader`. These batches contain a set of images and a set of labels. We pass them onto the model. 
- The model then evokes the `forward` function. Which first reshapes the images into the shape `(batch_size, input_size)`. Then this batch of reshaped images is passed into the the layers that we've defined in the `__init__` constructor. In this case we have one hidden layer and one output layer with an activation function between them.  
- Since we use the formula `y = x @ w.t() + b`, and the hidden output size is `32`, we get an output shape of `(128,32)`. The image vectors of size `784` are transformed into intermediate output vectors of length `32` by performing a matrix multiplication of inputs matrix with the transposed weights matrix of layer1 and adding the bias. Also the `layer1_outputs` and `inputs` have a linear relation. Each element in `layer_outputs` is a weighted sum of elements from `inputs`. Thus, even as we train the model and modify the weights, `layer1` can only capture linear relation between `inputs` and `outputs`. 
- We use the Rectified Linear Unit (ReLU) function as an activation function for the outputs to add non-linearity to our model. It has the formula `relu(x) = max(0, x)`. We replace the negative values in a given tensor with `0`. After appling `ReLU` activation function to our inputs, our new outputs and `inputs` no longer have a linear relation. 
- Next step is create an output layer that will convert the current vector output of length `hidden_size` into vectors of length 10, since that is the number of target labels that we have.
- We then take these predicted values and compare them with the true labels by first performing `softmax`, which basically converts the predicted values to probabilities. Then we use `cross-entropy`. It is the negative logarithm of the predicted probability of the correct label averaged over all training examples. It just means the predicted probability of the correct label is e^(-loss). The lower the loss, the higher thr probability of the correct label and the better the model.
- Then in order to reduce the loss, we perform `gradient descent`. And then update the weights and biases to improve the model. 
- Then we load the `validation_dataloader` and run it throught the model batch by batch to get the losses and accuracies. 
-We take the mean of all the losses and accuracies for each epoch to evaluate how our model is performing.

## Training

![training](https://user-images.githubusercontent.com/53920732/129561543-a7f8dcc5-59e3-4ee2-96e3-3a2b3700d318.png)


We get an accuracy of around `91%`.

## Testing

The examples with green title were correctly predicted and the examples with red title were wrongly predicted.

![testing](https://user-images.githubusercontent.com/53920732/129561650-78e846c7-b625-4b43-8d5d-9d846874859a.png)
