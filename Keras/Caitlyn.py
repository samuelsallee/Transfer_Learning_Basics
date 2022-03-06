import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, Flatten
from keras.optimizer_v1 import Adam
from keras.datasets import mnist, fashion_mnist
import numpy as np

tf.compat.v1.disable_eager_execution()

# creating a Conv2d model with trainable parameters
def create_trainable_model():
    model = Sequential([
        Input((28,28,1)),
        Conv2D(32, kernel_size=(3,3), activation='relu', trainable = True),
        Conv2D(32, kernel_size=(3,3), activation='relu', trainable = True),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=.001), metrics=['accuracy'])
    return model

# creating a model with frozen parameters except for the last layer
def create_transfer_model():
    model = Sequential([
        Input((28,28,1)),
        Conv2D(32, kernel_size=(3,3), activation='relu', trainable = False),
        Conv2D(32, kernel_size=(3,3), activation='relu', trainable = False),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=.001), metrics=['accuracy'])
    return model

# preparing mnist dataset
(mnist_training_inputs, mnist_training_answers),(mnist_testing_inputs, mnist_testing_answers) = mnist.load_data()

mnist_training_inputs = mnist_training_inputs/255
mnist_testing_inputs = mnist_testing_inputs/255

mnist_training_inputs = np.expand_dims(mnist_training_inputs, axis=-1)
mnist_testing_inputs = np.expand_dims(mnist_testing_inputs, axis=-1)

training_agent = create_trainable_model()

# training the agent on mnist
training_agent.fit(mnist_training_inputs, mnist_training_answers, verbose=2, epochs=4)

# making sure the training agent isn't overfit
training_agent.evaluate(mnist_testing_inputs, mnist_testing_answers, verbose=2)

training_agent.save('Caitlyn_pre_op.model')

# preparing dataset for the transfer model
(fashion_train_inputs, fashion_train_answers),(fashion_test_inputs, fashion_test_answers) = fashion_mnist.load_data()

fashion_train_inputs = fashion_train_inputs/255
fashion_test_inputs = fashion_test_inputs/255

fashion_train_inputs = np.expand_dims(fashion_train_inputs, axis=-1)
fashion_test_inputs = np.expand_dims(fashion_test_inputs, axis=-1)

transfer_agent = create_transfer_model()
transfer_agent.set_weights(training_agent.get_weights()) # grabbing the weights from the already trained model
# at this point, the transfer model has the same weights as the training model but the Convolutional layers cannot be changed

# training the output layer
transfer_agent.fit(fashion_train_inputs, fashion_train_answers, verbose=2, epochs=4)

# testing to see how well the final transfer model did
transfer_agent.evaluate(fashion_test_inputs, fashion_test_answers, verbose=2)


### Realistically, this is a poor test for transfer learning because the model is too small. This is just to learn the concept
### For a real test of transfer learning, something like 

transfer_agent.save('Caitlyn_post_op')

# just to show that the weights are still the same on each model
print(training_agent.get_weights()[0][0][0])

print(transfer_agent.get_weights()[0][0][0])