import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print("Iniciando...")


def plot_image(iImage, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else: 
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)       
    
    #crea una variable myvar que concatene un numero con un string
    myvar = str(iImage)  + "_" + class_names[predicted_label] +'.jpg'

    plt.savefig(myvar, format='jpeg') 
    

dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve']

main_train = metadata.splits['train'].num_examples
main_test = metadata.splits['test'].num_examples

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
BATHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(main_train).batch(BATHSIZE)
test_dataset = test_dataset.batch(BATHSIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(main_train/BATHSIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(main_test/32))


print('Accuracy on test dataset:', test_accuracy)

i=0
for test_images, test_labels in test_dataset.take(10):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    plot_image(i, predictions, test_labels, test_images)    
    i += 1

print("Fin")
