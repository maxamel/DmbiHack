import tensorflow as tf
from binaryAutoencoder import BinaryAutoEncoder

'''with open(r'smallLights.csv', 'r') as infile:
    with open(r'smallLights2.csv', 'w') as outfile:
        data = infile.read()
        data = data.replace('"', "")
        outfile.write(data)'''


batch_size = 50
num_inputs = 297
learning_rate = 0.0003
activation = tf.nn.relu
num_units = 100
iterations = 5000

model = BinaryAutoEncoder(batch_size,num_inputs,learning_rate,activation,num_units,iterations)

model.read_data('lights2.csv',test_percent=0.3)

model.build_model()

model.train_model()

model.close_session()