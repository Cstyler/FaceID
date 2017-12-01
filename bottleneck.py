import numpy as np
import keras.backend as K


class Bottleneck:
    def __init__(self, model):
        self.fn = K.function([model.layers[0].input, K.learning_phase()], [
            model.get_layer('bottleneck').output])

    def predict(self, x, batch_size=32, learning_phase=0):
        n_data = len(x)
        n_batches = n_data // batch_size + \
                    (0 if n_data % batch_size == 0 else 1)
        output = None
        for i in range(n_batches):
            batch_x = x[i * batch_size:(i + 1) * batch_size]
            batch_y = self.fn([batch_x, learning_phase])[0]

            if output is None:
                output = batch_y
            else:
                output = np.vstack([output, batch_y])
        return output
