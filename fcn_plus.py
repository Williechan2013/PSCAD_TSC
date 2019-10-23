'''
The is the Fully Convolutional Network model included in the study by...
        Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4), 917-963.

I have removed callbacks and fit() method.
These features are implemented in ModelTraining.py

This version of FCN includes various options to modify the model architecture

~~~ Nathan O'Sullivan Oct 2019 ~~~
'''

# FCN
import keras 

class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,
                     extraConv=False, extraDense=False, lessConv=False, dropout=0.0, L2norm=0.0):

		self.model = self.build_model(input_shape, nb_classes, extraConv, extraDense, lessConv, dropout, L2norm)
		if(verbose==True):
			self.model.summary()

	def build_model(self, input_shape, nb_classes, extraConv=False, extraDense=False, lessConv=False, dropout=0.0, L2norm=0.0):
		input_layer = keras.layers.Input(input_shape)

		if (extraConv == True):
                        convx = keras.layers.Conv1D(filters=128, kernel_size=11, padding='same')(input_layer)
                        convx = keras.layers.normalization.BatchNormalization()(convx)
                        convx = keras.layers.Activation(activation='relu')(convx)
                else:
                        convx = input_layer

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(convx)
		conv1 = keras.layers.normalization.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

                if (lessConv == True):
                        conv2 = conv1
                else:
                        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
                        conv2 = keras.layers.normalization.BatchNormalization()(conv2)
                        conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.normalization.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

                if (dropout > 0.0) and (extraDense == True):
                        drop_layer = keras.layers.Dropout(dropout)(gap_layer)
                else:
                        drop_layer = gap_layer

                if (L2norm > 0.0):
                        extraDense_layer = keras.layers.Dense(256, activation='softmax', kernel_regularizer=keras.regularizers.l2(L2norm))
                else:
                        extraDense_layer = keras.layers.Dense(256, activation='softmax')

                if (extraDense == True):
                        outputA_layer = extraDense_layer(drop_layer)
                else:
                        outputA_layer = drop_layer

                if (dropout > 0.0):
                        drop2_layer = keras.layers.Dropout(dropout)(outputA_layer)
                else:
                        drop2_layer = outputA_layer

                if (L2norm > 0.0):
                        output_layer = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(L2norm))(drop2_layer)
                else:
                        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(drop2_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		return model 

	
