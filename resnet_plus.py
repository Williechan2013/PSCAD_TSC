'''
The is the Residual Network model included in the study by...
        Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4), 917-963.

I have removed callbacks and fit() method.
These features are implemented in ModelTraining.py

This version of ResNet includes various options to modify the model architecture

~~~ Nathan O'Sullivan Oct 2019 ~~~
'''
# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras

class Classifier_RESNET: 

        def __init__(self, output_directory, input_shape, nb_classes, verbose=False,
                     extraConv=False, extraDense=False, lessConv=False, dropout=0.0, L2norm=0.0):

                self.model = self.build_model(input_shape, nb_classes, extraConv, extraDense, lessConv, dropout, L2norm)
                if(verbose==True):
                        self.model.summary()

        def build_model(self, input_shape, nb_classes, extraConv=False, extraDense=False, lessConv=False, dropout=0.0, L2norm=0.0):
                n_feature_maps = 64

                input_layer = keras.layers.Input(input_shape)
                
                # BLOCK 1 
                conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
                conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
                conv_x = keras.layers.Activation('relu')(conv_x)

                conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
                conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
                conv_y = keras.layers.Activation('relu')(conv_y)

                conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
                conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

                # expand channels for the sum 
                shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
                shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

                output_block_1 = keras.layers.add([shortcut_y, conv_z])
                output_block_1 = keras.layers.Activation('relu')(output_block_1)

                # BLOCK 2
                if (lessConv == True):
                        output_block_2 = output_block_1
                else:
                        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
                        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
                        conv_x = keras.layers.Activation('relu')(conv_x)

                        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
                        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
                        conv_y = keras.layers.Activation('relu')(conv_y)

                        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
                        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

                        # expand channels for the sum 
                        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
                        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

                        output_block_2 = keras.layers.add([shortcut_y, conv_z])
                        output_block_2 = keras.layers.Activation('relu')(output_block_2)

                # BLOCK X
                if (extraConv == True):
                        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
                        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
                        conv_x = keras.layers.Activation('relu')(conv_x)

                        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
                        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
                        conv_y = keras.layers.Activation('relu')(conv_y)

                        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
                        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

                        # no need to expand channels because they are equal
                        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

                        output_block_x = keras.layers.add([shortcut_y, conv_z])
                        output_block_2 = keras.layers.Activation('relu')(output_block_x)

                # BLOCK 3 
                conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
                conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
                conv_x = keras.layers.Activation('relu')(conv_x)

                conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
                conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
                conv_y = keras.layers.Activation('relu')(conv_y)

                conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
                conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

                if (lessConv == True) and (extraConv == False):
                        # expand channels for the sum 
                        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_2)
                        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
                else:
                        # no need to expand channels because they are equal 
                        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

                output_block_3 = keras.layers.add([shortcut_y, conv_z])
                output_block_3 = keras.layers.Activation('relu')(output_block_3)

                # FINAL 
                gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

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

                model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), 
                        metrics=['accuracy'])

                return model
        

