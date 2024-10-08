model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
       	activation='relu', padding='same',
       	kernel_initializer='he_normal',
   	input_shape=(300, 399, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
	# model.add(Dropout(0.3))

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
   	activation='relu', padding='same',
   	kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
	# model.add(Dropout(0.3))

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
   	activation='relu', padding='same',
   	kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
	# model.add(Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu',
   	kernel_initializer='he_normal'))
    model.add(Dropout(0.1))
    model.add(keras.layers.Dense(1, kernel_initializer='he_normal', activation='linear'))

    model.compile(loss="mean_absolute_error",
  	optimizer=keras.optimizers.Adam(learning_rate=0.000008),
  	metrics=["mae"])

    train_data_set = tf.data.Dataset.from_tensor_slices((trainX_new, trainy))
    valid_data_set = tf.data.Dataset.from_tensor_slices((validX_new, validy))
    test_data_set = tf.data.Dataset.from_tensor_slices((testX_new, testy))
    
    batch_size = 8
    train_data_set = train_data_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_data_set = valid_data_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data_set = test_data_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.9, patience=10, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True)
    
    model.summary()
    history = model.fit(train_data_set, epochs=1000,
                    	validation_data=valid_data_set,
                    	callbacks=[lr_scheduler, early_stopping])
