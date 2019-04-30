# mnist attention
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import pandas as pd
import keras
# from keras.utils import plot_model
from keras.layers import Input, Embedding, LSTM, Dense,Bidirectional


TIME_STEPS_1 = 10
TIME_STEPS_2=3
lstm_units = 64


main_x=pd.read_csv('train_para.csv')
main_x=main_x.values

main_y=pd.read_csv('train_label.csv')
main_y=main_y.values

y_train =main_y

add_x=pd.read_csv('train_ques.csv')
add_x=add_x.values


# print(x.shape)

# first way attention
def attention_3d_block(inputs,TIME_STEPS):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# build RNN model with attention
#问题输入，LSTM+attention
main_input = Input(shape=(len(main_x[0]),), dtype='float32', name='main_input')
train_x = Embedding(output_dim=512, input_dim=len(main_x), input_length=len(main_x[0]))(main_input)
lstm_out_1 = Bidirectional(LSTM(lstm_units, return_sequences=True),merge_mode='concat')(train_x)
attention_mul = attention_3d_block(lstm_out_1,TIME_STEPS_1)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)

#另一端输入 lstm
add_x=pd.read_csv('train_ques.csv')
add_x=add_x.values
auxiliary_input = Input(shape=(len(add_x[0]),), name='aux_input')
auxiliary_input_x=Embedding(output_dim=512, input_dim=len(add_x), input_length=len(add_x[0]))(auxiliary_input)
lstm_out_aux = Bidirectional(LSTM(lstm_units, return_sequences=False), name='bilstm')(auxiliary_input_x)
drop2_aux = Dropout(0.3)(lstm_out_aux)


x = keras.layers.concatenate([drop2 , drop2_aux])
#全连接
x_out = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x_out)

model = Model(inputs=[main_input,auxiliary_input], outputs=output)

# model = Model(inputs=[main_input,auxiliary_input], outputs=[output])

adam=Adam(lr=1e-3)
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])


# plot_model(model, 'Multi_input.png')

print(model.summary())

print('Training------------')

model.fit([main_x, add_x], y_train, validation_split=0.33,epochs=5, batch_size=32,verbose=1)

n_in_timestep=3
model.save('./model/my_model_combine_timestep_LSTM%s_1000days_0429.h5' % n_in_timestep)



# print('Testing--------------')
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('test loss:', loss)
# print('test accuracy:', accuracy)
