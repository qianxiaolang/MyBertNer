import tensorflow as tf

class MyLstm(object):
    def __init__(self,units=120):
        self.units=units
    def encoder(self,inputs,length):
        cell=tf.nn.rnn_cell.LSTMCell(num_units=self.units)
        zero_state=cell.zero_state(tf.shape(inputs)[0],dtype=tf.float32)
        lstm_out,state=tf.nn.dynamic_rnn(cell,inputs,sequence_length=length,initial_state=zero_state)
        return lstm_out