import os
import pickle

import numpy as np
import tensorflow as tf
from loader import load_sentences,updata_tag_scheme,char_mapping,tag_mapping,prepare_dataset
from dataUtil import DataManager
from tensorflow.contrib.crf import crf_log_likelihood,crf_decode
from bert import Transformer
from lstm import MyLstm
train_sentences=load_sentences('data/example.train')
# tag_scheme为BI0,BIOES
tag_scheme='bioes'

updata_tag_scheme(train_sentences,tag_scheme)


if os.path.isfile('char.pkl'):
    with open('char.pkl','rb') as f:
        char2id,id2char=pickle.load(f)
else:
    dico, char2id, id2char = char_mapping(train_sentences)
    with open('char.pkl', 'wb') as f:
        pickle.dump([char2id,id2char],f)

if os.path.isfile('tag.pkl'):
    with open('tag.pkl','rb') as f:
        tag2id,id2tag=pickle.load(f)
else:
    tag2id, id2tag = tag_mapping(train_sentences)
    with open('tag.pkl','wb') as f:
        pickle.dump([tag2id,id2tag],f)

train_data=prepare_dataset(train_sentences,char2id,tag2id)

# 设置一个batch_size
batch_size=64
train_manager=DataManager(train_data,batch_size)
char_num=len(char2id)
char_embedding_size=100
seg_num=4
seg_embedding_size=20

# 构建网络
char_input=tf.placeholder(dtype=tf.int32,shape=[None,None])
seg_input=tf.placeholder(dtype=tf.int32,shape=[None,None])

targets=tf.placeholder(dtype=tf.int32,shape=[None,None])

length=tf.reduce_sum(tf.sign(char_input),axis=-1)

# 构建embedding层
t=Transformer(char_num,char_embedding_size,seg_num,seg_embedding_size)
fc_output=t.encoder(char_input,seg_input,length,num_blocks=1)
l=MyLstm(char_embedding_size+seg_embedding_size)
fc_output=l.encoder(fc_output,length)

fc_output=tf.layers.dense(fc_output,13,activation=None,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
mask=tf.cast(tf.sequence_mask(length),dtype=tf.float32)

loss_fc=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=fc_output)


loss_lstm=tf.reduce_sum(tf.einsum('ij,ij->ij',loss_fc,mask))
exist_word=tf.reduce_sum(mask)

loss_fc=loss_lstm/exist_word

lstm_acc=tf.cast(tf.argmax(fc_output,axis=-1),dtype=tf.int32)

lstm_acc=tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(lstm_acc,targets),tf.sequence_mask(length)),dtype=tf.float32))
lstm_exist_word=tf.reduce_sum(tf.cast(tf.sequence_mask(length),dtype=tf.float32))
lstm_acc=lstm_acc/lstm_exist_word

# 下面开始构建CRF损失
small=-1000
logits=tf.concat([fc_output,small*tf.ones(shape=[tf.shape(char_input)[0],tf.shape(char_input)[1],1],dtype=tf.float32)],axis=-1)
start_logits=tf.concat([small*tf.ones(shape=[tf.shape(char_input)[0],1,13],dtype=tf.float32),tf.zeros(shape=[tf.shape(char_input)[0],1,1],dtype=tf.float32)],axis=-1)
logits=tf.concat([start_logits,logits],axis=1)

new_targets=tf.concat([len(tag2id)*tf.ones(shape=[tf.shape(char_input)[0],1],dtype=tf.int32),targets],axis=-1)


log_likelihood,ret_trans=crf_log_likelihood(inputs=logits,tag_indices=new_targets,sequence_lengths=length+1,transition_params=None)

viterbi_sequence,viterbi_score=crf_decode(potentials=logits,transition_params=ret_trans,sequence_length=length+1)
# 计算维特比算法的准确率

viter_acc=tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(viterbi_sequence,new_targets),tf.sequence_mask(length+1)),dtype=tf.float32))

viter_exist_word=tf.reduce_sum(tf.cast(tf.sequence_mask(length+1),dtype=tf.float32))
viter_acc=viter_acc/viter_exist_word

loss=-tf.reduce_mean(log_likelihood)

# op1=tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss_fc)
op=tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:
        for i in range(train_manager.len_data):
            data=train_manager.iter_batch_index(i)

            _ ,viter_acc_eval,v,t= sess.run([op,viter_acc,viterbi_sequence,new_targets],feed_dict={char_input: data[:, 0, :], seg_input: data[:, 1, :],
                                                           targets: data[:, 2, :]})
            print("当前批次为：{}，当前准确率为：{}".format(i,viter_acc_eval))
            print(v[3])
            print(t[3])

