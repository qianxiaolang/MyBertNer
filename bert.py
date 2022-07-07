import tensorflow as tf

class Transformer(object):
    def __init__(self,char_num,char_embedding_size,seg_num,seg_embedding_size):
        self.char_num=char_num
        self.char_embedding_size=char_embedding_size
        self.seg_num=seg_num
        self.seg_embedding_size=seg_embedding_size
        self.char_embedding=tf.get_variable(name='char_embedding',shape=(self.char_num,self.char_embedding_size),dtype=tf.float32
                                            ,initializer=tf.random_normal_initializer(stddev=0.1))

        self.seg_embedding=tf.get_variable(name='seg_embedding',shape=(self.seg_num,self.seg_embedding_size),dtype=tf.float32
                                           ,initializer=tf.random_normal_initializer(stddev=0.1))

        self.positon=tf.get_variable(name='position',shape=[1024,self.seg_embedding_size+self.char_embedding_size],dtype=tf.float32
                                     ,initializer=tf.random_normal_initializer(stddev=0.1))

    def get_position(self):
        t=10000**(tf.range(0,self.seg_embedding_size+self.char_embedding_size,delta=2,dtype=tf.float32)/(self.seg_embedding_size+self.char_embedding_size))
        t=1/t
        pos=tf.range(0,1024,dtype=tf.float32)
        pt=tf.einsum('i,j->ij',pos,t)
        return tf.concat([tf.sin(pt),tf.cos(pt)],axis=-1)

    def scaled_dot_product_attention(self,q,k,v,key_masks,dropuout_rate=0.1,casuality=False):
        with tf.variable_scope('scaled_dot_product_attention',reuse=tf.AUTO_REUSE):
            d_k=tf.to_float(tf.shape(q)[-1])
            outputs=tf.einsum('imk,ijk->imj',q,k)
            outputs/=d_k**0.5
            #下面进行pad_mask的操作
            key_masks=tf.to_float(key_masks)
            key_masks=key_masks*(-2**31+1)
            key_masks=tf.tile(key_masks,multiples=[tf.shape(outputs)[0]//tf.shape(key_masks)[0],1])
            key_masks=tf.expand_dims(key_masks,axis=1)
            outputs+=key_masks

            if casuality:
                #这里需要补充
                print(666)

            outputs=tf.nn.softmax(outputs,axis=-1)
            outputs=tf.layers.dropout(outputs,rate=dropuout_rate)
            outputs=tf.einsum('ijm,imv->ijv',outputs,v)

            return outputs

    def ln(self,inputs, epsilon=1e-8, scope='ln'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self,queries,keys,values,key_masks,num_head=5,head_size=50,dropout_rate=0.1,casuality=False):
        with tf.variable_scope('multihead_attention',reuse=tf.AUTO_REUSE):
            Q=tf.layers.dense(queries,num_head*head_size,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            K=tf.layers.dense(keys,num_head*head_size,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            V=tf.layers.dense(values,num_head*head_size,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))

            q=tf.concat(tf.split(Q,num_or_size_splits=num_head,axis=2),axis=0)
            k=tf.concat(tf.split(K,num_or_size_splits=num_head,axis=2),axis=0)
            v=tf.concat(tf.split(V,num_or_size_splits=num_head,axis=2),axis=0)
            outputs=self.scaled_dot_product_attention(q,k,v,key_masks)

            outputs=tf.concat(tf.split(outputs,num_or_size_splits=num_head,axis=0),axis=-1)

            outputs=tf.layers.dense(outputs,self.char_embedding_size+self.seg_embedding_size,use_bias=True)

            outputs+=queries

            outputs=self.ln(outputs)

            return outputs

    def ff(self,inputs):
        with tf.variable_scope('ff',reuse=tf.AUTO_REUSE):
            outputs=tf.layers.dense(inputs,units=2*(self.seg_embedding_size+self.char_embedding_size),activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            outputs=tf.layers.dense(outputs,units=self.seg_embedding_size+self.char_embedding_size,activation=None,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            outputs+=inputs
            outputs=self.ln(outputs)
            return outputs


    def encoder(self,char_input,seg_input,length,dropout=0.2,num_blocks=2):
        char=tf.nn.embedding_lookup(self.char_embedding,char_input)
        seg=tf.nn.embedding_lookup(self.seg_embedding,seg_input)
        all_embedding=tf.concat([char,seg],axis=-1)
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            all_embedding=tf.layers.dropout(all_embedding,dropout)

            for i in range(num_blocks):
                with tf.variable_scope('num_block_{}'.format(i),reuse=tf.AUTO_REUSE):
                    all_embedding=self.multihead_attention(queries=all_embedding,keys=all_embedding,values=all_embedding,key_masks=1-tf.to_float(tf.sequence_mask(length)))

                    #下面进行全连接操作
                    all_embedding=self.ff(all_embedding)

            return all_embedding