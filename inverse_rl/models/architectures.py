import tensorflow as tf
from inverse_rl.models.tf_util import relu_layer, linear, tanh_layer, mod_linear


def make_relu_net(layers=2, dout=1, d_hidden=32):
    def relu_net(x, last_layer_bias=True):
        out = x
        for i in range(layers):
            out = relu_layer(out, dout=d_hidden, name='l%d'%i)
        out = linear(out, dout=dout, name='lfinal', bias=last_layer_bias)
        return out
    return relu_net


def relu_net(x, layers=2, dout=1, d_hidden=32):
    print("Using Relu Net")
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d'%i)
    out = linear(out, dout=dout, name='lfinal')
    return out

def reward_net(x, layers=2, dout=1, d_hidden=25):
    print("Using Reward Net")
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d'%i)
    #out = linear(out, dout=dout, name='lfinal')
    out = tanh_layer(out,dout=dout,name='lfinal')
    return out

def sigmoid_net(x):
    shift = tf.get_variable('shift', shape=(1))
    a = x-shift
    steepness = tf.get_variable('steepness', shape=(1))
    b = steepness*a
    translate = tf.get_variable("translate",shape=(1))
    c = tf.where(x < 0, translate*tf.ones(tf.shape(x)), tf.sigmoid(b)+translate)
    d = -1.*c
    return d

def sigmoid_gcl_net(x,dout=1):
    shift = tf.get_variable('shift', shape=(dout))
    a = x-shift
    steepness = tf.get_variable('steepness', shape=(1))
    b = steepness*a
    translate = tf.get_variable("translate",shape=(1))
    c = tf.where(x < 0, translate*tf.ones(tf.shape(x)), 10*tf.sigmoid(b)+translate)
    d = -1.*c
    return d

def sigmoid_airl_net(x,dout=1):
    shift = tf.get_variable('shift', shape=(dout))
    a = x-shift
    steepness = tf.get_variable('steepness', shape=(1))
    b = steepness*a
    translate = tf.get_variable("translate",shape=(1))
    c = tf.where(x < 0, translate*tf.ones(tf.shape(x)), 10*tf.sigmoid(b)+translate)
    return c


def linear_net(x, dout=1):
    print("Using Linear Net")
    out = x
    out = linear(out, dout=dout, name='lfinal')
    return out

def neg_linear_net(x, dout=1):
    print("Using Linear Net")
    out = x
    out = linear(out, dout=dout, name='lfinal',bias=False)
    out = -1*out
    return out

def airl_linear_net(x, dout=1):
    print("Using Linear Net")
    out = x
    out = mod_linear(out, dout=dout, name='lfinal')
    return out


def gcl_linear_net(x, dout=1):
    print("Using Linear Net")
    out = x
    out = mod_linear(out, dout=dout, name='lfinal')
    out = -1*out
    return out


def feedforward_energy(obs_act, ff_arch=relu_net):
    # for trajectories, using feedforward nets rather than RNNs
    dimOU = int(obs_act.get_shape()[2])
    orig_shape = tf.shape(obs_act)

    obs_act = tf.reshape(obs_act, [-1, dimOU])
    outputs = ff_arch(obs_act) 
    dOut = int(outputs.get_shape()[-1])

    new_shape = tf.stack([orig_shape[0],orig_shape[1], dOut])
    outputs = tf.reshape(outputs, new_shape)
    return outputs


def rnn_trajectory_energy(obs_act):
    """
    Operates on trajectories
    """
    # for trajectories
    dimOU = int(obs_act.get_shape()[2])

    cell = tf.contrib.rnn.GRUCell(num_units=dimOU)
    cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    outputs, hidden = tf.nn.dynamic_rnn(cell_out, obs_act, time_major=False, dtype=tf.float32)
    return outputs

def maze_net_2dim(x, dout=1):
    dimOU = 2#int(x.get_shape()[1])
    shift = tf.get_variable('shift', shape=(dimOU))
    shift2 = tf.concat([shift,tf.zeros((1,))],axis=0)
    a = x - shift2
    b = -1*tf.norm(a, axis=1, keepdims=True)
    return b

def maze_net_2dim_gcl(x, dout=1):
    dimOU = 2#int(x.get_shape()[1])
    shift = tf.get_variable('shift', shape=(dimOU))
    shift2 = tf.concat([shift,tf.zeros((1,))],axis=0)
    a = x - shift2
    b = tf.norm(a, axis=1, keepdims=True)
    return b