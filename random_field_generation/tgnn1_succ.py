"""
0-50:
:
: x      (x,y,t)--->h                 0-50 51 time steps    0-18 | 19-50
:                                                      19 points    32 points
:0
:  0      y
:........................0-50...
1 50 0

"""
from scipy import interpolate  # Interpolation
from tensorflow.python.client import device_lib
import io
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow as tf
import sys
sys.path.append('/home/cc/CCFs/zhuxy/tgnn design')
from makedata.make_1_data import getdata
import math
from pyDOE import lhs
import os



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # Set the random seed to 11

checkpoint_save_path = "/home/cc/CCFs/zhuxy/tgnn design/checkpoints/tgnn1/tgnn1_weights.ckpt"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Prepare TensorBoard
log_dir = '../logs/tgnn1/' + current_time  # Log file path
summary_writer = tf.summary.create_file_writer(log_dir)  # Create the log file
tf.summary.trace_on(graph=True, profiler=True)
batchsz = 5000  # Batch size
epochs = 2000
lambda1 = 1    # mse
lambda2 = 10000000000  # 10000000000 # pde
lambda3 = 10  # bc
lambda4 = 0.1  # ek
lambda5 = 100  # ic
lambda6 = 100


def realstep(inp):  # Preprocessing
    inp = inp.astype(np.float)
    inp[:, 0] = inp[:, 0]*20/1000
    inp[:, 1] = inp[:, 1]*20/1000
    inp[:, 2] = inp[:, 2]*0.2/10
    return inp


def preprocess(input, output):  # Preprocessing
    input = tf.cast(input, dtype=tf.float32)  # Convert to float32
    # output = tf.cast(output, dtype=tf.float32)
    # output = (output - nu) / seta
    output = tf.cast(output, dtype=tf.float32)
    return input, output


""" def GetConstantPdeAndK(batchsz,step,epoch):
    N_pde = 50000
    x_pde = 1 + 50 * lhs(1, N_pde)
    y_pde = 1 + 50 * lhs(1, N_pde)
    t_pde = 50 * lhs(1, N_pde)
    pde_xyt = np.hstack((x_pde, y_pde, t_pde))
    pde_xyt = realstep(pde_xyt)
    re_pde_xyt = tf.cast(pde_xyt, dtype=tf.float32)

    # 提取k值
    k_rad_xy = 0.1*np.ones((N_pde,1))
    k_rad_xy = tf.reshape(tf.cast(k_rad_xy, dtype=tf.float32), (k_rad_xy.shape[0], 1))

    dk_dx = 0*np.ones((N_pde,1))
    dk_dx = tf.reshape(tf.cast(dk_dx, dtype=tf.float32), (dk_dx.shape[0], 1))  # k关于rad_xy的偏导

    dk_dy = 0*np.ones((N_pde,1))
    dk_dy = tf.reshape(tf.cast(dk_dy, dtype=tf.float32), (dk_dy.shape[0], 1))  # k关于rad_xy的偏导
    return re_pde_xyt,k_rad_xy,dk_dx,dk_dy
 """


def GetPdeAndK(batchsz, step, epoch):
    global re_pde_xyt, k_rad_xy, dk_dx, dk_dy

    return re_pde_xyt, k_rad_xy, dk_dx, dk_dy



# Build three nested gradient tapes for [dh/dx, dh/dy, dh/dt], [d2h/dx2, d2h/dy2, d2h/dt2], and total-loss gradients with respect to trainable parameters
def LossPde(re_pde_xyt, k_rad_xy, dk_dx, dk_dy, up_nflow, dw_nflow):
    with tf.GradientTape(persistent=True) as tp0:
        tp0.watch([re_pde_xyt])  # Watch the PDE input points to calculate [d2h/dx2, d2h/dy2, d2h/dt2]

        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([re_pde_xyt])  # Watch the PDE input points to calculate [dh/dx, dh/dy, dh/dt]
            tp1.watch([up_nflow])
            tp1.watch([dw_nflow])
            logits_pde = model(re_pde_xyt)  # Evaluate the model at the PDE input points
            up_out = model(up_nflow)
            dw_out = model(dw_nflow)
        dh = tp1.gradient(logits_pde, re_pde_xyt)  # First derivatives [dh/dx, dh/dy, dh/dt]
        d_nuph = tp1.gradient(up_out, up_nflow)
        d_ndwh = tp1.gradient(dw_out, dw_nflow)

    d2h = tp0.gradient(dh, re_pde_xyt)  # Second derivatives [d2h/dx2, d2h/dy2, d2h/dt2]

    dh_dx = tf.reshape(dh[:, 0], (-1, 1))/(1000)
    dh_dy = tf.reshape(dh[:, 1], (-1, 1))/(1000)
    dh_dt = tf.reshape(dh[:, 2], (-1, 1))/(10)
    d_nuph_dx = tf.reshape(d_nuph[:, 0], (-1, 1))/(1000)
    d_ndwh_dx = tf.reshape(d_ndwh[:, 0], (-1, 1))/(1000)
    d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1))/((1000*1000))
    d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1))/((1000*1000))

    loss_pde = tf.reduce_mean(tf.square(
        0.0001 * dh_dt - dk_dx * dh_dx - k_rad_xy * d2h_dx2 - dk_dy * dh_dy - k_rad_xy * d2h_dy2))
    loss_up_dw = tf.reduce_mean(tf.reduce_mean(
        tf.square(d_nuph_dx))+tf.reduce_mean(tf.square(d_ndwh_dx)))
    del tp0
    del tp1
    return loss_pde, loss_up_dw


@tf.function
def LossBdLogits(train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt):
    logits_data = model(train_xyt)  # Evaluate the model on the training data
    logits_data = logits_data  # * seta + nu
    lbc_out = model(lbc_xyt)  # * seta + nu  # Evaluate the model on the left boundary condition data
    rbc_out = model(rbc_xyt)  # * seta + nu  # Evaluate the model on the right boundary condition data
    lic_out = model(lic_xyt)  # * seta + nu  # Evaluate the model on the left initial condition data
    ric_out = model(ric_xyt)  # * seta + nu  # Evaluate the model on the right initial condition data

    # Governing equation loss
    loss_ek = tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(
        logits_data - 1)))) + tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(0 - logits_data))))
    # Boundary condition loss
    loss_bc = tf.reduce_mean(tf.square(lbc_out - 1)) + \
        tf.reduce_mean(tf.square(rbc_out - 0))
    # Initial condition loss
    loss_ic = tf.reduce_mean(tf.square(lic_out - 1)) + \
        tf.reduce_mean(tf.square(ric_out - 0))
    # Data loss
    loss_mse = tf.reduce_mean(tf.losses.MSE(train_h, logits_data))
    return loss_ek, loss_bc, loss_ic, loss_mse


def train_one_step(step, epoch, train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt):
    global batchsz, epochs, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6
    with tf.GradientTape() as tape:
        re_pde_xyt, k_rad_xy, dk_dx, dk_dy = GetPdeAndK(batchsz, step, epoch)
        loss_ek, loss_bc, loss_ic, loss_mse = LossBdLogits(
            train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt)
        loss_pde, loss_up_dw = LossPde(
            re_pde_xyt, k_rad_xy, dk_dx, dk_dy, up_nflow, dw_nflow)
        # Total loss
        total_loss = lambda1 * loss_mse + lambda2 * loss_pde + lambda3 * \
            loss_bc + lambda4 * loss_ek + lambda5 * loss_ic + lambda6 * loss_up_dw

    grads = tape.gradient(
        total_loss, model.trainable_variables)  # Compute total-loss gradients with respect to trainable parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Update gradients
    return total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic, loss_up_dw


# Training data
(train_xyt, train_h), (test_xyt, test_h) = getdata.get_data()  # Load the data
for i in range(2):
    train_xyt[:, i] = train_xyt[:, i]+1  # Shift input x and y from (0, 50) to (1, 51)
train_xyt = realstep(train_xyt)
# train_h=train_h/10 #Normalize the hydraulic head
train_h = train_h.astype(np.float)
for i in range(2):
    test_xyt[:, i] = test_xyt[:, i]+1
test_xyt = realstep(test_xyt)
# test_h=test_h/10
test_h = test_h.astype(np.float)


total_train_num = train_xyt.shape[0]

print(train_xyt.shape, train_h.shape)  # Print the training data shape
print(test_xyt.shape, test_h.shape)  # Print the test data shape


# Load k and compute its gradients
o_k31_hk_data = np.loadtxt(
    '/home/cc/CCFs/zhuxy/tgnn design/reproduce/case_study_1/hk3.1')
x0, y0 = np.mgrid[0:1020:20, 0:1020:20]
z0 = o_k31_hk_data

f = interpolate.interp2d(x0, y0, z0, kind='cubic')  # Construct a cubic spline interpolant from sample points
x1 = [_ for _ in range(0, 1020)]  # np.linspace(start, stop, num=50)
y1 = [_ for _ in range(0, 1020)]
z1 = f(x1, y1).T  # Interpolation results


k31_hk_data = z1
k31_grad_x = np.gradient(k31_hk_data, axis=0)  # （51,51,51）
k31_grad_y = np.gradient(k31_hk_data, axis=1)

N_pde = 50000
x_pde = np.random.randint(20, 1018, (N_pde, 1))
y_pde = np.random.randint(20, 1018, (N_pde, 1))
t_pde = np.random.randint(20, 1019, (N_pde, 1))
re_pde_xyt = np.hstack(((x_pde+1)/1000, (y_pde+1)/1000, t_pde/1000))
k_pde_xy = np.hstack((x_pde, y_pde))

k_rad_xy = tf.gather_nd(k31_hk_data, tf.cast(k_pde_xy, dtype=tf.int32))
k_rad_xy = tf.reshape(tf.cast(k_rad_xy, dtype=tf.float32),
                      (k_rad_xy.shape[0], 1))
dk_dx = tf.gather_nd(k31_grad_x, tf.cast(k_pde_xy, dtype=tf.int32))
dk_dx = tf.reshape(tf.cast(dk_dx, dtype=tf.float32),
                   (dk_dx.shape[0], 1))  # Partial derivatives of k with respect to rad_xy
dk_dy = tf.gather_nd(k31_grad_y, tf.cast(k_pde_xy, dtype=tf.int32))
dk_dy = tf.reshape(tf.cast(dk_dy, dtype=tf.float32),
                   (dk_dy.shape[0], 1))  # Partial derivatives of k with respect to rad_xy

re_pde_xyt = tf.cast(re_pde_xyt, dtype=tf.float32)


# Initial conditions
N_lic = 10000
x_lic = 1+50*lhs(1, N_lic)
y_lic = np.ones((N_lic, 1))
t_lic = np.zeros((N_lic, 1))
lic_xyt = np.hstack((x_lic, y_lic, t_lic))
lic_xyt = realstep(lic_xyt)
lic_xyt = tf.cast(lic_xyt, dtype=tf.float32)

""" N_mic=50000
x_mic=1+50*lhs(1, N_mic)
y_mic=2+48*lhs(1, N_mic)
t_mic=np.zeros((N_mic,1))
mic_xyt=np.hstack((x_mic,y_mic,t_mic))
mic_xyt = realstep(mic_xyt)
mic_xyt = tf.cast(mic_xyt, dtype=tf.float32)
 """
N_ric = 10000
x_ric = 1+50*lhs(1, N_ric)
y_ric = 2+49*lhs(1, N_ric)
t_ric = np.zeros((N_ric, 1))
ric_xyt = np.hstack((x_ric, y_ric, t_ric))
ric_xyt = realstep(ric_xyt)
ric_xyt = tf.cast(ric_xyt, dtype=tf.float32)


# Boundary conditions
N_lbc = 10000
x_lbc = 1+50*lhs(1, N_lbc)
y_lbc = np.ones((N_lbc, 1))
t_lbc = 50*lhs(1, N_lbc)
lbc_xyt = np.hstack((x_lbc, y_lbc, t_lbc))
lbc_xyt = realstep(lbc_xyt)
lbc_xyt = tf.cast(lbc_xyt, dtype=tf.float32)

N_rbc = 10000
x_rbc = 1+50*lhs(1, N_rbc)
y_rbc = 51*np.ones((N_rbc, 1))
t_rbc = 50*lhs(1, N_rbc)
rbc_xyt = np.hstack((x_rbc, y_rbc, t_rbc))
rbc_xyt = realstep(rbc_xyt)
rbc_xyt = tf.cast(rbc_xyt, dtype=tf.float32)


# No-flow boundary
N_up_nflow = 10000
x_up_nflow = np.ones((N_up_nflow, 1))
y_up_nflow = 1+50*lhs(1, N_up_nflow)
t_up_nflow = 50*lhs(1, N_up_nflow)
up_nflow = np.hstack((x_up_nflow, y_up_nflow, t_up_nflow))
up_nflow = realstep(up_nflow)
up_nflow = tf.cast(up_nflow, dtype=tf.float32)

N_dw_nflow = 10000
x_dw_nflow = 51*np.ones((N_dw_nflow, 1))
y_dw_nflow = 1+50*lhs(1, N_dw_nflow)
t_dw_nflow = 50*lhs(1, N_dw_nflow)
dw_nflow = np.hstack((x_dw_nflow, y_dw_nflow, t_dw_nflow))
dw_nflow = realstep(dw_nflow)
dw_nflow = tf.cast(dw_nflow, dtype=tf.float32)

db = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # Generate training data batches
db = db.map(preprocess).shuffle(50000).batch(batchsz)  # Map the preprocessing function

db_test = tf.data.Dataset.from_tensor_slices((test_xyt, test_h))  # Generate test data batches
db_test = db_test.map(preprocess).batch(batchsz)  # Map the preprocessing function


# Model
class MyDense(layers.Layer):
    # to replace standard layers.Dense()

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight(
            'w', [inp_dim, outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))  # 10
        self.bias = self.add_weight(
            'b', [outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))  # 10

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias

        return out


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = MyDense(3, 50)
        self.fc2 = MyDense(50, 50)
        self.fc3 = MyDense(50, 50)
        self.fc4 = MyDense(50, 50)
        self.fc5 = MyDense(50, 50)
        self.fc6 = MyDense(50, 50)
        self.fc7 = MyDense(50, 1)

    def call(self, inputs, training=None):
        """

        :param inputs: [b, 3]
        :param training:
        :return:
        """
        inp = tf.reshape(inputs, [-1, 3])

        o11 = self.fc1(inp)
        o12 = tf.nn.tanh(o11)

        o21 = self.fc2(o12)
        o22 = tf.nn.tanh(o21)

        o31 = self.fc3(o22)
        o32 = tf.nn.tanh(o31)

        o41 = self.fc4(o32)
        o42 = tf.nn.tanh(o41)

        o51 = self.fc5(o42)
        o52 = tf.nn.tanh(o51)

        o61 = self.fc6(o52)
        o62 = tf.nn.tanh(o61)

        o71 = self.fc7(o62)
        out = tf.nn.tanh(o71)
        # [b, 50] => [b, 50]
        # [b, 50] => [b]
        return out


model = MyModel()
model.build(input_shape=[None, 3])

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)  # Optimizer
# optimizer = optimizers.Adam(1e-3)  # Optimizer


def main():
    test_mse_log = 10
    total_loss_log = 10
    for epoch in range(epochs):  # Number of epochs
        # Training
        for step, (train_xyt, train_h) in enumerate(db):

            total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic, loss_up_dw = train_one_step(
                step, epoch, train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt)

            if step % 20 == 0:  # Print the loss every 20 steps
                tf.print(epoch, step,
                         'loss:tol', float(total_loss),
                         'loss:mse', float(lambda1 * loss_mse),
                         'loss:pde', float(lambda2 * loss_pde),
                         'loss:bc', float(lambda3 * loss_bc),
                         'loss:ek', float(lambda4 * loss_ek),
                         'loss:ic', float(lambda5 * loss_ic),
                         'loss:ud', float(lambda6 * loss_up_dw),)

                with summary_writer.as_default():  # Record logs with TensorBoard
                    tf.summary.scalar(
                        'loss:tol', float(total_loss), step=epoch)
                    tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                    tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                    tf.summary.scalar('loss:bc', float(loss_bc), step=epoch)
                    tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                    tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
        # Testing
        for test_xyt, test_h in db_test:
            # x: [b, 3] => [b]
            # y: [b]
            test_xyt = tf.reshape(test_xyt, [-1, 3])
            # [b, 10]
            logits = model(test_xyt)

            test_mse = tf.reduce_mean(tf.losses.MSE(test_h, logits))

        print(epoch, 'test mse:', test_mse)
        with summary_writer.as_default():
            tf.summary.scalar('test mse', float(test_mse), step=epoch)
        '''
        if test_mse < test_mse_log:
            test_mse_log = test_mse
            model.save_weights(checkpoint_save_path)
            print('-------------saved weights.---------------')
            print('best test mse：', float(test_mse_log))
        '''

        if total_loss < total_loss_log:
            total_loss_log = total_loss
            model.save_weights(checkpoint_save_path)
            print('-------------saved weights.---------------')
            print('best total_loss：', float(total_loss_log))

        '''
        model.save_weights(checkpoint_save_path)
        print('-------------saved weights.---------------')
        '''
    with summary_writer.as_default():
        tf.summary.trace_export(
            name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    main()

t = 0
