import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_grad(y):
    return 1 - y * y

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

class Lstm(object):
    
    def __init__(self, input_size, hidden_size, previous=None):
        if previous:
            self.previous = previous
            previous.next = self
            
        self.input_size, self.hidden_size = input_size, hidden_size

        range_unif, dim = (-1, 1), (hidden_size, hidden_size + input_size)
        # initialize weights
        self.W_f, self.b_f = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))
        self.W_i, self.b_i = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))
        self.W_ct, self.b_ct = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))
        self.W_o, self.b_o = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))

        # initalize gradients
        self.dW_f, self.db_f = np.zeros_like(self.W_f), np.zeros_like(self.b_f)
        self.dW_i, self.db_i = np.zeros_like(self.W_i), np.zeros_like(self.b_i)
        self.dW_ct, self.db_ct = np.zeros_like(self.W_ct), np.zeros_like(self.b_ct)
        self.dW_o, self.db_o = np.zeros_like(self.W_o), np.zeros_like(self.b_o)

        # list of all parameters
        self.params = [
            ('W_f', self.W_f, self.dW_f),
            ('W_i', self.W_i, self.dW_i),
            ('W_ct', self.W_ct, self.dW_ct),
            ('W_o', self.W_o, self.dW_o),

            ('b_f', self.b_f, self.db_f),
            ('b_i', self.b_i, self.db_i),
            ('b_ct', self.b_ct, self.db_ct),
            ('b_o', self.b_o, self.db_o)
        ]
        
        if previous:
            self.previous = previous
            previous.next = self
            
        self.initSequence()

    def initSequence(self):
        self.t = 0
        self.x = {}
        self.h = {}
        self.c = {}
        self.ct = {}

        self.forget_gate = {}
        self.input_gate = {}
        self.cell_update = {}
        self.output_gate = {}

        if hasattr(self, 'previous'):
            self.h[0] = self.previous.h[self.previous.t]
            self.c[0] = self.previous.c[self.previous.t]
        else:
            self.h[0] = np.zeros((self.hidden_size, 1))
            self.c[0] = np.zeros((self.hidden_size, 1))

        if hasattr(self, 'next'):
            self.dh_prev = self.next.dh_prev
            self.dc_prev = self.next.dc_prev
        else:
            self.dh_prev = np.zeros((self.hidden_size, 1))
            self.dc_prev = np.zeros((self.hidden_size, 1))

        # reset all gradients to zero
        for name, param, grad in self.params:
            grad[:] = 0

    def forward(self, x_t):
        self.t += 1
        x_t = x_t.reshape(-1, 1)

        t = self.t
        h = self.h[t-1]
        z = np.vstack((h, x_t))
        
        self.forget_gate[t] = sigmoid(np.dot(self.W_f, z) + self.b_f)
        self.input_gate[t] = sigmoid(np.dot(self.W_i, z) + self.b_i)
        self.cell_update[t] = tanh(np.dot(self.W_ct, z) + self.b_ct)
        self.output_gate[t] = sigmoid(np.dot(self.W_o, z) + self.b_o)

        self.c[t] = self.input_gate[t] * self.cell_update[t] + self.forget_gate[t] * self.c[t-1]
        self.ct[t] = tanh(self.c[t])
        self.h[t] = self.output_gate[t] * self.ct[t]

        self.x[t] = x_t

        return self.h[t]

    def backward(self, dh):
        t = self.t
        dh = dh.reshape(-1,1)
        
        dh = dh + self.dh_prev
        dC = tanh_grad(self.ct[t]) * self.output_gate[t] * dh + self.dc_prev

        d_forget = sigmoid_grad(self.forget_gate[t]) * self.c[t-1] * dC
        d_input = sigmoid_grad(self.input_gate[t]) * self.cell_update[t] * dC
        d_update = tanh_grad(self.cell_update[t]) * self.input_gate[t] * dC
        d_output = sigmoid_grad(self.output_gate[t]) * self.ct[t] * dh

        self.dc_prev = self.forget_gate[t] * dC

        self.db_f += d_forget
        self.db_i += d_input
        self.db_ct += d_update
        self.db_o += d_output

        z = np.vstack((self.h[t-1],self.x[t]))

        self.dW_i += np.dot(d_input, z.T)
        self.dW_f += np.dot(d_forget, z.T)
        self.dW_o += np.dot(d_output, z.T)
        self.dW_ct += np.dot(d_update, z.T)

        dz = np.dot(self.W_f.T, d_forget)
        dz += np.dot(self.W_i.T, d_input)
        dz += np.dot(self.W_ct.T, d_update)
        dz += np.dot(self.W_o.T, d_output)

        self.dh_prev = dz[:self.hidden_size]
        dX = dz[self.hidden_size:]

        self.t -= 1

        return dX
