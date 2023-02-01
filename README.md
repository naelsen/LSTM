lstm.py is implementing an LSTM cell. It first initializes the weight matrices and bias vectors for the four gates (forget gate, input gate, cell update and output gate). It then initializes the gradients for each of these weights and bias vectors. After that, it sets the initial values for the hidden and cell states of the cell, as well as the previous hidden and cell states. The forward() method is then used to compute the four gates, the cell state and the hidden state, given an input. Finally, the backward() method is used to compute the gradients of the weights and bias vectors with respect to the loss.

Here the maths behind :

![lstm](lstm.pdf)
