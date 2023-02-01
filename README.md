lstm.py is implementing an LSTM cell. It first initializes the weight matrices and bias vectors for the four gates (forget gate, input gate, cell update and output gate). It then initializes the gradients for each of these weights and bias vectors. After that, it sets the initial values for the hidden and cell states of the cell, as well as the previous hidden and cell states. The forward() method is then used to compute the four gates, the cell state and the hidden state, given an input. Finally, the backward() method is used to compute the gradients of the weights and bias vectors with respect to the loss.

Here the maths behind :

![image](https://user-images.githubusercontent.com/62066804/216072738-1ddb1d62-3b05-45a4-841e-d51b35209ed0.png)
![image](https://user-images.githubusercontent.com/62066804/216072877-f93f6f8b-d743-41c5-8f82-b0a894f0554b.png)
![image](https://user-images.githubusercontent.com/62066804/216072961-c76b23c1-6b8f-4a26-b10e-8dcbd863d34a.png)
![image](https://user-images.githubusercontent.com/62066804/216073030-891d81f2-df2f-40fd-99b5-d841c42363d1.png)
