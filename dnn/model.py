import torch
import torch.nn as nn

class dnn(nn.Module):
    def __init__(self,input_hidden,hidden_size1,hidden_size2,output_size):
        super().__init__()
        #Liner(x,y)针对输入的最后一个维度，相当于右乘一个x*y的矩阵，并加入偏置b
        self.layer1=nn.Linear(input_hidden,hidden_size1)
        self.layer2=nn.Linear(hidden_size1,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.1)
        self.output=nn.Linear(hidden_size2,output_size)
     
    
    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)  # dropout 放在激活函数之后 softmax 之前
        out = self.output(x)
        return out
