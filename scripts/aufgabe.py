#Script written by Aleksandr Schamberger (GitHub: JLEKS)
#Created: 2024-06-18
#Latest Version: 2024-06-19



class NeurolNetwork:

    def __init__(self,input_nodes:int,hidden_nodes:int,output_nodes:int,learning_rate:float,seed="False"):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.learning_rate = learning_rate
        self.wih = self.get_weights(seed)[0]
        self.who = self.get_weights(seed)[1]
        self.act_func = lambda x: special.expit(x)

    def get_weights(self,seed):
        if isinstance(seed,bool):
            np.random.default_rng()
        elif isinstance(seed,(int,float)):
            np.random.default_rng(seed)
        wih = np.random.normal(0,pow(self.h_nodes,-0.5),(self.h_nodes,self.i_nodes))
        who = np.random.normal(0,pow(self.o_nodes,-0.5),(self.o_nodes,self.h_nodes))
        return wih,who

    def train(self):
        pass

    def query(self,input_list):
        input = np.array(input_list,ndmin=2).T
        hidden_inputs = self.wih @ input
        hidden_outputs = self.act_func(hidden_inputs)
        final_inputs = self.who @ hidden_outputs
        final_outputs = self.act_func(final_inputs)
        return final_outputs

import numpy as np
from scipy import special

x = NeurolNetwork(3,3,3,0.5)
print(x.query([0.6, 0.9, 1.0]))