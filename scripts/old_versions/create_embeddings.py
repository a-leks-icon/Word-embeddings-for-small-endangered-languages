#Script written by Aleksandr Schamberger (GitHub: JLEKS)
#Created: 2024-06-07
#Latest Version: 2024-06-07

import pandas as pd
import numpy as np

def import_tokens(file):
    '''Loads a file *file*, where every row represents a token. Returns them as a list.'''
    with open(file,"r") as f:
        tokens = f.readlines()
    tokens = [row[:-1] for row in tokens]
    return tokens

class EmbeddingsModel():

    def __init__(self,tokens):
        self.voc = list(set(tokens))
        self.voc_dict = {tok: ind for ind,tok in enumerate(self.voc)}

    def train(self,window_size):
        ''''''

        def yield_range(*ranges):
            for range in ranges:
                yield from range

        def create_one_hot_vector(self,ind):
            '''DONE'''
            vector = np.zeros(len(self.voc_dict))
            vector[ind] = 1
            return vector

        def generate_vector_pairs(self,window_size):
            left_context = []
            right_context = []
            num_of_tokens = len(self.voc_dict)
            for tok_id in range(num_of_tokens):
                left_range = range(max(0,tok_id-window_size),tok_id)
                right_range = range(tok_id,min(num_of_tokens,tok_id+window_size+1))
                indeces = yield_range(left_range,right_range)
                for ind in indeces:
                    if tok_id == ind:
                        continue
                    token_vec = create_one_hot_vector(self,tok_id)
                    adjacent_vec = create_one_hot_vector(self,ind)
                    left_context.append(token_vec)
                    right_context.append(adjacent_vec)
            return np.asarray(left_context),np.asarray(right_context)

        x,y = generate_vector_pairs(self,window_size)
        print(f"shape of x: {x.shape}, len: {len(x)} and shape of y: {y.shape}, len: {len(y)}")

    def train2(self,window_size):
        ''''''

        def yield_range(*ranges):
            for range in ranges:
                yield from range

        def total_num_of_pairs(self,window_size):
            normal_count = len(self.voc)-(window_size*2)
            normal_count *= window_size*2
            special_count = range(window_size,window_size*2)
            special_count = sum(special_count)*2
            final_count = normal_count+special_count
            return final_count

        def generate_vector_pairs(self,window_size):
            num_of_tokens = len(self.voc_dict)
            num_rows = total_num_of_pairs(self,window_size)
            left_context = np.zeros((num_rows,num_of_tokens))
            right_context = np.zeros((num_rows,num_of_tokens))
            row_count = 0
            for tok_id1 in range(num_of_tokens):
                left_range = range(max(0,tok_id1-window_size),tok_id1)
                right_range = range(tok_id1,min(num_of_tokens,tok_id1+window_size+1))
                indeces = yield_range(left_range,right_range)
                for tok_id2 in indeces:
                    if tok_id1 == tok_id2:
                        continue
                    left_context[row_count,tok_id1] = 1
                    right_context[row_count,tok_id2] = 1
                    row_count += 1
            return left_context,right_context

        if (len(self.voc)/2) <= window_size:
            raise Exception(f"Window size of /{window_size}/ is too big for the size of the current vocabulary /{len(self.voc)}/.")

        x,y = generate_vector_pairs(self,window_size)
        print(f"shape of x: {x.shape}, len: {len(x)} and shape of y: {y.shape}, len: {len(y)}")
        z = list(x[5,:])
        for ind,n in enumerate(z):
            if n == 1:
                print(ind)

#Path and file name.
tokens_path = "../data/lang_data/prepared_data/"
data_file = "urum_tokens.txt"

#Get tokens from data.
tokens = import_tokens(tokens_path+data_file)

#Create the embeddings model and the training data.
model = EmbeddingsModel(tokens)
model.train2(10)
