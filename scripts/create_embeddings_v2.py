#Script written by Aleksandr Schamberger (GitHub: JLEKS)
#as part of the course by Roland Meyer and Mariia Razguliaeva 'Sprachenübergreifend: Computerlinguistik II – Digitale Sprachmodelle und ihre Anwendung (nicht nur in slawischen Sprachen)' at the Humboldt-University Berlin in the summer semester 2024.
#Script is partly based on the blog post 'Word2vec from Scratch' by Jake Tae from 13th July 2020 <https://jaketae.github.io/study/word2vec/> (link lastly accessed on 2024-07-02).
#Created: 2024-06-07
#Latest Version: 2024-07-03
#The script and its content is licensed under the Creative Commons Attribution ShareAlike (CC BY-SA) 4.0 license.
#Version 2: Changed method 'train' of EmbeddingsModel class. Using arrays instead of lists increasing not only speed but also memory usage.

#import pandas as pd
import numpy as np
from datetime import datetime
#np.seterr(all='raise')

def import_tokens(file):
    '''Loads a file *file*, where every row represents a token. Returns them as a list.'''
    with open(file,"r") as f:
        tokens = f.readlines()
    tokens = [row[:-1] for row in tokens]
    return tokens

class EmbeddingsModel():

    def __init__(self,tokens):
        self.tokens = tokens
        self.voc_dict = {tok: ind for ind,tok in enumerate(self._create_voc())}
        self.id_voc_dict = {ind: tok for tok,ind in self.voc_dict.items()}

    def _create_voc(self):
        vocabulary = list(set(self.tokens))
        return vocabulary

    def create_train_data(self,window_size):
        ''''''

        def yield_range(*ranges):
            for range in ranges:
                yield from range

        def total_num_of_pairs(self,window_size):
            normal_count = len(self.tokens)-(window_size*2)
            normal_count *= window_size*2
            special_count = range(window_size,window_size*2)
            special_count = sum(special_count)*2
            final_count = normal_count+special_count
            return final_count

        def generate_vector_pairs(self,window_size):
            num_rows = total_num_of_pairs(self,window_size)
            main_wd_hot_vec = np.zeros((num_rows,len(self.voc_dict)),dtype=np.int8)
            cnxt_wd_hot_vec = np.zeros((num_rows,len(self.voc_dict)),dtype=np.int8)
            row_count = 0
            for ind,word in enumerate(self.tokens):
                #print(f"\ntoken: {word}")
                tok_id1 = self.voc_dict[word]
                #print(f"token id: {tok_id1}")
                left_range = range(max(0,ind-window_size),ind)
                right_range = range(ind,min(len(self.tokens),ind+window_size+1))
                indeces = yield_range(left_range,right_range)
                for ind2 in indeces:
                    if ind == ind2:
                        continue
                    word2 = self.tokens[ind2]
                    #print(f"wd pair: {word2}")
                    tok_id2 = self.voc_dict[word2]
                    #print(f"wd pair tok id: {tok_id2}")
                    main_wd_hot_vec[row_count,tok_id1] = 1
                    #print(f"word1 one-hot-vector: {main_wd_hot_vec[row_count]}")
                    cnxt_wd_hot_vec[row_count,tok_id2] = 1
                    #print(f"word2 one-hot-vector: {cnxt_wd_hot_vec[row_count]}")
                    row_count += 1
            return main_wd_hot_vec,cnxt_wd_hot_vec

        if (len(self.tokens)/2) <= window_size:
            raise Exception(f"Window size of /{window_size}/ is too big for the size of the current list of tokens /{len(self.tokens)}/.")

        x,y = generate_vector_pairs(self,window_size)
        self.model = {}
        self.model["INP"] = x
        self.model["COMP"] = y
        #print(f"shape of self.input_hot_vec is {self.input_hot_vec.shape} with {(self.input_hot_vec.nbytes)/1e+9} giga-bytes and shape of self.compare_hot_vec is {self.compare_hot_vec.shape} with {(self.compare_hot_vec.nbytes)/1e+9} giga-bytes.")


    def train(self,feat_num:int,epochs=10,lr=0.5,dt=np.float32,seed=False):

        self.train_stats = {"lr": lr, "epochs": epochs, "feat_num": feat_num}

        def init_model(self,feat_num,dt,seed):
            if isinstance(seed,bool):
                rng = np.random.default_rng()
            elif isinstance(seed,int):
                rng = np.random.default_rng(seed)
            self.model["HLW"] = rng.standard_normal((len(self.voc_dict),feat_num),dtype=dt)
            #self.model["HLW"] = np.random.randn(len(self.voc_dict),feat_num)
            self.model["OLW"] = rng.standard_normal((feat_num,len(self.voc_dict)),dtype=dt)
            #self.model["OLW"] = np.random.randn(feat_num,len(self.voc_dict))

        def forward_propagation(self):

            def softmax(matrix):
                new_maxtrix = np.zeros(matrix.shape,dtype=np.float32)
                for ind,vec in enumerate(matrix):
                    #using this instead of simply np.exp(vec) removes the overflow Runtime Warning as well as the division in the line after that.
                    exp = np.exp(vec - np.max(vec))
                    new_maxtrix[ind] = exp/exp.sum()
                return new_maxtrix

            self.model["HLO"] = self.model["INP"] @ self.model["HLW"]
            self.model["OLI"] = self.model["HLO"] @ self.model["OLW"]
            self.model["OLO"] = softmax(self.model["OLI"])

        def back_propagation(self,lr):
            #Ableitung Cross Entropy als Fehlerfunktion.
            error_OL = self.model["OLO"] - self.model["COMP"]
            error_OLW = self.model["HLO"].T @ error_OL
            self.model["HLO"] = 0
            error_HL = error_OL @ self.model["OLW"].T
            error_HLW = self.model["INP"].T @ error_HL
            error_HL = 0
            self.model["HLW"] -= (lr * error_HLW)
            self.model["OLW"] -= (lr * error_OLW)
            return cross_entropy(self.model["OLO"],self.model["COMP"])
            #return - np.sum(np.log(self.model["OLO"]) * self.model["COMP"])

        def cross_entropy(z, y):
            #using log1p instead of log is a bandate; fo cases where a zero is there, for which there is no log value available.
            return - np.sum(np.log1p(z) * y)

        def store_embeddings(self):
            self.word_embeddings = {}
            for word,id in self.voc_dict.items():
                one_hot_vec = np.zeros(len(self.voc_dict))
                one_hot_vec[id] = 1
                word_embedding = one_hot_vec @ self.model["HLW"]
                self.word_embeddings[word] = word_embedding

        init_model(self,feat_num,dt,seed)
        print(f"Model initialized. Start training.")
        #print(f"\nHLW-init:\n{model['HLW']}")
        #print(f"\nOLW-init:\n{model['OLW']}")
        results = []
        for i in range(epochs):
            print(f"model wights:\n{self.model['HLW'][0]}")
            print(f"Epoche {i+1}")
            forward_propagation(self)
            error = back_propagation(self,lr)
            print(f"Error Rate: {error}")
            results.append(error)

        self.train_stats["errors"] =  results
        store_embeddings(self)

    def save_embeddings(self,path):
        file_name = path+str(datetime.now())[:16]+".txt"
        with open(file_name,"w") as file:
            file.write(str(self.train_stats)+"\n\n")
            for word,embed in self.word_embeddings.items():
                file.write(word+"\n")
                for val in embed:
                    file.write("\t"+str(val))
                file.write("\n\n")
    
    def get_embedding(self,word):
        try:
            embed = self.word_embeddings[word]
        except KeyError:
            print(f"Word not part of the vocabulary, which was trained.")
        return embed
    
    def get_assoc_words(self,word):

        def forward(one_hot_vec):
            def softmax(matrix):
                new_maxtrix = np.zeros(matrix.shape,dtype=np.float32)
                for ind,vec in enumerate(matrix):
                    #using this instead of simply np.exp(vec) removes the overflow Runtime Warning as well as the division in the line after that.
                    exp = np.exp(vec - np.max(vec))
                    new_maxtrix[ind] = exp/exp.sum()
                return new_maxtrix

            hlo = one_hot_vec @ self.model["HLW"]
            oli = hlo @ self.model["OLW"]
            hlo = 0
            olo = softmax(oli)
            oli = 0
            return olo[0]

        embed = self.get_embedding(word)
        id = self.voc_dict[word]
        one_hot_vec = np.zeros(len(self.voc_dict))
        one_hot_vec[id] = 1
        sm_vec = forward([one_hot_vec])
        assoc_wds = []
        for id in np.argsort(sm_vec)[::-1]:
            wd = self.id_voc_dict[id]
            assoc_wds.append(wd)
        return assoc_wds

    def get_nearest_word(self,word):
        embed = self.get_embedding(word)
        dist_wd = {}
        for wd,embedding in self.word_embeddings.items():
            if wd == word:
                continue
            dist = abs(embed - embedding)
            dist = np.sum(dist)
            dist_wd[dist] = wd
        nearest_wd = dist_wd[min(dist_wd.keys())]
        return nearest_wd




#Path and file name.
tokens_path = "../data/lang_data/prepared_data/"
save_embeddings_path = "../data/word_embeddings/"
data_file = "urum_tokens.txt"#"urum_tokens_50%.txt"#"example.txt"

#Get tokens from data.
tokens = import_tokens(tokens_path+data_file)

#For testing, understanding and debugging!
#tokens = ['Auge', 'Arm', 'Nase', 'Auge', 'Fuß', 'Mund', 'Arm', 'Auge', 'Nase', 'Mund', 'Mund', 'Fuß', 'Arm', 'Arm', 'Fuß', 'Fuß', 'Mund', 'Fuß', 'Arm', 'Mund', 'Arm', 'Auge', 'Arm', 'Auge', 'Arm', 'Auge', 'Mund', 'Nase', 'Auge', 'Nase', 'Mund', 'Fuß', 'Auge', 'Arm', 'Nase', 'Arm', 'Auge', 'Auge', 'Nase', 'Nase', 'Nase', 'Mund', 'Arm', 'Auge', 'Arm', 'Arm', 'Auge', 'Auge', 'Mund', 'Auge', 'Nase', 'Nase', 'Mund', 'Fuß', 'Arm', 'Auge', 'Nase', 'Mund', 'Auge', 'Fuß', 'Auge', 'Nase', 'Auge', 'Nase', 'Auge', 'Nase', 'Arm', 'Nase', 'Arm', 'Arm', 'Fuß', 'Nase', 'Nase', 'Fuß', 'Fuß', 'Auge', 'Fuß', 'Auge', 'Auge', 'Fuß', 'Fuß', 'Nase', 'Mund', 'Fuß', 'Nase', 'Nase', 'Auge', 'Auge', 'Mund', 'Fuß', 'Arm', 'Auge', 'Auge', 'Arm', 'Mund', 'Nase', 'Auge', 'Arm', 'Mund', 'Nase']
#https://www.kas.de/de/web/geschichte-der-cdu/personen/biogramm-detail/-/content/angela-merkel-1

#Create the embeddings model and the training data.
model = EmbeddingsModel(tokens)
model.create_train_data(20)

#Train.
model.train(10,lr=0.005,epochs=20)#bei Harry: 0,005 und 50 epochen.
#seed=876273465

#Get embedding:
word = "Angela"#Angela#Harry
print(model.get_nearest_word(word))
model.save_embeddings(save_embeddings_path)
wds = model.get_assoc_words(word)
print(wds[:10])
print(model.get_embedding(word))
