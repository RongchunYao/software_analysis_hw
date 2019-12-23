
import numpy as np
from .parse_asm import Instruction
from typing import *

class Asm2VecModel:
    def __init__(self, vocabulary_labels:List[str], d:int=200, lr:float=1e-3, k:int=10):
        """Init the model
                
        Arguments:
            vocabulary_labels {List[str]} -- [types of tokens]
        
        Keyword Arguments:
            d {int} -- [Dimension of the vector] (default: {200})
            lr {float} -- [Learning rate] (default: {1e-3})
             k {int} -- [negative sample count] (default: {10})
        """
        self.d = d
        self.lr = lr
        self.k = k
        self.vocabulary = {}  # The representation of v for each token
        self.vocabulary_2d = {}  # The representation of v' for each token
        self.functions = {}  # The representation of functions
        for t in vocabulary_labels:
            self.vocabulary[t] = np.random.randn(self.d)
            self.vocabulary_2d[t] = np.random.randn(self.d*2)
    
    def fit(self, seqs:List[Instruction], func_name:str="None"):
        if len(seqs) != 3:
            print("Error: the length of the sequence does not equals to 3!")
            exit()

        tmp_func_vector = self._getFunctionVector(name=func_name)  # Get the function vector by name
        ct_prior = self._getCtVector(seqs[0])  # Get the prior neighbor's vector
        ct_later = self._getCtVector(seqs[1])  # Get the later neighbor's vector
        delta = (1/3)*(ct_prior + ct_later + tmp_func_vector)  # Get the delta vector

        grad_theta = np.zeros(2*self.d)  # Init the gradients
        grad_in_prior = np.zeros(2*self.d)  # Init the gradients
        grad_in_later = np.zeros(2*self.d)  # Init the gradients

        # process with tokens
        tokens = self._getTokens(seqs[0])

        for token in tokens:
            sample_tokens = [list(self.vocabulary.keys())[position] for position in np.random.randint(low=0, high=len(self.vocabulary), size=self.k)]
            sample_tokens.append(token)
            f = self._sigmoid(self._getTokenVector2D(name=token)*delta)
            tmp_grad_theta = (1/3)*(self._countExpectation(f, sample_tokens)*self._getTokenVector2D(name=token))
            v_grad = 1 - f*delta
            self.vocabulary_2d[token] -= self.lr*(v_grad)
            grad_in_prior += np.append(tmp_grad_theta[:self.d], (1/len(seqs[0].arg_list()))*tmp_grad_theta[self.d:]) 
            grad_in_later += np.append(tmp_grad_theta[:self.d], (1/len(seqs[2].arg_list()))*tmp_grad_theta[self.d:])
            grad_theta += tmp_grad_theta

        # Update vectors
        for index, token in enumerate(self._getTokens(seqs[0])):
            if index == 0:
                self.vocabulary[token] -= self.lr*(grad_in_prior[:self.d])
            else:
                self.vocabulary[token] -= self.lr*(grad_in_prior[self.d:])

        for index, token in enumerate(self._getTokens(seqs[2])):
            if index == 0:
                self.vocabulary[token] -= self.lr*(grad_in_later[:self.d])
            else:
                self.vocabulary[token] -= self.lr*(grad_in_later[self.d:])
        
        self.functions[func_name] -= self.lr*(grad_theta)

    
    def estimate(self, seqs:List[Instruction], func_name:str="None"):
        if len(seqs) != 3:
            print("Error: the length of the sequence does not equals to 3!")
            exit()

        tmp_func_vector = self._getFunctionVector(name=func_name)  # Get the function vector by name
        ct_prior = self._getCtVector(seqs[0])  # Get the prior neighbor's vector
        ct_later = self._getCtVector(seqs[1])  # Get the later neighbor's vector
        delta = (1/3)*(ct_prior + ct_later + tmp_func_vector)  # Get the delta vector

        grad_theta = np.zeros(2*self.d)  # Init the gradients

        # process with tokens
        tokens = self._getTokens(seqs[0])

        for token in tokens:
            sample_tokens = [list(self.vocabulary.keys())[position] for position in np.random.randint(low=0, high=len(self.vocabulary), size=self.k)]
            sample_tokens.append(token)
            f = self._sigmoid(self._getTokenVector2D(name=token)*delta)
            tmp_grad_theta = (1/3)*(self._countExpectation(f, sample_tokens)*self._getTokenVector2D(name=token))
            grad_theta += tmp_grad_theta

        # Update function vectors
        self.functions[func_name] -= self.lr*(grad_theta)

    # def getFunctionVector(self, func_name:str = "None") -> np.array[float]:
    #     return self._getFunctionVector(func_name = func_name)

    
    def _getTokens(self, seq:Instruction):
        tokens = [seq.op()]
        for arg in seq.arg_list():
            tokens.append(arg)
        return tokens


    def _countExpectation(self, f:np.array, sample_tokens:List[str]):
        sum_result = 0
        for token in sample_tokens:
            tmp = 0
            if token == sample_tokens[len(sample_tokens)-1]:
                tmp = 1
            sum_result += (tmp - f)
        return sum_result


    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    
    def _getFunctionVector(self, name:str="None"):
        if name == 'None':
            print("Error: lack of the name of the function!")
            exit()
        
        # Look up the function vector by the name
        if name not in self.functions:
            self.functions[name] = np.random.randn(2*self.d)*0.01  # Init the function vector
        
        return self.functions[name]


    def _getCtVector(self, seq:Instruction) -> np.array:
        if seq == None:
            print("Error: the ct is none!")
            exit()
        vp_name = seq.op()  # Get operation name 
        vp = self._getTokenVector(name=vp_name)  # Get the vector of operation
        v_arg = np.array([self._getTokenVector(name) for name in seq.arg_list()])
        if (len(v_arg) == 2):
            v_arg = np.array((v_arg[0]+v_arg[1])/(len(v_arg)))
        else:
            v_arg = v_arg[0]
        v = np.append(vp, v_arg)
        return v


    def _getTokenVector(self, name:str="None") -> np.array:
        if name == 'None':
            print("Error: lack of the name of the token!")
            exit()
        
        # Look up the token vector by the name
        if name not in self.vocabulary:
            self.vocabulary[name] = np.zeros(self.d)  # Init the function vector
        
        return self.vocabulary[name]


    def _getTokenVector2D(self, name:str="None") -> np.array:
        if name == 'None':
            print("Error: lack of the name of the token!")
            exit()
        
        # Look up the token vector by the name
        if name not in self.vocabulary_2d:
            self.vocabulary_2d[name] = np.zeros(self.d*2)  # Init the function vector
        
        return self.vocabulary_2d[name]
