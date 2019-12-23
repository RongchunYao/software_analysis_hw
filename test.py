import asm2vec.parse_asm
from asm2vec.model import Asm2VecModel
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    sfile = open('./tiny_test/libz.so.1.2.11_gcc_O0.asm_strip', 'r')
    ffile = open('./tiny_test/libz.so.1.2.11_gcc_O0.asm_f', 'r')

    function_names = []

    for line in ffile.readlines():
        function_names.append(line.strip())

############################

    parse1 = asm2vec.parse_asm.asm_parse(name='libzO0', func_list=function_names)
    for line in sfile.readlines():
        parse1.parse_line(line)
    parse1.flush()
    parse1.inline_callee()

############################

    name_list = []

    for function in parse1.get_functions():
        for blk in function.blocks():
            for inst in blk.instructions():
                op = inst.op()
                if op not in name_list:
                    name_list.append(op)
                for arg in inst.arg_list():
                    if arg not in name_list:
                        name_list.append(arg)

    ## Train
    model = Asm2VecModel(d=200, lr=0.001, vocabulary_labels=name_list, k=25)
    for function in parse1.get_functions():
        seqs = []
        print("process "+function.name())
        for i in range(10):
            seqs.append(function.generate_seq())
        for seq in seqs:
            if len(seq) < 3:
                continue
            for i in range(1, len(seq)-1):
                model.fit(seqs = [seq[i-1], seq[i], seq[i+1]], func_name=parse1.name()+'$'+function.name())

    
    sfile.close()
    ffile.close()

    ## Test

    sfile = open('./tiny_test/libz.so.1.2.11_gcc_O2.asm_strip', 'r')
    ffile = open('./tiny_test/libz.so.1.2.11_gcc_O2.asm_f', 'r')

    function_names = []

    for line in ffile.readlines():
        function_names.append(line.strip())

    parse1 = asm2vec.parse_asm.asm_parse(name='libzO2', func_list=function_names)
    for line in sfile.readlines():
        parse1.parse_line(line)
    parse1.flush()
    parse1.inline_callee()

    for function in parse1.get_functions():
        seqs = []
        print("process "+function.name())
        for i in range(10):
            seqs.append(function.generate_seq())
        for seq in seqs:
            if len(seq) < 3:
                continue
            for i in range(1, len(seq)-1):
                model.estimate(seqs = [seq[i-1], seq[i], seq[i+1]], func_name=parse1.name()+'$'+function.name())
    
    for function in parse1.get_functions():
        v2 = model._getFunctionVector(name='libzO2$'+function.name())
        v1 = model._getFunctionVector(name='libzO0$'+function.name())
        print(function.name()+':'+str(cosine_similarity(v1, v2)))
        