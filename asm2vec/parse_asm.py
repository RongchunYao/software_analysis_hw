from typing import *
import copy
import re
import random

jmp_op = {
    'jmp', 'ja', 'jae', 'jb', 'jbe', 'jc', 'jcxz', 'jecxz', 'jrcxz', 'je', 'jg', 'jge', 'jl', 'jle', 'jna',
    'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl', 'jnle', 'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp',
    'jpe', 'jpo', 'js', 'jz', 'jmpq'
}

uncond_jmp = {
    'jmp'
}

call_op = {
    'call', 'callq'
}

ret_op = {
    'ret', 'retq'
}


class Instruction:

    def __init__(self, op:str, args: str):
        self._op = op.strip()
        self._args = args
        self._args = self._args.strip()
        self._args = self._args.replace('\t', ' ')
        args_list = args.split(' ')
        self._args_list = []
        self._function_callee = None
        for item in args_list:
            if item is not None:
                self._args_list.append(item)

    def op(self):
        return self._op

    def args(self):
        return self._args

    def arg_list(self):
        return list(self._args_list)

    def count_callee(self, call_time):
        if self._op in jmp_op or self._op in call_op:
            for arg in self._args_list:
                for function in call_time:
                    # print('arg is', arg)
                    if arg == function.name():
                        call_time[function] += 1
                        self._function_callee = str(arg)
                        return 1
        return 0

    def get_callee(self):
        return self._function_callee



class IdaBlock:

    _next_unused_id:int = 1

    def __init__(self, label: str = 'Not_have_name'):

        self._id = self.__class__._next_unused_id
        self.__class__._next_unused_id += 1
        self._instructions: List[Instruction] = []
        self._name = label
        self._function_out_degree = -1

        self.function_name: str = 'Not_have_function_name'

    def name(self):
        return self._name

    def function_out_degree(self):
        return self._function_out_degree

    def __len__(self):
        return len(self._instructions)

    def id(self):
        return self._id

    def add_instruction(self, instr: Instruction):
        self._instructions.append(instr)

    def instructions(self):
        return self._instructions

    def count_callee(self, call_time: Dict['Function', int]):

        self._function_out_degree = 0
        for instr in self._instructions:
            self._function_out_degree += instr.count_callee(call_time)
        return self._function_out_degree

    def expand_callee(self, callee_list: List['Function']):
        count = 0
        tmp_list: List['Instruction'] = []

        for index in range(len(self._instructions)):
            find = 0
            for f in callee_list:
                if f.name() == self._instructions[index].get_callee():

                    for blk in f.blocks():
                        for inst in blk.instructions():
                            op = inst.op()
                            new_inst = Instruction(op=inst.op(),args=inst.args())
                            if op in ret_op:
                                new_inst = Instruction('jmp', self.function_name+'caller'+str(count))
                                count += 1
                            tmp_list.append(new_inst)

                    find = 1
                    break
            if find == 0:
                tmp_list.append(self._instructions[index])

        self._instructions = tmp_list



class Function:

    _next_unused_id: int = 1

    def __init__(self, label: str = 'Not_have_name'):

        self._id = self.__class__._next_unused_id
        self.__class__._next_unused_id += 1
        self._blocks: List[IdaBlock] = []
        self._name = label
        self._function_out_degree = -1

    def function_out_degree(self):
        return self._function_out_degree

    def __len__(self):
        return len(self._blocks)

    def name(self):
        return self._name

    def inst_len(self):
        count = 0
        for item in self._blocks:
            count += len(item)
        return count

    def id(self):
        return self._id

    def add_block(self, block: IdaBlock):
        block.function_name = self._name
        self._blocks.append(block)

    def count_callee(self, call_time: Dict['Function',int]):
        self._function_out_degree = 0
        for blk in self._blocks:
            self._function_out_degree += blk.count_callee(call_time)

    def blocks(self):
        return self._blocks

    def expand_callee(self, callee_list: List['Function']):
        if self.__len__() <= 10:
            for blk in self._blocks:
                blk.expand_callee(callee_list)
        else:
            tmp_list = []
            for item in callee_list:
                if float(len(item))/float(self.__len__()) < 0.6:
                    tmp_list.append(item)

            for blk in self._blocks:
                blk.expand_callee(tmp_list)

    def generate_seq(self):
        seq = []
        if len(self._blocks) == 0:
            return seq
        blk = 0
        index = 0
        blk_name = {}

        for i in range(len(self._blocks)):
            blk_name[self._blocks[i].name()] = i

        total_inst = self.inst_len()
        while True:
            # too long or index overflow
            if len(seq) > 2*total_inst:
                break
            if blk >= len(self.blocks()):
                break
            if len(self._blocks[blk]) == 0:
                blk += 1
                continue
            if index >= len(self._blocks[blk]):
                index = 0
                blk += 1
                continue
            inst_list = self._blocks[blk].instructions()
            inst = inst_list[index]
            if inst.op() in jmp_op :
                find = 0
                for arg in inst.arg_list():
                    if arg in blk_name:
                        find = 1
                        tmp = random.random()
                        if tmp > 0.5:
                            index +=1
                            break
                        else:
                            index = 0
                            blk = blk_name[arg]
                        continue
                    elif arg != 'short' and arg != 'long':
                        tmp = random.random()
                        find = 1
                        if inst.op() in uncond_jmp:
                            seq.append(inst)
                            return seq
                        if tmp > 0.5:
                            index += 1
                            break
                        else:
                            seq.append(inst)
                            return seq

                if find == 0:
                    index += 1
                    seq.append(inst)

            elif inst.op() in ret_op or inst.op() in call_op:
                seq.append(inst)
                return seq

            else:
                index += 1
                if index < len(self._blocks[blk]):
                    seq.append(inst)

        return seq


class asm_parse:

    def __init__(self, name:str, func_list: List[str]):
        self.filename = name
        self.functions_list = copy.deepcopy(func_list)
        self._functions : List[Function]= []
        self._current_function = None
        self._current_block = None
        self._call_time: Dict['Function',int]= {}

    def name(self):
        return self.filename

    def get_functions(self) -> List[Function]:
        return self._functions

    def inline_callee(self):

        self._call_time = {}
        for function in self._functions:
            self._call_time[function] = 0

        for function in self._functions:
            function.count_callee(self._call_time)


        to_expand = []
        to_expand_name = []

        for func in self._call_time:
            in_degree = float(self._call_time[func])
            out_degree = float(func.function_out_degree())
            if in_degree>0 and out_degree/in_degree < 0.01:
                to_expand.append(func)
                to_expand_name.append(func.name())

        for function in self._functions:
            if function.name() not in to_expand_name:
                function.expand_callee(to_expand)

    def flush(self):
        self._functions.append(self._current_function)

    def parse_line(self, line:str):
        line = line.strip()
        line = line.replace('\t', ' ')
        if len(line) == 0 :
            return
        pattern= re.compile(r'<(.*?)>:', re.S)
        result = re.search(pattern, line)
        if result is not None:
            # this means that we met a function
            if self._current_function is not None:
                self._functions.append(self._current_function)
            self._current_function = Function(label=result.group(1))
            self._current_block = None
            return

        pattern2 = re.compile(r'([^\s]+):', re.S)
        result2 = re.match(pattern2, line)
        if result2 is not None:
            # this means that we met a block in ida asm
            if self._current_block is not None :
                if self._current_function is None:
                    print("error! find a block before meet a function")
                else:
                    self._current_function.add_block(self._current_block)

            self._current_block = IdaBlock(label=result2.group(1))
            return

        for index in range(len(line)):
            if line[index] == ' ':
                inst = Instruction(op=line[:index],args=(line[index:]).strip())
                if self._current_block is not None:
                    self._current_block.add_instruction(inst)
                elif self._current_function is not None:
                    self._current_block = IdaBlock(label=self._current_function.name())
                    self._current_block.add_instruction(inst)
                    self._current_function.add_block(self._current_block)
                else:
                    print('error! find an instruction before meet a function')
                return

        inst = Instruction(op=line, args='')
        if self._current_block is not None:
            self._current_block.add_instruction(inst)
        else:
            print('error! find an instruction before meet a block')




