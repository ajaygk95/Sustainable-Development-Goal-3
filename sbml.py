# ----------------------------------------------------------------------
# sbml.py
#
# Name: Ajay Gopal Krishna
# SBU ID: 112688765
# ----------------------------------------------------------------------

import ply.lex as lex
import ply.yacc as yacc
import sys

tokens = ['REAL_NUM', 'INT_NUM', 'BOOLEAN', 'STRING', 'L_BRAC', 'R_BRAC', 'L_PAREN', 'R_PAREN', 'COMMA', 'TUPLE_HASH',
          'EXPONENT', 'MUL', 'DIV', 'ADD', 'SUB', 'CONS', 'LT', 'LE',
          'EQUALS', 'NE', 'GE', 'GT', 'VAR', 'ASSIGN', 'SEMICOLON', 'L_CURL', 'R_CURL']

reserved = {
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'print': 'PRINT',
    'andalso': 'CONJ',
    'orelse': 'DISJ',
    'not': 'NEG',
    'in': 'MEMBER',
    'div': 'INT_DIV',
    'mod': 'MOD',
    'fun': 'FUNC'
}
tokens += list(reserved.values())


# TODO: Keep t_REAL_NUMBER before t_INT_NUMBER, otherwise 12.12 will give 12-int and 0.12-real
def t_REAL_NUM(t):
    r'\-?\d*\.\d*[Ee]\-?\d+ | \-?\d*\.\d*'
    try:
        t.value = float(t.value)
    except ValueError:
        print("Not able to convert to Float: ", t.value)
        t.value = 0.0
    return t


# Reference ply_demo.py
def t_INT_NUM(t):
    r'\d+'
    try:
        t.value = int(t.value)
    except ValueError:
        print("Not able to convert to Int: ", t.value)
        t.value = 0
    return t


def t_BOOLEAN(t):
    r'True|False'
    if t.value[0] == 'T':
        t.value = True
    else:
        t.value = False
    return t


# TODO : change regex expression. \ not being skipped
def t_STRING(t):
    r'(\"[^\"]*\")|(\'[^\']*\')'
    t.value = t.value[1:-1]
    return t


def t_VAR(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    if t.value in reserved:
        t.type = reserved[t.value]  # Check for reserved words
    return t


t_L_BRAC = r'\['
t_R_BRAC = r'\]'
t_L_PAREN = r'\('
t_R_PAREN = r'\)'
t_COMMA = r'\,'
t_TUPLE_HASH = r'\#'
t_EXPONENT = r'\*\*'
t_MUL = r'\*'
t_DIV = r'/'
t_ADD = r'\+'
t_SUB = r'\-'
t_CONS = r'\:\:'
t_LT = r'\<'
t_LE = r'\<\='
t_EQUALS = r'=='
t_NE = r'\<\>'
t_GE = r'\>\='
t_GT = r'\>'
t_ASSIGN = r'\='
t_SEMICOLON = r'\;'
t_L_CURL = r'\{'
t_R_CURL = r'\}'

t_ignore = ' \t'


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")


def t_error(t):
    raise SyntaxException("Illegal Character '%s', at %d, %d" % (t.value[0], t.lineno, t.lexpos))


precedence = (('left', 'DISJ'),
              ('left', 'CONJ'),
              ('left', 'NEG'),
              ('left', 'LT', 'LE', 'EQUALS', 'NE', 'GE', 'GT'),
              ('right', 'CONS'),
              ('left', 'MEMBER'),
              ('left', 'ADD', 'SUB'),
              ('left', 'MUL', 'DIV', 'INT_DIV', 'MOD'),
              ('right', 'UMINUS'),
              ('right', 'EXPONENT'),
              ('left', 'L_BRAC', 'R_BRAC'),
              ('left', 'TUPLE_HASH'),
              ('left', 'COMMA'),
              ('left', 'L_PAREN', 'R_PAREN'),)


class SemanticException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SyntaxException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# Reference TTgrammar.py
class Node:
    def __init__(self):
        self.parent = None

    def parentCount(self):
        count = 0
        current = self.parent
        while current is not None:
            count += 1
            current = current.parent
        return count


class Value(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self):
        self.typecheck()
        return self.value

    def typecheck(self):
        if type(self.value) not in [str, int, float, bool]:
            raise SemanticException("Value is not of any know type")

    def __str__(self):
        res = "\t" * self.parentCount() + str(self.value)
        return res


class Negation(Node):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.child.parent = self

    def eval(self):
        child_eval = self.child.eval()
        self.typecheck(child_eval)
        return not child_eval

    def typecheck(self, val):
        # members = [str, int, float, bool, list, tuple]
        members = [bool]
        if type(val) not in members:
            raise SemanticException("Value Type is not in {}".format(members))

    def __str__(self):
        res = "\t" * self.parentCount() + "Negation"
        res += "\n" + str(self.child)
        return res


operations = {'**': "EXPONENT", '*': "MULTIPLICATION", '/': "DIVISION", 'div': "INT_DIV", 'mod': "MOD", '+': "ADDITION",
              '-': "SUBTRACTION", 'andalso': "CONJUNCTION", 'orelse': "DISJUNCTION", '<': "LESS_THAN",
              '<=': "LESS_EQUAL", '==': "EQUALS", '<>': "NOT_EQUAL", '>=': "GREATER_EQUAL", '>': "GREATER_THAN"}


class BinaryOperation(Node):
    def __init__(self, left, right, bin_op):
        super().__init__()
        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self
        self.bin_op = bin_op

    def eval(self):
        left_eval = self.left.eval()
        right_eval = self.right.eval()
        self.typecheck(left_eval, right_eval)

        if self.bin_op == '**':
            return left_eval ** right_eval
        elif self.bin_op == '*':
            return left_eval * right_eval
        elif self.bin_op == '/':
            return left_eval / right_eval
        elif self.bin_op == 'div':
            return left_eval // right_eval
        elif self.bin_op == 'mod':
            return left_eval % right_eval
        elif self.bin_op == '-':
            return left_eval - right_eval

    def typecheck(self, val1, val2):
        members = [int, float]
        if (type(val1) not in members) or (type(val2) not in members):
            raise SemanticException("Value1: {} or Value2: {} is not of type {}".format(val1, val2, members))
        if self.bin_op in ['/', 'div', 'mod'] and val2 == 0:
            raise SemanticException("Value2 cannot be 0")

    def __str__(self):
        res = "\t" * self.parentCount() + operations[self.bin_op]
        res += "\n" + str(self.left)
        res += "\n" + str(self.right)
        return res


class AdditionOperation(Node):
    def __init__(self, left, right, bin_op):
        super().__init__()
        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self
        self.bin_op = bin_op

    def eval(self):
        left_eval = self.left.eval()
        right_eval = self.right.eval()
        self.typecheck(left_eval, right_eval)
        return left_eval + right_eval

    def typecheck(self, val1, val2):
        members = [int, float, str, list]
        if (type(val1) not in members) or (type(val2) not in members):
            raise SemanticException("Value1: {} or Value2: {} is not of type {}".format(val1, val2, members))

        if (type(val1) in [list, str]) or (type(val2) in [list, str]):
            if type(val1) != type(val2):
                raise SemanticException("{} and {} cannot be added".format(type(val1), type(val2)))

    def __str__(self):
        res = "\t" * self.parentCount() + operations[self.bin_op]
        res += "\n" + str(self.left)
        res += "\n" + str(self.right)
        return res


class BinaryComparison(Node):
    def __init__(self, left, right, bool_op):
        super().__init__()
        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self
        self.bool_op = bool_op

    def eval(self):
        left_eval = self.left.eval()
        right_eval = self.right.eval()
        self.typecheck(left_eval, right_eval)

        if self.bool_op == '<':
            return self.left.eval() < self.right.eval()
        elif self.bool_op == '<=':
            return self.left.eval() <= self.right.eval()
        elif self.bool_op == '==':
            return self.left.eval() == self.right.eval()
        elif self.bool_op == '<>':
            return self.left.eval() != self.right.eval()
        elif self.bool_op == '>=':
            return self.left.eval() >= self.right.eval()
        elif self.bool_op == '>':
            return self.left.eval() > self.right.eval()

    def typecheck(self, val1, val2):
        members = [int, float, str]
        if (type(val1) not in members) or (type(val2) not in members):
            raise SemanticException("Value1: {} or Value2: {} is not of type {}".format(val1, val2, members))

        if (type(val1) in [str]) or (type(val2) in [str]):
            if type(val1) != type(val2):
                raise SemanticException("{} and {} cannot be compared".format(type(val1), type(val2)))

    def __str__(self):
        res = "\t" * self.parentCount() + operations[self.bool_op]
        res += "\n" + str(self.left)
        res += "\n" + str(self.right)
        return res


class BooleanOperation(Node):
    def __init__(self, left, right, bool_op):
        super().__init__()
        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self
        self.bool_op = bool_op

    def eval(self):
        left_eval = self.left.eval()
        right_eval = self.right.eval()
        self.typecheck(left_eval, right_eval)

        if self.bool_op == 'andalso':
            return self.left.eval() and self.right.eval()
        elif self.bool_op == 'orelse':
            return self.left.eval() or self.right.eval()

    def typecheck(self, val1, val2):
        members = [bool]
        if (type(val1) not in members) or (type(val2) not in members):
            raise SemanticException("Value1: {} or Value2: {} is not of type {}".format(val1, val2, members))

    def __str__(self):
        res = "\t" * self.parentCount() + operations[self.bool_op]
        res += "\n" + str(self.left)
        res += "\n" + str(self.right)
        return res


class ListNode(Node):
    def __init__(self, lis):
        super().__init__()
        if lis:
            self.lis = lis
            self.lis.parent = self
        else:
            self.lis = []

    def eval(self):
        if self.lis:
            return [self.lis.eval()]
        else:
            return self.lis

    def __str__(self):
        res = "\t" * self.parentCount() + "List-Create"
        if self.lis:
            res += "\n" + str(self.lis)
        return res


class ListNodeAppend(Node):
    def __init__(self, prev_list, ele):
        super().__init__()
        self.prev_list = prev_list
        self.ele = ele
        self.prev_list.parent = self
        self.ele.parent = self

    def eval(self):
        prev_l = self.prev_list.eval()
        val = self.ele.eval()
        prev_l.append(val)
        return prev_l

    def __str__(self):
        res = "\t" * self.parentCount() + "List-Add-Elements"
        res += "\n" + str(self.prev_list)
        res += "\n" + str(self.ele)
        return res


class ListIndex(Node):
    def __init__(self, prev_list, ele):
        super().__init__()
        self.prev_list = prev_list
        self.ele = ele
        self.prev_list.parent = self
        self.ele.parent = self

    def eval(self):
        prev_l = self.prev_list.eval()
        index = self.ele.eval()
        self.typecheck(prev_l, index)
        return prev_l[index]

    def typecheck(self, lis, index):
        members = [str, list]
        if type(lis) not in members:
            raise SemanticException("Value Type is not in {}".format(members))

        if type(index) != int:
            raise SemanticException("list indices must be integers not {}".format(type(index)))

        if index >= len(lis):
            raise SemanticException("list index out of range")

    def __str__(self):
        res = "\t" * self.parentCount() + "List-Indexing"
        res += "\n" + str(self.prev_list)
        res += "\n" + str(self.ele)
        return res


class ListCons(Node):
    def __init__(self, con_ele, prev_list):
        super().__init__()
        self.con_ele = con_ele
        self.prev_list = prev_list
        self.con_ele.parent = self
        self.prev_list.parent = self

    def eval(self):
        head_ele = self.con_ele.eval()
        prev_l = self.prev_list.eval()
        self.typecheck(prev_l)

        prev_l.insert(0, head_ele)
        return prev_l

    def typecheck(self, lis):
        if type(lis) != list:
            raise SemanticException("Operand {} must be of type list".format(lis))

    def __str__(self):
        res = "\t" * self.parentCount() + "CONS"
        res += "\n" + str(self.con_ele)
        res += "\n" + str(self.prev_list)
        return res


class ListStrSearch(Node):
    def __init__(self, search_ele, prev_list):
        super().__init__()
        self.search_ele = search_ele
        self.prev_list = prev_list
        self.search_ele.parent = self
        self.prev_list.parent = self

    def eval(self):
        search_ele = self.search_ele.eval()
        prev_l = self.prev_list.eval()
        self.typecheck(prev_l, search_ele)
        return search_ele in prev_l

    def typecheck(self, lis, element):
        members = [str, list]
        if type(lis) not in members:
            raise SemanticException("Value: {} is not of type {}".format(lis, members))

        if (type(lis) == str) and (type(element) != str):
            raise SemanticException("left operand cannot be of type {}".format(type(element)))

    def __str__(self):
        res = "\t" * self.parentCount() + "Membership"
        res += "\n" + str(self.search_ele)
        res += "\n" + str(self.prev_list)
        return res


class TupleNode(Node):
    def __init__(self, tup1, tup2):
        super().__init__()
        self.tup1 = tup1
        self.tup1.parent = self
        self.tup2 = None
        if tup2:
            self.tup2 = tup2
            self.tup2.parent = self

    def eval(self):
        if self.tup2:
            return tuple([self.tup1.eval(), self.tup2.eval()])
        return tuple([self.tup1.eval()])

    def __str__(self):
        res = "\t" * self.parentCount() + "Tuple-Create"
        res += "\n" + str(self.tup1)
        if self.tup2:
            res += "\n" + str(self.tup2)
        return res


class TupleNodeAppend(Node):
    def __init__(self, prev_tup, ele):
        super().__init__()
        self.prev_tup = prev_tup
        self.ele = ele
        self.prev_tup.parent = self
        self.ele.parent = self

    def eval(self):
        prev_l = list(self.prev_tup.eval())
        val = self.ele.eval()
        prev_l.append(val)
        return tuple(prev_l)

    def __str__(self):
        res = "\t" * self.parentCount() + "Tuple-Append"
        res += "\n" + str(self.prev_tup)
        res += "\n" + str(self.ele)
        return res


class TupleIndex(Node):
    def __init__(self, prev_tuple, ele):
        super().__init__()
        self.prev_tuple = prev_tuple
        self.ele = ele
        self.prev_tuple.parent = self
        self.ele.parent = self

    def eval(self):
        prev_tup = self.prev_tuple.eval()
        # tuple starts from 0. But given index starts from 1
        index = self.ele.eval()
        self.typecheck(prev_tup, index)
        index = index - 1

        return prev_tup[index]

    def typecheck(self, tup, index):
        if type(self.ele) not in [Variable, Value]:
            raise SyntaxException("tuple indices must be integers not expression")

        if type(index) != int:
            raise SyntaxException("tuple indices must be integers not {}".format(type(index)))

        if type(tup) != tuple:
            raise SemanticException("Operand {} must be of type tuple".format(tup))

        if (index - 1) >= len(tup) or (index - 1) < 0:
            raise SemanticException("tuple index out of range")

    def __str__(self):
        res = "\t" * self.parentCount() + "Tuple-Indexing"
        res += "\n" + str(self.prev_tuple)
        res += "\n" + str(self.ele)
        return res


global_variables = [{}]


class Variable(Node):
    def __init__(self, var):
        super().__init__()
        self.name = var

    def eval(self):
        self.typecheck()

        local_scope_var = global_variables[0]
        return local_scope_var[self.name]

    def typecheck(self):
        local_scope_var = global_variables[0]
        if self.name not in local_scope_var:
            raise SemanticException("Variable {} not found".format(self.name))

        if not isinstance(self.name, str):
            raise SyntaxException("Variable {} name is not string".format(self.name))

    def __str__(self):
        res = "\t" * self.parentCount() + str(self.name)
        return res


class AssignmentVar(Node):
    def __init__(self, var, expr):
        super().__init__()
        self.var = var
        self.expr = expr
        self.var.parent = self
        self.expr.parent = self

    def eval(self):
        rvalue = self.expr.eval()
        self.typecheck(rvalue)

        lvalue = self.var.name
        global_variables[0][lvalue] = rvalue

    def typecheck(self, rval):
        members = [int, float, str, list, bool, tuple]
        if type(rval) not in members:
            raise SemanticException("Right value {} is not in type {}".format(rval, members))

    def __str__(self):
        res = "\t" * self.parentCount() + "Variable-Assignment"
        res += "\n" + str(self.var)
        res += "\n" + str(self.expr)
        return res


class AssignmentList(Node):
    def __init__(self, var, expr):
        super().__init__()
        self.var = var
        self.expr = expr
        self.var.parent = self
        self.expr.parent = self

    def eval(self):
        rvalue = self.expr.eval()
        self.typecheck(self.var, rvalue)
        var = self.var
        list_index = []
        while isinstance(var, ListIndex):
            list_index.append(var.ele.eval())
            var = var.prev_list

        list_name = var.name

        local_scope_var = global_variables[0]
        prev_list = local_scope_var[list_name]

        update_list = prev_list
        for i in reversed(list_index[1:]):
            update_list = update_list[i]
        update_list[list_index[0]] = rvalue

    def typecheck(self, lval, rval):
        lval.eval()
        members = [int, float, str, list, bool, tuple]
        if type(rval) not in members:
            raise SemanticException("Right value {} is not in type {}".format(rval, members))
        if not isinstance(lval, ListIndex):
            raise SemanticException("Left value cannot be of type {}".format(type(lval)))

    def __str__(self):
        res = "\t" * self.parentCount() + "List-Assignment"
        res += "\n" + str(self.var)
        res += "\n" + str(self.expr)
        return res


class Print(Node):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr
        self.expr.parent = self

    def eval(self):
        print(self.expr.eval())

    def __str__(self):
        res = "\t" * self.parentCount() + "Print-Statement"
        res += "\n" + str(self.expr)
        return res


class Block(Node):
    def __init__(self, statements):
        super().__init__()
        self.statements = statements

    def eval(self):
        for statement in self.statements:
            if statement is not None:
                statement.eval()

    def __str__(self):
        res = "\t" * self.parentCount() + "Block"
        res += "\n" + str(self.statements)
        return res


class IfBlock(Node):
    def __init__(self, condition, statements):
        super().__init__()
        self.condition = condition
        self.condition.parent = self
        self.statements = None
        if statements:
            self.statements = statements
            self.statements.parent = self

    def eval(self):
        condition = self.condition.eval()
        self.typecheck(condition)
        if self.statements is None:
            return
        if condition:
            self.statements.eval()

    def typecheck(self, condition):
        if not type(condition) == bool:
            raise SemanticException("If Condition has to be of type bool but is of type {}".format(type(condition)))

    def __str__(self):
        res = "\t" * self.parentCount() + "If-Block"
        res += "\n" + str(self.condition)
        res += "\n" + str(self.statements)
        return res


class IfElseBlock(Node):
    def __init__(self, condition, if_statements, else_statements):
        super().__init__()
        self.condition = condition
        self.condition.parent = self
        self.if_statements = None
        self.else_statements = None
        if if_statements:
            self.if_statements = if_statements
            self.if_statements.parent = self
        if else_statements:
            self.else_statements = else_statements
            self.else_statements.parent = self

    def eval(self):
        condition = self.condition.eval()
        self.typecheck(condition)
        if condition:
            if self.if_statements:
                self.if_statements.eval()
        else:
            if self.else_statements:
                self.else_statements.eval()

    def typecheck(self, condition):
        if not type(condition) == bool:
            raise SemanticException(
                "If-Else Condition has to be of type bool but is of type {}".format(type(condition)))

    def __str__(self):
        res = "\t" * self.parentCount() + "If-Else-Block"
        res += "\n" + str(self.condition)
        res += "\n" + str(self.if_statements)
        res += "\n" + str(self.else_statements)
        return res


class WhileBlock(Node):
    def __init__(self, condition, statements):
        super().__init__()
        self.condition = condition
        self.condition.parent = self
        self.statements = None
        if statements:
            self.statements = statements
            self.statements.parent = self

    def eval(self):
        condition = self.condition.eval()
        self.typecheck(condition)
        if self.statements is None:
            return
        while condition:
            self.statements.eval()
            condition = self.condition.eval()

    def typecheck(self, condition):
        if not type(condition) == bool:
            raise SemanticException("While Condition has to be of type bool but is of type {}".format(type(condition)))

    def __str__(self):
        res = "\t" * self.parentCount() + "While-Block"
        res += "\n" + str(self.condition)
        res += "\n" + str(self.statements)
        return res


class SBML(Node):
    def __init__(self, funcs, block):
        super().__init__()
        self.block = block
        if block:
            self.block.parent = self
        self.funcs = funcs

    def eval(self):
        if self.funcs:
            for fun in self.funcs:
                fun.eval()
        if self.block:
            self.block.eval()

    def __str__(self):
        res = "\t" * self.parentCount() + "Main-Prog"
        if self.block:
            res += "\n" + str(self.block)
        return res


global_functions = {}


class FuncDef(Node):
    def __init__(self, func_name, func_args, func_body, ret_stat):
        self.func_name = func_name
        self.func_body = func_body
        self.func_args = func_args
        self.func_ret = ret_stat
        self.func_ret.parent = self
        if func_body:
            self.func_body.parent = self

    def eval(self):
        global_functions[self.func_name] = self

    def __str__(self):
        res = "\t" * self.parentCount() + "Function-Def"
        res += "\n" + str(self.func_name)
        res += "\n" + str(self.func_ret)
        return res


class FuncExec(Node):
    def __init__(self, func_name, func_args):
        self.func_name = func_name
        self.func_args = func_args

    def eval(self):
        self.typecheck()
        fun = global_functions[self.func_name]
        local_func_stack = {}

        if self.func_args:
            for i, arg in enumerate(self.func_args):
                arg_value = arg.eval()
                if isinstance(arg_value, list):
                    arg_value = arg_value.copy()
                local_func_stack[fun.func_args[i]] = arg_value

        global_variables.insert(0, local_func_stack)
        if fun.func_body:
            fun.func_body.eval()
        output = fun.func_ret.eval()
        global_variables.pop(0)
        return output

    def typecheck(self):
        if self.func_name not in global_functions:
            raise SemanticException("Function {} not found".format(self.func_name))

        fun = global_functions[self.func_name]
        if not fun.func_args:
            fun_arg_len = 0
        else:
            fun_arg_len = len(fun.func_args)
        if not self.func_args:
            pass_arg_len = 0
        else:
            pass_arg_len = len(self.func_args)
        if fun_arg_len != pass_arg_len:
            raise SemanticException(
                "Number of arguments passed : {} doesnt match number of parameters of the function : {}".format(
                    pass_arg_len, fun_arg_len))

    def __str__(self):
        res = "\t" * self.parentCount() + "Function-Execution"
        res += "\n" + str(self.func_name)
        if self.func_args:
            res += "\n" + str(self.func_args)
        return res


def p_prog_start(p):
    """sbml : functions block"""
    p[0] = SBML(p[1], p[2])


def p_prog_no_func_start(p):
    """sbml : block"""
    p[0] = SBML(None, p[1])


def p_functions(p):
    """functions : functions function"""
    p[0] = p[1] + [p[2]]


def p_functions_func(p):
    """functions : function"""
    p[0] = [p[1]]


def p_function_def(p):
    """function : FUNC VAR L_PAREN args R_PAREN ASSIGN block expression SEMICOLON"""
    p[0] = FuncDef(p[2], p[4], p[7], p[8])


def p_function_no_args(p):
    """args : """
    p[0] = None


def p_function_args(p):
    """args : args COMMA VAR"""
    p[0] = p[1] + [p[3]]


def p_function_arg(p):
    """args : VAR"""
    p[0] = [p[1]]


def p_block_start(p):
    """block : L_CURL statements R_CURL"""
    p[0] = Block(p[2])


def p_block_empty(p):
    """block : L_CURL R_CURL"""
    p[0] = None


def p_block_statements(p):
    """statements : statements statement"""
    p[0] = p[1] + [p[2]]


def p_statements_statement(p):
    """statements : statement"""
    p[0] = [p[1]]


def p_statement_expression(p):
    """statement : expression SEMICOLON"""
    p[0] = p[1]


def p_function_call_no_arg(p):
    """expression : VAR L_PAREN R_PAREN"""
    p[0] = FuncExec(p[1], None)


def p_function_call_all_args(p):
    """expression : VAR L_PAREN arguments R_PAREN"""
    p[0] = FuncExec(p[1], p[3])


def p_arguments(p):
    """arguments : arguments COMMA expression"""
    p[0] = p[1] + [p[3]]


def p_argument(p):
    """arguments : expression"""
    p[0] = [p[1]]


def p_statement_block(p):
    """statement : block"""
    p[0] = p[1]


def p_statement_assign(p):
    """statement : expression ASSIGN expression SEMICOLON"""
    if isinstance(p[1], Variable):
        p[0] = AssignmentVar((p[1]), p[3])
    else:
        p[0] = AssignmentList(p[1], p[3])


def p_statement_print(p):
    """statement : PRINT L_PAREN expression R_PAREN SEMICOLON"""
    p[0] = Print(p[3])


def p_if_block(p):
    """statement : IF L_PAREN expression R_PAREN block"""
    p[0] = IfBlock(p[3], p[5])


def p_if_else_block(p):
    """statement : IF L_PAREN expression R_PAREN block ELSE block"""
    p[0] = IfElseBlock(p[3], p[5], p[7])


def p_while_block(p):
    """statement : WHILE L_PAREN expression R_PAREN block"""
    p[0] = WhileBlock(p[3], p[5])


# expr --> ( expr )
def p_expression_group(p):
    """expression : L_PAREN expression R_PAREN"""
    p[0] = p[2]


# expr bin_op expr
def p_expression_bin_op(p):
    """expression : expression EXPONENT expression
                | expression MUL expression
                | expression DIV expression
                | expression INT_DIV expression
                | expression MOD expression
                | expression SUB expression """
    p[0] = BinaryOperation(p[1], p[3], p[2])


# expr + expr
def p_expression_add_op(p):
    """expression : expression ADD expression """
    p[0] = AdditionOperation(p[1], p[3], p[2])


# expr binary_comparison expr
def p_expression_comp_op(p):
    """expression : expression LT expression
                | expression LE expression
                | expression EQUALS expression
                | expression NE expression
                | expression GE expression
                | expression GT expression """
    p[0] = BinaryComparison(p[1], p[3], p[2])


# expr boolean_operator expr
def p_expression_bool_op(p):
    """expression : expression CONJ expression
                | expression DISJ expression """
    p[0] = BooleanOperation(p[1], p[3], p[2])


# expr --> - expr
def p_expr_uminus(p):
    """expression : SUB expression %prec UMINUS"""
    p[0] = Value(-p[2].eval())


# expr --> value
def p_expression_term(p):
    """expression : value"""
    p[0] = p[1]


# int, real, bool, str
def p_factor_value(p):
    """value : INT_NUM
            | REAL_NUM
            | BOOLEAN
            | STRING"""
    p[0] = Value(p[1])


def p_expression_var(p):
    """expression : VAR"""
    p[0] = Variable(p[1])


# not expr
def p_expression_negation(p):
    """expression : NEG expression"""
    p[0] = Negation(p[2])


# expr --> list
def p_expression_list(p):
    """expression : list"""
    p[0] = p[1]


# [ list ]
def p_list_element(p):
    """list : L_BRAC list_ele R_BRAC"""
    p[0] = p[2]


# [ list_ele,]
def p_list_ele_comma(p):
    """list_ele : list_ele COMMA"""
    p[0] = p[1]


# list_ele --> expr or empty
def p_list_creation(p):
    """list_ele : expression
            | """
    if len(p) > 1:
        p[0] = ListNode(p[1])
    else:
        p[0] = ListNode(None)


# list_ele --> list_ele , expr
def p_list_append(p):
    """list_ele : list_ele COMMA expression"""
    p[0] = ListNodeAppend(p[1], p[3])


# #1([4],)[0] or ("J"+"H")[0] or "JH"[0] or [1,2][0]
def p_list_index(p):
    """expression : expression index"""
    p[0] = ListIndex(p[1], p[2])


# [list][(expr)]
def p_expression_index(p):
    """index : L_BRAC expression R_BRAC"""
    p[0] = p[2]


def p_list_cons(p):
    """expression : expression CONS expression"""
    p[0] = ListCons(p[1], p[3])


def p_list_str_search(p):
    """expression : expression MEMBER expression"""
    p[0] = ListStrSearch(p[1], p[3])


# expr --> tuple
def p_expression_tuple(p):
    """expression : tuple"""
    p[0] = p[1]


# ( tuple )
def p_tuple_element(p):
    """tuple : L_PAREN tup_ele R_PAREN"""
    p[0] = p[2]


# ( tuple ,)
def p_tup_ele_comma(p):
    """tup_ele : tup_ele COMMA"""
    p[0] = p[1]


# ( exp, ) or ( exp1, exp2)
def p_tuple_creation(p):
    """tup_ele : expression COMMA
            | expression COMMA expression"""
    if len(p) == 3:
        p[0] = TupleNode(p[1], None)
    else:
        p[0] = TupleNode(p[1], p[3])


# (exp1, exp2, expr3,...) --> starts appending only from 3 element. For ele1, ele2 p_tuple_creation is called
def p_tuple_append(p):
    """tup_ele : tup_ele COMMA expression"""
    p[0] = TupleNodeAppend(p[1], p[3])


#  #expression tuple
def p_tuple_index(p):
    """tuple : tup_index tuple"""
    p[0] = TupleIndex(p[2], p[1])


def p_tuple_index_var(p):
    """tuple : tup_index VAR"""
    p[0] = TupleIndex(Variable(p[2]), p[1])


# #1([(1,)][0])
# def p_tuple_index_expression(p):
#     """tuple : tup_index L_PAREN expression R_PAREN"""
#     p[0] = TupleIndex(p[3], p[1])


def p_expression_tup_index(p):
    """tup_index : TUPLE_HASH expression"""
    p[0] = p[2]


def p_error(p):
    if p is not None:
        raise SyntaxException("Syntax error at '%s' (%d, %d)" % (p.value, p.lineno, p.lexpos))
    else:
        raise SyntaxException("Syntax Error. Unexpected end of input")


# def tokenize(inp):
#     lexer.input(inp)
#     while True:
#         tok = lexer.token()
#         if not tok:
#             break
#         print(tok)


lexer = lex.lex(debug=False)
parser = yacc.yacc(debug=True)


def main():
    try:
        file_name = sys.argv[1]
        with open(file_name, 'r') as file:
            data = file.read().replace('\n', '')
    except Exception as e:
        print("Error while reading file: ", e)
        sys.exit(1)

    try:
        result = parser.parse(data, debug=0)
        if result is not None:
            result.eval()
    except SemanticException as e:
        print("SEMANTIC ERROR : ", e)
    except SyntaxException as e:
        print("SYNTAX ERROR : ", e)
    except Exception as e:
        print("UNKNOWN ERROR: ", e)


if __name__ == "__main__":
    main()
