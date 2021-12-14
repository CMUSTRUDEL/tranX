from pycparser import c_parser
from asdl.asdl import ASDLGrammar
from asdl.lang.c.c_utils import *
from asdl.lang.c.c_transition_system import *
from asdl.hypothesis import *
import astor
# read in the grammar specification of Python 2.7, defined in ASDL
asdl_text = open('asdl/lang/c/c_asdl.txt').read()
grammar = ASDLGrammar.from_text(asdl_text)
c_code = """int main() { printf("Hello world!"); return; }"""
# get the (domain-specific) python AST of the example Python code snippet
c_ast = c_parser.CParser().parse(c_code)
# convert the python AST into general-purpose ASDL AST used by tranX
converter = ASTConverter(grammar)
asdl_ast = converter.c_ast_to_asdl_ast(c_ast)
print('String representation of the ASDL AST: \n%s' % asdl_ast.to_string())
print('Size of the AST: %d' % asdl_ast.size)
