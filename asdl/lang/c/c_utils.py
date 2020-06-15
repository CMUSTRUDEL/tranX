import functools
from typing import Dict, Iterator, List, Type

from pycparser.c_ast import Node as ASTNode
from pycparser.c_lexer import CLexer as _CLexer

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField

__all__ = [
    "ASTConverter",
    "c_ast_to_asdl_ast",
    "asdl_ast_to_c_ast",
    "get_c_ast_node_class",
    "KEYWORDS",
    "OPERATORS",
    "RESERVED_WORDS",
    "LexToken",
    "CLexer",
    "SPM_SPACE",
]

SPM_SPACE = "▁"

AVAILABLE_NODES: Dict[str, Type[ASTNode]] = {klass.__name__: klass for klass in ASTNode.__subclasses__()}


def get_c_ast_node_class(name: str) -> Type[ASTNode]:
    return AVAILABLE_NODES[name]


C_TO_ASDL_TERMINAL_MAP = {
    "QUAL": {
        "const": "Const",
        "volatile": "Volatile",
        "restrict": "Restrict",
    },
    "DIM_QUAL": {
        "const": "ConstDim",
        "volatile": "VolatileDim",
        "restrict": "RestrictDim",
        "static": "StaticDim",
    },
    "STORAGE": {
        "extern": "Extern",
        "static": "Static",
        "register": "Register",
        "auto": "Auto",
        "typedef": "TypedefStorage",
    },
    "FUNCSPEC": {
        "inline": "Inline",
    },
    "REF_TYPE": {
        "->": "Arrow",
        ".": "Dot",
    },
    "ASSIGN_OPER": {
        "=": "NormalAssign",
        "+=": "AddAssign",
        "-=": "SubAssign",
        "*=": "MulAssign",
        "/=": "DivAssign",
        "%=": "ModAssign",
        "&=": "BitAndAssign",
        "|=": "BitOrAssign",
        "^=": "BitXorAssign",
        "<<=": "LShiftAssign",
        ">>=": "RShiftAssign",
    },
    "UNARY_OP": {
        "+": "UAdd",
        "-": "USub",
        "p++": "PreInc",
        "++": "PostInc",
        "p--": "PreDec",
        "--": "PostDec",
        "!": "Not",
        "~": "BitNot",
        "*": "Deref",
        "sizeof": "SizeOf",
        "&": "AddressOf"
    },
    "BINARY_OP": {
        "=": "Assign",
        "+": "Add",
        "-": "Sub",
        "*": "Mul",
        "/": "Div",
        "%": "Mod",
        "==": "Eq",
        "!=": "NotEq",
        "<": "Lt",
        "<=": "LtE",
        ">": "Gt",
        ">=": "GtE",
        "&&": "And",
        "||": "Or",
        "&": "BitAnd",
        "|": "BitOr",
        "^": "BitXor",
        "<<": "LShift",
        ">>": "RShift",
    },
    "LITERAL_TYPE": {
        "int": "IntLiteral",
        "long int": "IntLiteral",
        "long long int": "IntLiteral",
        "unsigned int": "IntLiteral",
        "unsigned long int": "IntLiteral",
        "unsigned long long int": "IntLiteral",
        "float": "FloatLiteral",
        "double": "FloatLiteral",
        "long double": "FloatLiteral",
        "char": "CharLiteral",
        "string": "StringLiteral",
    },
}

ASDL_TO_C_TERMINAL_MAP = {
    key: {typ: tok for tok, typ in tok_map.items()}
    for key, tok_map in C_TO_ASDL_TERMINAL_MAP.items()
}
ASDL_TO_C_TERMINAL_MAP["LITERAL_TYPE"] = {
    "IntLiteral": "int",
    "FloatLiteral": "double",
    "CharLiteral": "char",
    "StringLiteral": "string",
}

# Keywords come from the `pycparser` lexer keyword list.
KEYWORDS = list(_CLexer.keyword_map.keys())
# Operators are the terminals that do not begin with letters.
OPERATORS = [tok for tok_map in C_TO_ASDL_TERMINAL_MAP.values()
             for tok in tok_map.keys() if not tok[0].isalpha()]
RESERVED_WORDS = KEYWORDS + OPERATORS + ["(", ")", "[", "]", "{", "}", ",", ";", ":", "...", "?"]


class LexToken:  # stub
    type: str
    value: str
    lineno: int
    lexpos: int


class CLexer:
    @staticmethod
    def _error_func(msg, loc0, loc1):
        pass

    @staticmethod
    def _brace_func():
        pass

    @staticmethod
    def _type_lookup_func(typ):
        return False

    def __init__(self) -> None:
        self.lexer = _CLexer(self._error_func, self._brace_func, self._brace_func, self._type_lookup_func)
        self.lexer.build(optimize=True, lextab='pycparser.lextab')

    def lex_tokens(self, code: str) -> Iterator[LexToken]:
        self.lexer.reset_lineno()
        self.lexer.input(code)
        while True:
            token = self.lexer.token()
            if token is None:
                break
            yield token

    def lex(self, code: str) -> List[str]:
        return [token.value for token in self.lex_tokens(code)]


UNFILLED = "@unfilled@"


class ASTConverter:
    def __init__(self, grammar: ASDLGrammar):
        self.grammar = grammar

    def c_ast_to_asdl_ast(self, ast_node: ASTNode) -> AbstractSyntaxTree:
        # Node should be composite.
        node_type = type(ast_node).__name__
        production = self.grammar.get_prod_by_ctr_name(node_type)

        fields = []
        for field in production.fields:
            field_value = getattr(ast_node, field.name)
            asdl_field = RealizedField(field)
            if field_value is not None:
                if field.cardinality != "multiple":
                    field_value = [field_value]
                if field.type.name == "EXPR":  # the only recursive type
                    for val in field_value:
                        child_node = self.c_ast_to_asdl_ast(val)
                        asdl_field.add_value(child_node)
                elif self.grammar.is_primitive_type(field.type):
                    for val in field_value:
                        asdl_field.add_value(str(val))
                else:
                    candidate_ctrs = C_TO_ASDL_TERMINAL_MAP[field.type.name]
                    for val in field_value:
                        asdl_field.add_value(AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(candidate_ctrs[val])))
            fields.append(asdl_field)
            if field.cardinality != "optional" and asdl_field.value is None:
                assert False

        asdl_node = AbstractSyntaxTree(production, realized_fields=fields)

        return asdl_node

    def asdl_ast_to_c_ast(self, asdl_node: AbstractSyntaxTree, ignore_error: bool = False) -> ASTNode:
        if ignore_error and not isinstance(asdl_node, AbstractSyntaxTree):
            return asdl_node
        klass = get_c_ast_node_class(asdl_node.production.constructor.name)
        kwargs = {}

        for field in asdl_node.fields:
            field_value = []
            if field.value is not None:
                values = field.as_value_list
                if field.type.name == "EXPR":
                    for val in values:
                        node = self.asdl_ast_to_c_ast(val, ignore_error)
                        field_value.append(node)
                elif self.grammar.is_primitive_type(field.type):
                    field_value = values
                else:
                    candidate_tokens = ASDL_TO_C_TERMINAL_MAP[field.type.name]
                    field_value = [candidate_tokens[val.production.constructor.name]
                                   if isinstance(val, AbstractSyntaxTree) else val for val in values]

            if field.cardinality == "single":
                if not ignore_error:
                    assert len(field_value) == 1
                    field_value = field_value[0]
                else:
                    field_value = field_value[0] if len(field_value) > 0 else UNFILLED
            elif field.cardinality == "optional":
                if not ignore_error:
                    assert len(field_value) <= 1
                field_value = field_value[0] if len(field_value) > 0 else None
            kwargs[field.name] = field_value

        if klass.__name__ == "Struct":
            # The conversion has some quirks with the `decls` field of `Struct`. In actual ASTs, `decls` can either
            # be a list of `EXPR` (a `struct` definition with members), or `None` (a declaration, as in
            # `struct MyStruct foo;`).
            # This is more of an abuse of the syntax, and the ASDL AST class does not support having `None` for a
            # multiple field, and it's implicitly converted into an empty list.
            # However, the C standard doesn't allow empty `struct`s, so we can just assume an empty `decls` list means
            # `None`. (See https://stackoverflow.com/questions/755305/empty-structure-in-c for details)
            if kwargs["decls"] == []:
                kwargs["decls"] = None

        return klass(**kwargs)


@functools.lru_cache(maxsize=8)
def _get_converter_instance(grammar: ASDLGrammar):
    return ASTConverter(grammar)


def c_ast_to_asdl_ast(ast_node: ASTNode, grammar: ASDLGrammar) -> AbstractSyntaxTree:
    return _get_converter_instance(grammar).c_ast_to_asdl_ast(ast_node)


def asdl_ast_to_c_ast(asdl_node: AbstractSyntaxTree, grammar: ASDLGrammar, ignore_error: bool = False) -> ASTNode:
    return _get_converter_instance(grammar).asdl_ast_to_c_ast(asdl_node, ignore_error=ignore_error)
