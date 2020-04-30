from typing import Dict, Type

from pycparser.c_ast import Node as ASTNode

from asdl.asdl_ast import AbstractSyntaxTree, RealizedField

__all__ = [
    "c_ast_to_asdl_ast",
    "asdl_ast_to_c_ast",
    "get_c_ast_node_class",
]

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
}

ASDL_TO_C_TERMINAL_MAP = {
    key: {typ: tok for tok, typ in tok_map.items()}
    for key, tok_map in C_TO_ASDL_TERMINAL_MAP.items()
}


def c_ast_to_asdl_ast(ast_node, grammar):
    # Node should be composite.
    node_type = type(ast_node).__name__
    production = grammar.get_prod_by_ctr_name(node_type)

    fields = []
    for field in production.fields:
        field_value = getattr(ast_node, field.name)
        asdl_field = RealizedField(field)
        if field_value is not None:
            if field.cardinality != "multiple":
                field_value = [field_value]
            if field.type.name == "EXPR":  # the only recursive type
                for val in field_value:
                    child_node = c_ast_to_asdl_ast(val, grammar)
                    asdl_field.add_value(child_node)
            elif field.type.name in ["IDENT", "STR"]:
                for val in field_value:
                    asdl_field.add_value(str(val))
            else:
                candidate_ctrs = C_TO_ASDL_TERMINAL_MAP[field.type.name]
                for val in field_value:
                    asdl_field.add_value(AbstractSyntaxTree(grammar.get_prod_by_ctr_name(candidate_ctrs[val])))
        fields.append(asdl_field)
        if field.cardinality != "optional" and asdl_field.value is None:
            assert False

    asdl_node = AbstractSyntaxTree(production, realized_fields=fields)

    return asdl_node


def asdl_ast_to_c_ast(asdl_node, grammar):
    klass = get_c_ast_node_class(asdl_node.production.constructor.name)
    kwargs = {}

    for field in asdl_node.fields:
        field_value = []
        if field.value is not None:
            values = field.value
            if field.cardinality != "multiple":
                values = [field.value]
            if field.type.name == "EXPR":
                for val in values:
                    node = asdl_ast_to_c_ast(val, grammar)
                    field_value.append(node)
            elif field.type.name in ["STR", "IDENT"]:
                field_value = values
            else:
                candidate_tokens = ASDL_TO_C_TERMINAL_MAP[field.type.name]
                field_value = [candidate_tokens[val.production.constructor.name] for val in values]

        if field.cardinality == "single":
            assert len(field_value) == 1
            field_value = field_value[0]
        elif field.cardinality == "optional":
            assert len(field_value) <= 1
            if len(field_value) == 1:
                field_value = field_value[0]
            else:
                field_value = None
        kwargs[field.name] = field_value

    return klass(**kwargs)
