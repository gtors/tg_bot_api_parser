#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

# std
from typing import *
from grabmodels import ApiType, ApiMethod

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

RUST_LINE_BOUND = 80

RUST_FIELDS_RENAME_RULES = {
    'type': 'ty',
}

#------------------------------------------------------------------------------
# Rust types generator
#------------------------------------------------------------------------------

def transform_to_rust_type(t: FieldType, optional: bool):
    tpl = 'Option<{}>' if optional else '{}'

    if isinstance(t, String):
        return tpl.format('String')
    if isinstance(t, Integer):
        return tpl.format('i64')
    if isinstance(t, Float):
        return tpl.format('f64')
    if isinstance(t, Bool):
        return tpl.format('bool')
    if isinstance(t, Array):
        rust_type = transform_to_rust_type(t.of, False)
        return tpl.format(f'Vec<{rust_type}>')
    if isinstance(t, Ref):
        return tpl.format(f'{t.name}')

    raise RuntimeError(f'Unknown type: {t}')


def generate_rust_struct(t: ApiType):
    out = []
    out.append(doc_line_wrap(t.description))
    out.append('#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]')
    out.append(f'pub struct {t.name} ' + '{')
    for f in t.fields:
        out.append(doc_line_wrap(f.description, indent=4))

        # TODO: Add handling multiple types
        if len(f.types) > 1:
            raise NotImplementedError("Multiple types not supported")

        ty = f.types[0]
        rust_type = transform_to_rust_type(ty, f.optional)
        rust_name = RUST_FIELDS_RENAME_RULES.get(f.name, f.name)
        if rust_name != f.name:
            out.append(f'    #[serde(rename = "{f.name}")]')
        if isinstance(ty, Ref) and ty.recursive:
            out.append(f'    pub {rust_name}: Box<{rust_type}>,')
        else:
            out.append(f'    pub {rust_name}: {rust_type},')
    out.append('}')
    return '\n'.join(out)


def generate_rust_enum(t: ApiType):
    out = []
    out.append(doc_line_wrap(t.description))
    out.append('#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]')
    out.append(f'pub enum {t.name} ' + '{')
    for r in t.kinds:
        out.append(f'    {r.name}({r.name}),')
    out.append('}')
    return '\n'.join(out)


def generate_rust_types(types: Iterator[ApiType]) -> str:
    out = []
    out.append(
        '#[macro_use]\n'
        'extern crate serde_derive;\n'
        'extern crate serde;\n'
        'extern crate serde_json;\n')
    for t in types:
        out.append('\n')
        if t.kinds is not None:
            out.append(generate_rust_enum(t))
        elif t.fields is not None:
            out.append(generate_rust_struct(t))
        else:
            raise RuntimeError(f"Unknown type: {t}")
    return '\n'.join(out)


def generate_rust_methods(methods: Iterator[ApiMethod]):
    pass


def doc_line_wrap(s: str, *, indent: int = 0) -> str:
    spaces = ' ' * indent
    return f'{spaces}/// ' + f'\n{spaces}/// '.join(textwrap.wrap(s, RUST_LINE_BOUND - indent))
