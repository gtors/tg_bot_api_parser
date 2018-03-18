#!/usr/bin/env python3

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

# std
import re
import io
import logging
import textwrap
from typing import *
from functools import lru_cache

# 3rdparty
import requests
from lxml import etree
from lxml.etree import Element as Elem
import ipdb

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------


TYPE_DESC_KEYWORDS = (
    'this object',
    'represent',
    'contain'
)

METHOD_DESC_KEYWORDS = (
    'use this method',
    'a simple method',
)

METHOD_RETURN_TYPE_KEYWORDS = (
    'is returned',
    'object is returned',
    'On success',
    'Returns',
)

TYPE_RENAME_RULES = {
    'CallbackGame': 'String',
    'Bool': 'Boolean',
    'True': 'Boolean',
    'true': 'Boolean',
    'False': 'Boolean',
    'Float number': 'Float',
    'Str': 'String',
    'Int': 'Integer',
}

RUST_LINE_BOUND = 80

RUST_FIELDS_RENAME_RULES = {
    'type': 'ty',
}

#: Signature consumation return codes
SIGN_NEXT = 0
SIGN_MATCHED = 1
SIGN_STOP = 2

#------------------------------------------------------------------------------
# Logger configuration
#------------------------------------------------------------------------------

log = logging.getLogger('console')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


#------------------------------------------------------------------------------
# API types extraction logic
#------------------------------------------------------------------------------

#: As alternative for ADT
String = NamedTuple('String', [])
Integer = NamedTuple('Integer', [])
Float = NamedTuple('Float', [])
Bool = NamedTuple('Bool', [])
Array = NamedTuple('Array', [('of', Any)])
Ref = NamedTuple('Ref', [('name', str), ('recursive', bool)])
FieldType = Union[String, Integer, Float, Bool, Array, Ref]


class Field(NamedTuple):
    name: str
    description: str
    types: List[FieldType]
    optional: bool


class ApiType(NamedTuple):
    name: str
    description: str
    #: Used for regular type only
    fields: List[Field]
    #: Used for enum type only
    kinds: List[Ref]


class ApiTypeSignature:
    """ HTML signature of API Type.

    The API Type present in HTML in follow order:
    * h4 - title with anchor,
    * p - desciption,
    * blockquote - additional description,
    * table/ul - detailed structure / types enumeration.
    """
    h4: Elem = None
    p: Elem = None
    blockquote: Optional[Elem] = None
    table: Optional[Elem] = None
    ul: Optional[Elem] = None

    def consume(self, elem: Elem) -> int:
        if self._is_name(elem):
            self.h4 = elem
            return SIGN_NEXT

        if self._is_desc(elem):
            self.p = elem
            return SIGN_NEXT

        if self._is_additional_desc(elem):
            self.blockquote = elem
            return SIGN_NEXT

        if self._is_enum(elem):
            self.ul = elem
            return SIGN_MATCHED

        if self._is_struct(elem):
            self.table = elem
            return SIGN_MATCHED

        self.clear()
        return SIGN_STOP

    def clear(self):
        self.h4 = None
        self.p = None
        self.table = None
        self.ul = None
        self.matched = False

    def _is_name(self, elem: Elem) -> bool:
        if self.h4 is None and elem.tag == 'h4':
            text = plain_text(elem)
            uppercased = text and text[0].isupper()
            has_anchor = match_xpath(elem, './a[starts-with(@href, "#")]')
            return uppercased and has_anchor
        else:
            return False

    def _is_desc(self, elem: Elem) -> bool:
        if self.h4 is not None and elem.tag == 'p':
            text = plain_text(elem).lower()
            return any((keyword in text) for keyword in TYPE_DESC_KEYWORDS)
        else:
            return False

    def _is_additional_desc(self, elem: Elem) -> bool:
        return self.p is not None and elem.tag == 'blockquote'

    def _is_struct(self, elem: Elem) -> bool:
        if self.p is not None and elem.tag == 'table':
            has_header = lambda hd: match_xpath(elem, f'.//td/strong[. = "{hd}"]')
            return (
                has_header('Field') and
                has_header('Type') and
                has_header('Description')
            )
        else:
            return False

    def _is_enum(self, elem: Elem) -> bool:
        return self.p is not None and elem.tag == 'ul'


class Context:
    types_repository: Dict[str, FieldType] = {}
    current_type_name: str


def extract_type(ctx: Context, sign: ApiTypeSignature) -> ApiType:
    name = plain_text(sign.h4)
    desc = plain_text(sign.p)
    ctx.current_type_name = name
    if sign.ul is not None:
        return ApiType(
            name=name,
            description=desc,
            fields=None,
            kinds=[
                Ref(name=type_name.strip(), recursive=False)
                for type_name in sign.ul.xpath('.//a/text()')
            ]
        )
    else:
        return ApiType(
            name=name,
            description=desc,
            fields=extract_type_fields(ctx, sign.table),
            kinds=None,
        )


def extract_type_fields(ctx: Context, table: Elem) -> List[Field]:
    fields: List[Field] = []
    # First row skipped because it contains headers
    for tr in table.xpath('.//tr[position()>1]'):
        #try:
        (td1, td2, td3) = tr.getchildren()
        #except ValueError:
            # Sometimes, tr contain just a single td.
        #    continue

        name = plain_text(td1).strip()
        types = plain_text(td2).strip()
        desc = plain_text(td3).strip()
        fields.append(Field(
            name=name,
            optional=desc.startswith('Optional'),
            description=desc,
            types=[
                determine_field_type(ctx, x)
                for x in types.split(' or ')
            ]
        ))
    return fields


@lru_cache(maxsize=1024)
def determine_field_type(ctx: Context, raw: str) -> Type:
    raw = raw.strip()
    raw = TYPE_RENAME_RULES.get(raw, raw)
    if raw == 'String':
        return String()
    if raw == 'Integer':
        return Integer()
    if raw == 'Boolean':
        return Bool()
    if raw == 'Float':
        return Float()
    if raw.startswith('Array of'):
        raw = raw.replace('Array of', '', 1)
        return Array(of=determine_field_type(ctx, raw.strip()))
    if raw[0].isupper():
        recursive=(ctx.current_type_name == raw)
        return Ref(name=raw, recursive=recursive)
    raise RuntimeError(f"Unknown type: {raw}")


def iter_types(ctx: Context, html: bytes) -> Iterator[ApiType]:
    doc = etree.iterwalk(
        etree.HTML(html),
        events=("start",),
        tag=("h4"),
    )

    sign = ApiTypeSignature()

    for (action, elem) in doc:
        if sign.consume(elem) != SIGN_NEXT:
            continue

        for sibl in elem.itersiblings():
            resp = sign.consume(sibl)

            if resp == SIGN_MATCHED:
                ty = extract_type(ctx, sign)
                ctx.types_repository[ty.name] = ty
                yield ty
                sign.clear()
                break

            if resp == SIGN_STOP:
                break


def match_xpath(elem: Elem, xpath: str) -> bool:
    return len(elem.xpath(xpath)) == 1


def plain_text(elem: Elem) -> str:
    return ''.join(filter(None, elem.itertext())).strip()

#------------------------------------------------------------------------------
# API methods extraction logic
#------------------------------------------------------------------------------

class ApiMethod(NamedTuple):
    name: str
    description: str
    params: List[Field]
    rets: List[Type]


class ApiMethodSignature:
    """ HTML signature of API method

    The API Method present in HTML in follow order:
    * h4 - title,
    * p - description,
    * blockquote - additional desciption,
    * table - params description
    """

    h4: Elem = None
    p: Elem = None
    blockquote: Optional[Elem] = None
    table: Optional[Elem] = None

    def consume(self, elem: Elem) -> int:
        if self._is_name(elem):
            self.h4 = elem
            return SIGN_NEXT

        if self._is_desc(elem):
            self.p = elem
            return SIGN_NEXT

        if self._is_additional_desc(elem):
            self.blockquote = elem
            return SIGN_NEXT

        # Description continuation after blockquote. So far, we will skip this.
        if (self.blockquote is not None or self.p is not None) and elem.tag == 'p':
            return SIGN_NEXT

        if self._is_params(elem):
            self.table = elem
            return SIGN_MATCHED

        # Sometimes method may be without params
        if self.h4 is not None and self.p is not None:
            return SIGN_MATCHED

        self.clear()
        return SIGN_STOP

    def clear(self):
        self.h4 = None
        self.p = None
        self.blockquote = None
        self.table = None

    def _is_name(self, elem: Elem) -> bool:
        if self.h4 is None and elem.tag == 'h4':
            text = plain_text(elem)
            lowercased = text and text[0].islower()
            has_anchor = match_xpath(elem, './a[starts-with(@href, "#")]')
            return lowercased and has_anchor
        else:
            return False

    def _is_desc(self, elem: Elem) -> bool:
        if self.h4 is not None and elem.tag == 'p':
            text = plain_text(elem).lower()
            return any((kw in text) for kw in METHOD_DESC_KEYWORDS)
        else:
            return False

    def _is_additional_desc(self, elem: Elem) -> bool:
        return self.p is not None and elem.tag == 'blockquote'

    def _is_params(self, elem: Elem) -> bool:
        if self.p is not None and elem.tag == 'table':
            has_header = lambda hd: match_xpath(elem, f'.//td/strong[. = "{hd}"]')
            return (
                has_header('Parameters') and
                has_header('Type') and
                has_header('Required') and
                has_header('Description')
            )
        else:
            return False


def extract_return_types(ctx: Context, elem: Elem) -> List[Type]:
    desc = plain_text(elem)
    # Find line which contains return type description
    s = next(s for s in desc.split('.') if any((kw in s) for kw in METHOD_RETURN_TYPE_KEYWORDS))
    # Strip all except type names and commas
    type_names = re.findall(r'((?:Array of|False|True|\btrue|Str|Int|Bool|Float|,|' + '|'.join(ctx.types_repository.keys()) + ')\w*)', s)
    # Remove redundant commas and spaces
    type_names = ' '.join(type_names).strip(' ,').split(' , ')
    return [determine_field_type(ctx, t) for t in type_names]


def extract_method_params(ctx: Context, table: Elem) -> List[Field]:
    fields: List[Field] = []

    if table is None:
        return fields

    # First row skipped because it contains headers
    for tr in table.xpath('.//tr[position()>1]'):
        (td1, td2, td3, td4) = tr.getchildren()
        name = plain_text(td1).strip()
        types = plain_text(td2).strip()
        required = plain_text(td3).strip()
        desc = plain_text(td4).strip()

        fields.append(Field(
            name=name,
            optional=(required == 'Optional'),
            description=desc,
            types=[
                determine_field_type(ctx, x)
                for x in types.split(' or ')
            ]
        ))

    return fields


def extract_method(ctx: Context, sign: ApiMethodSignature) -> ApiMethod:
    return ApiMethod(
        name=plain_text(sign.h4),
        description=plain_text(sign.p),
        params=extract_method_params(ctx, sign.table),
        rets=extract_return_types(ctx, sign.p))


def iter_methods(ctx: Context, html: bytes) -> Iterator[ApiType]:
    doc = etree.iterwalk(
        etree.HTML(html),
        events=("start",),
        tag=("h4"),
    )

    sign = ApiMethodSignature()

    for (_, elem) in doc:
        if sign.consume(elem) != SIGN_NEXT:
            continue

        for sibl in elem.itersiblings():
            resp = sign.consume(sibl)

            if resp == SIGN_MATCHED:
                method = extract_method(ctx, sign)
                yield method
                sign.clear()
                break

            if resp == SIGN_STOP:
                break

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

def doc_line_wrap(s: str, *, indent: int = 0) -> str:
    spaces = ' ' * indent
    return f'{spaces}/// ' + f'\n{spaces}/// '.join(textwrap.wrap(s, RUST_LINE_BOUND - indent))

#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

def main():
    html_doc = requests.get('https://core.telegram.org/bots/api').content
    ctx = Context()
    types = tuple(iter_types(ctx, html_doc))
    methods = iter_methods(ctx, html_doc)
   
    for m in methods:
        print()
        print(m.name)
        for p in m.params:
            print(f'    {p.name}')
            print(f'         {p.types}')
        print(f' -> {m.rets}')

    #print(generate_rust_types(types))

    #methods = extract_methods(html_doc, types)
    #generate_rust_module(types, methods)

if __name__ == '__main__':
    main()
