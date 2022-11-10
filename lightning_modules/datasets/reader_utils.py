import os
import random
import logging

from functools import reduce
from math import gcd
from typing import List, Any, Union, Dict, Tuple

from tree_sitter import Parser, Language

# TODO: add those tokens to the vocab of whatever transformer models
INDENT_TOKEN = "@@INDENT@@"
NEWLINE_TOKEN = "@@NEWLINE@@"
END_OF_CELL_TOKEN = "<|endofcell|>"
COM_STMTS = ['if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement',
             'function_definition', 'class_definition']
PY_MODULES = ['module', 'block', 'decorated_definition']

logger = logging.getLogger('reader')

def get_indent_level(line: str) -> int:
    """return the indent level of the line, measuring the leading tabs or spaces"""
    code_start_idx = line.find(line.lstrip())

    if code_start_idx == 0:
        return 0

    indent_str = line[:code_start_idx]
    if ' ' in indent_str and '\t' in indent_str:
        logger.info('indentation string contains both spaces and tabs')

    if indent_str.replace(' ', '').replace('\t', '') != '':
        logger.info('indentation string is not all spaces or tabs')

    return len(indent_str)


def preprocess_cell(cell: Dict[str, Any], line_offset: int, add_special_token_for_code: bool=False) -> Tuple[List[str], Dict]:
    """preprocess the cell"""

    cell_type = cell['type']
    cell_lines = cell['lines']

    if not add_special_token_for_code:
        # special NEWLINE token will be added, so no need to add it here
        cell_lines = [line+'\n' for line in cell_lines]

    if 'stmts' in cell:
        stmts = cell['stmts']
    else:
        stmts = []

    for stmt in stmts:
        start, end = stmt['start_point'], stmt['end_point']
        stmt['start_point'] = (start[0]+line_offset, start[1])
        stmt['end_point'] = (end[0]+line_offset, end[1])

    if cell_type == 'markdown':
        commented_lines = [f"# {line}" for line in cell_lines]
        return commented_lines, stmts

    if cell_type == 'code' and add_special_token_for_code:
        # first figure out the indent levels
        indent_levels = [get_indent_level(line) for line in cell_lines]

        # there are cases where the indentations are actually line breakers for long line, 
        # use 20 as a threshold to distinguish between actual indent and line breakers
        indent_levels = [x if (x > 1 and x % 2 == 0 and x <= 20) else 0 for x in indent_levels]

        gcd_indent_level = reduce(gcd, indent_levels)
        if gcd_indent_level not in [0, 2, 4, 8]:
            # logger.info(f'indentation level is not a power of 2 but {gcd_indent_level}, setting all indentations to 0')
            indent_levels = [0] * len(indent_levels)
        if gcd_indent_level != 0:
            indent_levels = [i // gcd_indent_level for i in indent_levels]

        # add indentation and newline tokens
        lines = [' '.join([INDENT_TOKEN]*i + [line[i*gcd_indent_level:]] + [NEWLINE_TOKEN]) for i, line in zip(indent_levels, cell_lines)]
        return lines, stmts

    return cell_lines, stmts

def construct_from_lines(all_lines: List[str], start: Tuple[int, int], 
                         end: Tuple[int, int], is_byte_idx: bool=False):
    if is_byte_idx:
        start = (start[0], byte_idx_to_char_idx(start[1], all_lines[start[0]]))
        end = (end[0], byte_idx_to_char_idx(end[1], all_lines[end[0]]))

    # construct back the statement string
    statement_str = ''
    for i in range(start[0], end[0] + 1):
        if i == start[0]:
            if i == end[0]: # same line statement
                statement_str += (all_lines[i][start[1]:end[1]])
            else:
                statement_str += (all_lines[i][start[1]:])
        elif i == end[0]:
            statement_str += (all_lines[i][:end[1]])
        else:
            statement_str += all_lines[i]

    return statement_str

def byte_idx_to_char_idx(byte_idx: int, line: str) -> int:
    """convert byte index to char index"""
    return len(bytes(line, 'utf-8')[:byte_idx].decode('utf-8'))

def last_char_idx_to_token_idx(char_idx: int, tokens: List[str]) -> int:
    """ find the token that ends with the given char index """
    # calculate the token end indices
    token_end_indices = []
    total_len = 0
    for token in tokens:
        total_len += len(token)
        token_end_indices.append(total_len)

    if char_idx+1 in token_end_indices:
        return token_end_indices.index(char_idx+1)
    else:
        # here is a special case, when the last char of a stmt is not the ending 
        # char of a token. An example is `a = f(x);`, while ')' is the last char
        # of the stmt, BPE gives ');' as a token, thus we will have to add the whole token
        return token_end_indices.index(char_idx+2)

def last_byte_idx_to_token_idx(byte_idx: int, tokens: List[str], tokenizer) -> int:
    """ find the token that ends with the given byte index """
    # calculate the token end indices
    token_end_indices = []
    total_len = 0
    for token in tokens:
        total_len += len(bytearray([tokenizer.byte_decoder[c] for c in token]))
        token_end_indices.append(total_len)

    if byte_idx+1 in token_end_indices:
        return token_end_indices.index(byte_idx+1)
    else:
        # here is a special case, when the last byte of a stmt is not the ending 
        # char of a token. An example is `a = f(x);`, while ')' is the last char
        # of the stmt, BPE gives ');' as a token, thus we will have to add the whole token
        # NOTE: the example above is actually the only one we observe
        return token_end_indices.index(byte_idx+2)

def get_statements_from_code(code: str, parser, tolerate_errors: bool=False) -> List[Dict[str, Any]]:
    parsed_tree = parser.parse(bytes(code, 'utf-8'))

    # do a dfs on the parsed tree to record all the simple statements
    target_stmts: List[Dict] = []
    node_stack = [parsed_tree.root_node]
    while len(node_stack) > 0:
        node = node_stack.pop()

        if (node.type.endswith('statement') or node.type in ['comment', 'decorator']) \
            and node.type not in COM_STMTS:
            # this is a simple statement or a comment, so we can add it to the list
            target_stmts.append({'type': node.type, 'start_point': node.start_point, 
                                 'end_point': node.end_point, 'start_byte': node.start_byte, 
                                 'end_byte': node.end_byte})
        elif node.type in COM_STMTS or node.type.endswith('clause'):
            # separate the header and the body by the ":" token
            children_types = [c.type for c in node.children]
            separator_idx = children_types.index(':')
            assert separator_idx != -1

            # start of the header is the starter of the complex stmt, end is the end of the ":" token
            target_stmts.append({'type': node.type+'_header', 'start_point': node.start_point, 
                                 'start_byte': node.children[separator_idx].start_byte,
                                 'end_point': node.children[separator_idx].end_point, 
                                 'end_byte': node.children[separator_idx].end_byte})
            node_stack.extend(node.children[separator_idx+1:][::-1])
        elif node.type in PY_MODULES:
            node_stack.extend(node.children[::-1])
        elif node.type == 'ERROR':
            # err_code_line = code[:byte_idx_to_char_idx(node.end_byte, code)].split('\n')[-1]
            # print(f"failed to parse code: #########\n{err_code_line}\n#########")
            if tolerate_errors:
                continue
            else:
                # failed to parse tree, return None NOTE: previously returning [], but this will get 
                # confused with blank cells
                return None
        else:
            # other types, not sure what it contains, but assume it doesn't contain more statements
            print(f'unexpected node type: {node.type}')
            assert 'statement' not in node.sexp()

    return target_stmts


if __name__ == '__main__':
    language_build_path = os.path.join(os.path.dirname(__file__), '../../preprocessing/py-tree-sitter.so')
    PY_LANGUAGE = Language(language_build_path, 'python')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    code = "#a simple test\nif a == 0:\n  if b == 0:\n    b=1\nelif a>1:\n  a*=2\nelse:\n  a/=2"
    print(code)

    stmts = get_statements_from_code(code, parser)

    start = 0
    stmt_str = []
    for stmt in stmts:
        stmt_str.append(code[byte_idx_to_char_idx(start, code): byte_idx_to_char_idx(stmt['end_byte'], code)])
        start = stmt['end_byte']
        

    tree = parser.parse(bytes(code, 'utf-8'))

    root = tree.root_node

    print("")

