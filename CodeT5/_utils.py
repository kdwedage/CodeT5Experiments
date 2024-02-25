import json
import sys
from tree_sitter import Language, Parser
sys.path.append('/home/CodeT5Experiments/CodeT5/parser/')
from DFG import DFG_python
from parser_utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index,
                   tree_to_token_nodes)

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize' or 'finetune' in task:
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix: 
        task_prefix = args.task
        if task_prefix == 'pretrain0' or task_prefix == 'pretrain2' or task_prefix == 'pretrain3':
            task_prefix = '<DENOISE>'
        elif task_prefix == 'pretrain1':
            task_prefix = 'summarize <AST>'
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(task_prefix, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(task_prefix, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task='',
                 ast='',
                 dfg='',
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples

def preorder_traversal(node, include_leaf_value=False):
    result = ""
    type = str(node.type)
    result += type + ' '
    if not node.children and node.text.decode('utf-8') != type and include_leaf_value:
        result += '<' + node.text.decode('utf-8') + '> '

    # Recursively traverse the children in preorder
    for child in node.children:
        result += preorder_traversal(child)
    
    return result



def read_pretrain0_examples(filename, data_num, preorder=True):
    """ Input is corrupted Code + DFG + AST. Output is uncorrupted. Corrupted occurs after tokenization."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node)
            else:
                ast = root_node.sexp()

            # DFG.
            ast_token_nodes = tree_to_token_nodes(root_node)
            tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
            original_code=original_code.split('\n')
            code_tokens=[index_to_code_token(x,original_code) for x in tokens_index] 
            index_to_code={index:(idx,code_) for idx,(index,code_) in enumerate(zip(tokens_index,code_tokens))}
    
            try:
                dfg,_ = DFG_python(root_node,index_to_code,{}) 
            except Exception as e:
                dfg = []
                print(str(e))
            for d in dfg:
                assert (d[2]=='comesFrom' or d[2]=='computedFrom')
            dfg = [(d[1], d[4]) for d in dfg if (len(d[4])>0)] # left comes from right

            # Only save variables with unique names.
            concise_dfg = set()
            for x,y in dfg:
                if len(y) == 1 and code_tokens[x] == code_tokens[y[0]]:
                    continue
                valid = (code_tokens[x], tuple([code_tokens[z] for z in y]))
                concise_dfg.add(valid)
            dfg = str(concise_dfg).replace("'",'').replace('{','').replace('}','')

            unintended_code += '<DFG>' + dfg + '<AST>' + ast
            examples.append(
                Example(
                    idx=idx,
                    source=unintended_code,
                    target=unintended_code,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_pretrain1_examples(filename, data_num, preorder=True):
    """ Input is full AST. Output is original code."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node, include_leaf_value=True)
            else:
                ast = root_node.sexp()

            examples.append(
                Example(
                    idx=idx,
                    source=ast,
                    target=unintended_code,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_pretrain2_examples(filename, data_num):
    """ Input is corrupted code + DFG. Output is uncorrupted."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node

            # DFG.
            ast_token_nodes = tree_to_token_nodes(root_node)
            tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
            original_code=original_code.split('\n')
            code_tokens=[index_to_code_token(x,original_code) for x in tokens_index] 
            index_to_code={index:(idx,code_) for idx,(index,code_) in enumerate(zip(tokens_index,code_tokens))}
    
            try:
                dfg,_ = DFG_python(root_node,index_to_code,{}) 
            except Exception as e:
                dfg = []
                print(str(e))
            for d in dfg:
                assert (d[2]=='comesFrom' or d[2]=='computedFrom')
            dfg = [(d[1], d[4]) for d in dfg if (len(d[4])>0)] # left comes from right

            # Only save variables with unique names.
            concise_dfg = set()
            for x,y in dfg:
                if len(y) == 1 and code_tokens[x] == code_tokens[y[0]]:
                    continue
                valid = (code_tokens[x], tuple([code_tokens[z] for z in y]))
                concise_dfg.add(valid)
            dfg = str(concise_dfg).replace("'",'').replace('{','').replace('}','')

            unintended_code += '<DFG>' + dfg
            examples.append(
                Example(
                    idx=idx,
                    source=unintended_code,
                    target=unintended_code,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_pretrain3_examples(filename, data_num, preorder=True):
    """ Input is corrupted full AST. Output is uncorrupted."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node, include_leaf_value=True)
            else:
                ast = root_node.sexp()

            ast = '<AST> ' + ast

            examples.append(
                Example(
                    idx=idx,
                    source=ast,
                    target=ast,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_finetune0_examples(filename, data_num, preorder=True):
    """ Input is Code + DFG + AST. Output is NL summary."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node)
            else:
                ast = root_node.sexp()

            # DFG.
            ast_token_nodes = tree_to_token_nodes(root_node)
            tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
            original_code=original_code.split('\n')
            code_tokens=[index_to_code_token(x,original_code) for x in tokens_index] 
            index_to_code={index:(idx,code_) for idx,(index,code_) in enumerate(zip(tokens_index,code_tokens))}
    
            try:
                dfg,_ = DFG_python(root_node,index_to_code,{}) 
            except Exception as e:
                dfg = []
                print(str(e))
            for d in dfg:
                assert (d[2]=='comesFrom' or d[2]=='computedFrom')
            dfg = [(d[1], d[4]) for d in dfg if (len(d[4])>0)] # left comes from right

            # Only save variables with unique names.
            concise_dfg = set()
            for x,y in dfg:
                if len(y) == 1 and code_tokens[x] == code_tokens[y[0]]:
                    continue
                valid = (code_tokens[x], tuple([code_tokens[z] for z in y]))
                concise_dfg.add(valid)
            dfg = str(concise_dfg).replace("'",'').replace('{','').replace('}','')

            unintended_code += '<DFG>' + dfg + '<AST>' + ast
            examples.append(
                Example(
                    idx=idx,
                    source=unintended_code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_finetune1_examples(filename, data_num, preorder=True):
    """ Input is full AST. Output is NL summary."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node, include_leaf_value=True)
            else:
                ast = root_node.sexp()

            examples.append(
                Example(
                    idx=idx,
                    source=ast,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_finetune2_examples(filename, data_num, preorder=True):
    """ Input is Code + DFG. Output is NL summary."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node, include_leaf_value=True)
            else:
                ast = root_node.sexp()

            # DFG.
            ast_token_nodes = tree_to_token_nodes(root_node)
            tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
            original_code=original_code.split('\n')
            code_tokens=[index_to_code_token(x,original_code) for x in tokens_index] 
            index_to_code={index:(idx,code_) for idx,(index,code_) in enumerate(zip(tokens_index,code_tokens))}
    
            try:
                dfg,_ = DFG_python(root_node,index_to_code,{}) 
            except Exception as e:
                dfg = []
                print(str(e))
            for d in dfg:
                assert (d[2]=='comesFrom' or d[2]=='computedFrom')
            dfg = [(d[1], d[4]) for d in dfg if (len(d[4])>0)] # left comes from right

            # Only save variables with unique names.
            concise_dfg = set()
            for x,y in dfg:
                if len(y) == 1 and code_tokens[x] == code_tokens[y[0]]:
                    continue
                valid = (code_tokens[x], tuple([code_tokens[z] for z in y]))
                concise_dfg.add(valid)
            dfg = str(concise_dfg).replace("'",'').replace('{','').replace('}','')

            unintended_code += '<DFG>' + dfg

            examples.append(
                Example(
                    idx=idx,
                    source=unintended_code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_finetune3_examples(filename, data_num, preorder=True):
    """ Input consists of three different inputs. Output is NL summary."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            unintended_code = ' '.join(js['code_tokens']).replace('\n', ' ')
            unintended_code = ' '.join(unintended_code.strip().split())

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            dfg_function={
            'python':DFG_python
            }

            parsers={}        
            for lang in dfg_function:
                LANGUAGE = Language('/home/CodeT5Experiments/CodeT5/parser/my-languages2.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE) 
                parser = [parser,dfg_function[lang]]    
                parsers[lang]= parser
            
            # Add AST.
            original_code = js['code']
            tree = parsers['python'][0].parse(bytes(original_code,'utf8')) 
            root_node = tree.root_node
            if preorder:
                ast = preorder_traversal(root_node, include_leaf_value=True)
            else:
                ast = root_node.sexp()

            # DFG.
            ast_token_nodes = tree_to_token_nodes(root_node)
            tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
            original_code=original_code.split('\n')
            code_tokens=[index_to_code_token(x,original_code) for x in tokens_index] 
            index_to_code={index:(idx,code_) for idx,(index,code_) in enumerate(zip(tokens_index,code_tokens))}
    
            try:
                dfg,_ = DFG_python(root_node,index_to_code,{}) 
            except Exception as e:
                dfg = []
                print(str(e))
            for d in dfg:
                assert (d[2]=='comesFrom' or d[2]=='computedFrom')
            dfg = [(d[1], d[4]) for d in dfg if (len(d[4])>0)] # left comes from right

            # Only save variables with unique names.
            concise_dfg = set()
            for x,y in dfg:
                if len(y) == 1 and code_tokens[x] == code_tokens[y[0]]:
                    continue
                valid = (code_tokens[x], tuple([code_tokens[z] for z in y]))
                concise_dfg.add(valid)
            dfg = str(concise_dfg).replace("'",'').replace('{','').replace('}','')

            examples.append(
                Example(
                    idx=idx,
                    source=unintended_code,
                    target=nl,
                    ast='<AST> ' + ast,
                    dfg='<DFG> ' + dfg,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
