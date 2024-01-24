
from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  '/home/CodeT5/parser/my-languages.so',

  # Include one or more languages
  [
    '/home/CodeT5/parser/vendor/tree-sitter-go',
    '/home/CodeT5/parser/vendor/tree-sitter-javascript',
    '/home/CodeT5/parser/vendor/tree-sitter-python',
    '/home/CodeT5/parser/vendor/tree-sitter-java',
    '/home/CodeT5/parser/vendor/tree-sitter-ruby',
  ]
)
