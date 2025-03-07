cd <sfile>:h

args python3/*.py
edit python3/tests.py

let g:vimspector_configurations = {
\   'Python': {
\     'adapter': 'debugpy',
\     'configuration': {
\       'name': 'Launch',
\       'type': 'python',
\       'request': 'launch',
\       'program': '${file}',
\     }
\   }
\ }
