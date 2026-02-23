import ast, sys
with open('dashboard/app.py', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print('dashboard/app.py: OK - no syntax errors')
except SyntaxError as e:
    print(f'SYNTAX ERROR at line {e.lineno}: {e.msg}')
    sys.exit(1)
