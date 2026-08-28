import base64, sys
b64 = sys.stdin.read().strip()
content = base64.b64decode(b64).decode('utf-8')
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
print(f'README.md written: {len(content)} bytes')