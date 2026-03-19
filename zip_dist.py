import zipfile
from pathlib import Path
src = Path('dist/ClicketySplit')
out = Path('ClicketySplit_v102.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in src.rglob('*'):
        if f.is_file():
            z.write(f, f.relative_to(src.parent))
print('done', out.stat().st_size // 1024 // 1024, 'MB')
