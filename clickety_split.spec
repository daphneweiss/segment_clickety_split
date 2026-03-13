# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Segment Clickety Split
#
# Build:
#   pip install pyinstaller
#   pyinstaller clickety_split.spec
#
# Output: dist/ClicketySplit/ClicketySplit (macOS/Linux)
#         dist/ClicketySplit/ClicketySplit.exe (Windows)
#
# The resulting folder (dist/ClicketySplit/) can be zipped and shared.
# Recipients just double-click the executable — no Python required.
#
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

a = Analysis(
    ['launch.py'],
    pathex=[],
    binaries=[],
    datas=[
        # HTML frontend files
        ('index.html', '.'),
        ('review_tool.html', '.'),
        # Silero VAD ONNX model (used by silero_onnx.py — no torch needed)
        ('silero_vad.onnx', '.'),
        # soundfile shared library
        *collect_data_files('soundfile'),
    ],
    hiddenimports=[
        'soundfile',
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.io',
        'scipy.io.wavfile',
        'flask',
        'werkzeug',
        'werkzeug.serving',
        'noisereduce',
        'onnxruntime',
        'parselmouth',
        'tqdm',
        'silero_vad',
        'torch',
        'torch.nn',
        'torch.jit',
        # tkinter (folder picker dialog)
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torchvision',
        'matplotlib',      # only used for CLI plot_overview, not needed at runtime
        'IPython',
        'pytest',
        'triton',
        'nvidia',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ClicketySplit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,   # keep console visible so users can see errors / Ctrl+C
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ClicketySplit',
)
