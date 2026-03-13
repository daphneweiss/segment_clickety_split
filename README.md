
# Clickety Split: Audio Segmentation for Speech Experiments

Interactive recording segmentation tool for speech perception experiments. Detects word boundaries in recordings using VAD + acoustic analysis, lets you review and adjust in-browser, then exports individual tokens as WAV files.



### Disclaimer
Like many academic scripts, this project was built by one person with one set of experiments in mind, and has not been extensively tested outside that context. Always run any processing tool on a COPY of your original stimuli. 

I recommend running this natively in Python rather than using the executable.

---


## Quickstart — Executable in Browser (Windows)
First, make sure that your directory is structured as described below (recordings -> speakers -> conditions -> .wav)

You must also pass a stimulus list, which constrains the possible filenames. You may use one .txt or .csv for the whole experiment, or one per condition. Acceptable formats are one single word column, or one column with word and one column with condition.

1. Unzip `ClicketySplit.zip` anywhere (e.g. your Desktop)
2. Double-click `ClicketySplit`` ClicketySplit.exe
3. A folder-picker dialog appears — select your recordings directory
4. The browser opens automatically at http://127.0.0.1:5000 (switch to GOOGLE CHROME if it opens in Firefox/other browser)
5. Choose the speaker and map each condition to a desired conditions list, if it is not automatically mapped with "auto detect" 
6. Switch to the segmenting tab and follow the prompts to accept the token as segmented, or to adjust boundaries/rename.

## Quickstart — Browser Only (no install required)

If you already have audio files and a segments JSON (e.g. produced by a colleague or a previous detection run), you can review and export tokens entirely in-browser with no Python or installation required.

1. Open `review_tool.html` directly in Chrome or Edge (double-click it, or drag it into the browser)
2. **Load Audio** — click the Audio button and pick your WAV file
3. **Load Segments** — click the Segments button and pick your `proposed_segments.json` or `reviewed_segments.json`
4. Enter your **Speaker ID** (e.g. `m1`) — this is prepended to exported filenames
5. **Review Tokens** tab — step through each detected token, adjust boundaries, accept/reject, fix labels
6. Click **Export Reviewed JSON** to save `reviewed_segments.json` (you can save it directly to your experiment folder)
7. Switch to **Select Best** tab — pick the best token for each word, then click **⬇ Export WAVs (ZIP)**
8. Unzip — your tokens are ready: `{speaker}_{word}-{N}.wav`

> **Browser compatibility**: Chrome and Edge are recommended. Firefox works for review and JSON export, but the "Save directly to folder" dialog (`showSaveFilePicker`) requires Chrome/Edge. ZIP export works everywhere.

---


### Build

You will need to do this once, with a python install, if you are running Mac or Linux rather than using the included executable

```bash
pip install pyinstaller
pip install -r requirements.txt
pyinstaller clickety_split.spec
```

This produces a `dist/ClicketySplit/` folder. You can now run ClicketySplit, or distribute the zipped folder

### Use (no Python required)

1. Unzip `ClicketySplit.zip` anywhere (e.g. your Desktop)
2. Double-click `ClicketySplit` (macOS/Linux) or `ClicketySplit.exe` (Windows)
3. A folder-picker dialog appears — select your project directory (the one with `recordings/`)
4. The browser opens automatically at http://127.0.0.1:5000 (switch to GOOGLE CHROME if it opens in Firefox/other browser)
5. Follow the four-step workflow as normal (full ML detection included)
6. Press Ctrl+C in the terminal window to quit

> **Note**: The executable bundles the full ML pipeline (Silero VAD, noise reduction, etc.). First-run startup may be a few seconds slower than the Python version.

---

## Python alternative - preferred if you are familiar with Python

```bash
pip install -r requirements.txt
python app.py /path/to/your/project
```

Then open http://localhost:5000 and follow the four-step workflow in the browser.

### Project directory structure

Clickety Split expects recordings in one parent directory, with subdirectories per speaker and then per condition. If more than one audio file is in a condition folder, they will be concatenated automatically.

```
your_project/
├── recordings/          < speaker recordings
│   ├── f1/
│   │   ├── critical_s_normal/
│   │   │   └── recording.wav
│   │   └── filler_word/
│   │       └── recording.wav
│   └── m1/
│       └── ...
├── experiment/          < created automatically
│   ├── stimulus_lists/
│   └── {speaker}/{condition}/
│       ├── proposed_segments.json
│       ├── reviewed_segments.json
│       └── tokens/
└── sessions/            < auto-created for session persistence
```

Stimulus lists are plain text files, one word per line. The segmenter uses them to validate labels and power autocomplete during review. You can use the same list for all conditions, or one per condition.

Output filenames: `{speaker}_{word}-{N}.wav`

---

## Workflow

### Step 1: Setup Experiment (Flask only)

- **Pick a speaker** from the dropdown (auto-populated from `recordings/`)
- **Map conditions** to recording folders and stimulus lists
  - Click **Auto-detect** to populate from available condition folders
  - Assign a stimulus list to each condition (or one master list for all)
- **Adjust detection parameters**:
  - *Min word duration (ms)* — segments shorter than this are rejected as noise (default 500ms)
  - *Max word duration (ms)* — longer segments flagged as crosstalk (default 1400ms)
  - *Min silence gap (ms)* — minimum gap between words (default 150ms)
  - *Noise reduction* — spectral-gating noise reduction on/off
- Click **Run Detection on All Conditions** — this runs ML-based VAD; it may take a moment

### Step 2: Review & Adjust Segments

Review detected tokens one by one:

- **Switch conditions** with the buttons at the top
- **Waveform + Spectrogram** with adjustable boundaries
- **Boundary adjustment**: click near a boundary to snap it, or drag the L/R handles
- **Zoom**: scroll wheel (anchored at cursor)
- **Pan**: Shift+drag
- **Label editing**: type to rename; autocomplete from stimulus list
- **Actions**:
  - **Accept (Enter)** — approve with current boundaries and label
  - **Reject (R)** — mark as rejected (excluded from export)
  - **Skip (S)** — move on without marking
  - **Back (←)** — go to previous
  - **Add Token (A)** — manually add a missed token by clicking start then end

Reviewed segments are **automatically saved** to `experiment/{speaker}/{condition}/reviewed_segments.json` after each action (Flask mode) or on demand via Export Reviewed JSON (standalone mode).

### Step 3: Select Tokens

Pick which tokens to export for each word:

- Tokens are grouped by word with filename and duration shown
- Click to toggle selection (checkboxes; multiple tokens per word supported)
- **Tokens per word** picker: 1, 2, 3, or Any
- **Select All / Deselect All** for bulk operations
- **▶** on each row to play and compare

### Step 4: Export

- **Flask mode**: click **Export All Conditions** — writes WAV files to `experiment/{speaker}/{condition}/tokens/`
- **Standalone mode** (`review_tool.html`): click **⬇ Export WAVs (ZIP)** — downloads a ZIP of all selected tokens

Each token file: `{speaker}_{word}-{N}.wav`
A `token_manifest.json` is saved alongside the exports (Flask mode).

---

## Save / Load Progress (Flask mode)

Click **💾 Save** or **📂 Load** in the top bar:

- Sessions saved as `.pkl` files in `sessions/` at the project root
- Each session captures speaker, conditions, parameters, and full review progress
- Auto-save runs every 2 seconds during review

---

## Keyboard Shortcuts (Review Mode)

| Key | Action |
|-----|--------|
| Tab / Space | Play current segment |
| Enter | Accept token |
| R | Reject token |
| S | Skip to next |
| ← | Go back |
| A | Toggle add-token mode |
| Esc | Cancel add-token / unfocus input |
| Scroll wheel | Zoom in/out |
| Shift+drag | Pan view |

---

## CLI Options (Flask mode)

```
python app.py PROJECT_DIR [--port 5000] [--host 127.0.0.1] [--debug]
```

| Argument | Description |
|----------|-------------|
| `PROJECT_DIR` | Project directory with `recordings/` and `experiment/` |
| `--port` | Server port (default: 5000) |
| `--host` | Server host (default: 127.0.0.1) |
| `--debug` | Enable Flask debug mode |
