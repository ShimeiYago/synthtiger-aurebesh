# SynthTIGER for Aurebesh

Fork of [SynthTIGER](https://github.com/clovaai/synthtiger) specialized for generating synthetic Aurebesh (Star Wars alphabet) text images. This project enables the creation of Aurebesh text data for training text recognition models.

## Contents

- [Installation](#installation)
- [Preparation](#preparation)
- [Usage](#usage)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [License](#license)

## Installation

SynthTIGER requires `python>=3.6` and `libraqm`.

Recommended: use Python **3.11** for this fork (validated with the provided `constraints.txt`).

To install SynthTIGER from PyPI:

```bash
pip install -c constraints.txt synthtiger wordfreq
```

If you see a dependency error when you install or run SynthTIGER, install [dependencies](depends).

## Preparation

### Place Aurebesh font files

Copy your `.otf / .ttf` fonts into the style buckets:

```
aurebesh/fonts/core     # canonical fonts (used 70% of the time)
aurebesh/fonts/variant  # bold / italic / condensed (used 30% of the time)
```

**Recommended fonts**

| Category | Font (Style)                     | License                                     | Download                                                              |
| -------- | -------------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| Core     | **Aurebesh AF – Canon**          | Public Domain                               | [FontSpace](https://www.fontspace.com/aurebesh-af-font-f49637)        |
|          | **FT Aurebesh – Regular**        | SIL OFL 1.1                                 | [DeeFont](https://www.deefont.com/ft-aurebesh-font-family/)           |
|          | **FT Aurebesh – UltraLight**     | SIL OFL 1.1                                 | [DeeFont](https://www.deefont.com/ft-aurebesh-font-family/)           |
|          | **Aurek-Besh – Regular**         | Freeware                                    | [FontSpace](https://www.fontspace.com/aurek-besh-font-f9639)          |
| Variant  | **FT Aurebesh – Black**          | SIL OFL 1.1                                 | [DeeFont](https://www.deefont.com/ft-aurebesh-font-family/)           |
|          | **Aurebesh Font – Italic**       | Freeware, commercial use requires donation  | [FontSpace](https://www.fontspace.com/aurebesh-font-f17959)           |
|          | **Aurek-Besh – Narrow**          | Freeware                                    | [FontSpace](https://www.fontspace.com/aurek-besh-font-f9639)          |

Note: I do not own the copyright to these fonts.

Then generate charset file for each font:

```bash
python tools/extract_font_charset.py aurebesh/fonts/core
python tools/extract_font_charset.py aurebesh/fonts/variant
```

### Generate Aurebesh Corpus

Create a mixed English + Star Wars term corpus (uppercase, 1–4 words per line) that SynthTIGER will sample from:

```bash
python tools/generate_aurebesh_corpus.py --size 150000
```

Adjust distribution or Star Wars term rate with: `--len-dist "1:0.5,2:0.3,3:0.1,4:0.1"`, `--p-sw 0.05`, `--inject-punct 0.05`. The default Star Wars vocab file is at `aurebesh/vocab/starwars-vocab.txt`.

## Usage

```bash
# Set environment variable (for macOS)
$ export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

```
usage: synthtiger [-h] [-o DIR] [-c NUM] [-w NUM] [-s NUM] [-v] SCRIPT NAME [CONFIG]

positional arguments:
  SCRIPT                Script file path.
  NAME                  Template class name.
  CONFIG                Config file path.

optional arguments:
  -h, --help            show this help message and exit
  -o DIR, --output DIR  Directory path to save data.
  -c NUM, --count NUM   Number of output data. [default: 100]
  -w NUM, --worker NUM  Number of workers. If 0, It generates data in the main process. [default: 0]
  -s NUM, --seed NUM    Random seed. [default: None]
  -v, --verbose         Print error messages while generating data.
```

### Generate SynthTIGER text images

```bash
# horizontal
synthtiger -o results/synthtiger_output -w 4 -c 300000 examples/synthtiger/template.py SynthTiger aurebesh/config_horizontal.yaml
```

- `images`: a directory containing images.
- `gt.txt`: a file containing text labels.
- `coords.txt`: a file containing bounding boxes of characters with text effect.
- `glyph_coords.txt`: a file containing bounding boxes of characters without text effect.
- `masks`: a directory containing mask images with text effect.
- `glyph_masks`: a directory containing mask images without text effect.

### Export to PaddleOCR recognizer dataset

```bash
python tools/export_to_paddleocr.py -i results/synthtiger_output -o results/paddleocr_data
```

## Advanced Usage

### Non-Latin language data generation

<img src="https://user-images.githubusercontent.com/12423224/167302532-dbd5fa60-bcba-4f77-92ee-58bb6efda51c.png" width="40%"/>

1. Prepare corpus

   `txt` format, line by line ([example](resources/corpus/mjsynth.txt)).

2. Prepare fonts

   See [font customization](#font-customization) for more details.

3. Edit corpus path and font path in config file ([example](examples/synthtiger/config_horizontal.yaml))

4. Run synthtiger

### Font customization

1. Prepare fonts

   `ttf`/`otf` format ([example](resources/font)).

2. Extract renderable charsets

   ```bash
   python tools/extract_font_charset.py -w 4 fonts/
   ```

   This script extracts renderable charsets for all font files ([example](resources/font/Ubuntu-Regular.txt)).

   Text files are generated in the input path with the same names as the fonts.

3. Edit font path in config file ([example](examples/synthtiger/config_horizontal.yaml))

4. Run synthtiger

### Colormap customization

1. Prepare images

   `jpg`/`jpeg`/`png`/`bmp` format.

2. Create colormaps

   ```bash
   python tools/create_colormap.py --max_k 3 -w 4 images/ colormap.txt
   ```

   This script creates colormaps for all image files ([example](resources/colormap/iiit5k_gray.txt)).

3. Edit colormap path in config file ([example](examples/synthtiger/config_horizontal.yaml))

4. Run synthtiger

### Template customization

You can implement custom templates by inheriting the base template.

```python
from synthtiger import templates


class MyTemplate(templates.Template):
    def __init__(self, config=None):
        # initialize template.

    def generate(self):
        # generate data.

    def init_save(self, root):
        # initialize something before save.

    def save(self, root, data, idx):
        # save data to specific path.

    def end_save(self, root):
        # finalize something after save.
```

## Citation

```bibtex
@inproceedings{yim2021synthtiger,
  title={SynthTIGER: Synthetic Text Image GEneratoR Towards Better Text Recognition Models},
  author={Yim, Moonbin and Kim, Yoonsik and Cho, Han-Cheol and Park, Sungrae},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={109--124},
  year={2021},
  organization={Springer}
}
```

## License

```
SynthTIGER
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

The following directories and their subdirectories are licensed the same as their origins. Please refer to [NOTICE](NOTICE)

```
docs/
resources/font/
```
