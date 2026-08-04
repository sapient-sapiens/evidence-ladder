# Third-party notices

This project is MIT licensed. It bundles or invokes the components below.

## Bundled in the Docker image

| Component | Use | License |
| --- | --- | --- |
| [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) `eng.traineddata` | Tesseract LSTM model, OEM 1 | Apache-2.0 |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) + bundled PP-OCR ONNX models | neural OCR fallback | Apache-2.0 |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | ONNX inference | MIT |
| [scikit-learn](https://scikit-learn.org/), [numpy](https://numpy.org/), [joblib](https://joblib.readthedocs.io/) | model artifacts and inference | BSD-3-Clause |
| [pypdf](https://github.com/py-pdf/pypdf) | PDF text layer | BSD-3-Clause |
| [Pillow](https://python-pillow.org/) | raster handling | MIT-CMU |

The vendored `tessdata_best/eng.traineddata` is unmodified upstream; its
provenance, upstream URL and sha256 are recorded in
`tessdata_best/PROVENANCE.json`, and the Apache-2.0 text ships beside it in
`tessdata_best/LICENSE`.

## Invoked as system binaries

`tesseract` (Apache-2.0) and `poppler-utils` / `pdftoppm` (GPL-2.0-or-later) are
installed by the Dockerfile from Debian and executed as separate processes via
`subprocess`. They are not linked into this work and do not affect its license.

## Data

No challenge data is redistributed here. `dev/make_truth.py` regenerates the
DEV800 truth slice from the challenge repository's own public
`data/train_labels.csv` (MIT, 8090-inc/mib-doc-challenge).

## Prior art

The mechanisms in this repository were derived independently from the field
manual and the evaluator. Where the experiment ledger records an idea sourced
from a public submission, the source is named in `dev/EXPERIMENT_LEDGER.md`. No
third-party solution code is incorporated; every externally sourced idea listed
there was measured and rejected.
