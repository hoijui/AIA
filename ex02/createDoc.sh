#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
DOC_DIR="${SCRIPT_DIR}/src/main/doc"

cd "${SCRIPT_DIR}"

./run.sh

cd "${DOC_DIR}"
pdflatex \
		-interaction=nonstopmode \
		-output-format=pdf \
		-output-directory="${TARGET_DIR}" \
		doc.tex > /dev/null

cd "${TARGET_DIR}"
mv doc.pdf AIA_WS1213_GruppeE_Ex02.pdf

