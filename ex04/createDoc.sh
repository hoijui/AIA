#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
DOC_DIR="${SCRIPT_DIR}/src/main/doc"

cd "${SCRIPT_DIR}"

#if [ ! -e "${TARGET_DIR}/result.png" ]; then # TODO
	#./run.sh
#fi

cd "${DOC_DIR}"
pdflatex \
		-interaction=nonstopmode \
		-output-format=pdf \
		-output-directory="${TARGET_DIR}" \
		doc.tex

cd "${TARGET_DIR}"
mv doc.pdf AIA_WS1213_GruppeE_Ex04.pdf

