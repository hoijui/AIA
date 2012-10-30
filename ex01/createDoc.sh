#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
DOC_DIR="${SCRIPT_DIR}/src/main/doc"

mkdir -p "${TARGET_DIR}"

cd "${TARGET_DIR}"
if [ ! -f input.jpg ]; then
	#wget http://farm4.static.flickr.com/3337/3442591520_ef627a310f.jpg
	wget http://3.bp.blogspot.com/_T5WCAhoADs4/S7EJ_FzTObI/AAAAAAAACUI/5-yWOaXoRGc/s1600/peeps.jpg
	mv peeps.jpg input.jpg
fi

if [ ! -f result.jpg ]; then
	if [ ! -f aia1 ]; then
		if [ ! -f CMakeCache.txt ]; then
			cmake ..
		fi
		make
	fi
	./aia1 input.jpg
fi

cd "${DOC_DIR}"
# When using this, relative paths get messed up (one needs to use an extra "../")
#texi2pdf \
#		--build=tidy \
#		--batch \
#		--build-dir="${TARGET_DIR}" \
#		--output="${TARGET_DIR}/AIA_WS1213_GruppeE_Ex01.pdf" \
#		doc.tex

pdflatex \
		-output-format=pdf \
		-output-directory="${TARGET_DIR}" \
		doc.tex

cd "${TARGET_DIR}"
mv doc.pdf AIA_WS1213_GruppeE_Ex01.pdf

