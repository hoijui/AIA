#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources"
IMG_DIR="${RESOURCES_DIR}/img"
EXECUTABLE="${TARGET_DIR}/aia4"

cd "${SCRIPT_DIR}"

if [ ! -f "${EXECUTABLE}" ]; then
	./build.sh
fi

cd "${TARGET_DIR}"

# TODO
#ARGUMENTS="${IMG_DIR}/Input.png ${IMG_DIR}/Input.png ${IMG_DIR}/rotes_Auto_weiß.png ${IMG_DIR}/rotes_Auto_schwarz.png"
#ARGUMENTS="${IMG_DIR}/Input.png ${IMG_DIR}/Input.png ${IMG_DIR}/weißes_Auto_weiß.png ${IMG_DIR}/weißes_Auto_schwarz.png"
#ARGUMENTS="${IMG_DIR}/Input.png ${IMG_DIR}/Input.png ${IMG_DIR}/schwarzes_Auto_weiß.png ${IMG_DIR}/schwarzes_Auto_schwarz.png"
ARGUMENTS="${IMG_DIR}/Input.png ${IMG_DIR}/Input.png ${IMG_DIR}/Background_weiß.png ${IMG_DIR}/Background_schwarz.png"


if [ "${2}" = "debug" ]; then
	echo "ARGUMENTS: ${ARGUMENTS}"
	ddd "${EXECUTABLE}" &
else
	"${EXECUTABLE}" ${ARGUMENTS}
fi

