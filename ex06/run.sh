#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources"
IMG_DIR="${RESOURCES_DIR}/img"
EXECUTABLE="${TARGET_DIR}/aia6"

cd "${SCRIPT_DIR}"

if [ ! -f "${EXECUTABLE}" ]; then
	./build.sh
fi

cd "${TARGET_DIR}"

if [ ! -e img ]; then
	ln -s ../src/main/resources/img img
fi

if [ ! -e Conf_matrix_PC2_Cluster5.txt ]; then
	ln -s ../src/main/resources/Conf_matrix_PC2_Cluster5.txt .
fi

if [ ! -e Conf_matrix_PC25_Cluster10.txt ]; then
	ln -s ../src/main/resources/Conf_matrix_PC25_Cluster10.txt .
fi

ARGUMENTS=""

if [ "${1}" = "debug" ]; then
	echo "ARGUMENTS: ${ARGUMENTS}"
	ddd "${EXECUTABLE}" &
else
	"${EXECUTABLE}" ${ARGUMENTS}
fi

