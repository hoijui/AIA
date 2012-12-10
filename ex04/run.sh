#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources"
EXECUTABLE="${TARGET_DIR}/aia4"

cd "${SCRIPT_DIR}"

if [ ! -f "${EXECUTABLE}" ]; then
	./build.sh
fi

cd "${TARGET_DIR}"

ARGUMENTS="" # TODO

if [ "${2}" = "debug" ]; then
	echo "ARGUMENTS: ${ARGUMENTS}"
	ddd "${EXECUTABLE}" &
else
	"${EXECUTABLE}" ${ARGUMENTS}
fi

