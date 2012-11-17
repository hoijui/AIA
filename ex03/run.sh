#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources"
EXECUTABLE="${TARGET_DIR}/aia3"

cd "${SCRIPT_DIR}"

if [ ! -f "${EXECUTABLE}" ]; then
	./build.sh
fi

cd "${TARGET_DIR}"
"${EXECUTABLE}" \
	"${RESOURCES_DIR}/img/template.jpg" \
	"${RESOURCES_DIR}/img/test.jpg"

