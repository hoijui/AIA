#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources"
EXECUTABLE="${TARGET_DIR}/aia3"

cd "${SCRIPT_DIR}"

if [ ! -f "${EXECUTABLE}" ]; then
	./build.sh
fi

if [ ! -f "${RESOURCES_DIR}/img/poker.jpg" ]; then
	if [ ! -f aia3-material.zip ]; then
		echo "Please download the additional materials (aia3-material.zip) from the AIA course ISIS page"
		exit 1
	fi
	mkdir -p "${RESOURCES_DIR}"
	cd "${RESOURCES_DIR}"
	unzip "../../../aia3-material.zip"
fi

cd "${TARGET_DIR}"
"${EXECUTABLE}" \
	"${RESOURCES_DIR}/img/moneyTemplate100.jpg" \
	"${RESOURCES_DIR}/img/poker.jpg"

