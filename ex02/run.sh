#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR="${SCRIPT_DIR}/target"
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources"

cd "${SCRIPT_DIR}"

./build.sh

cd "${TARGET_DIR}"
./aia2 \
	"${RESOURCES_DIR}/img/orig.jpg" \
	"${RESOURCES_DIR}/img/blatt_art1.jpg" \
	"${RESOURCES_DIR}/img/blatt_art2.jpg"

