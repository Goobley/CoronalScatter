#!/bin/bash

if [[ $(uname) = "Darwin" ]]; then
    ISPC_SUFFIX="macOS"
else
    ISPC_SUFFIX="linux"
fi
wget "https://github.com/ispc/ispc/releases/download/v1.18.0/ispc-v1.18.0-${ISPC_SUFFIX}.tar.gz"
tar xvzf "ispc-v1.18.0-${ISPC_SUFFIX}.tar.gz"