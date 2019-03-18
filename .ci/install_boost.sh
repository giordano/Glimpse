#!/bin/bash -x

brew uninstall --ignore-dependencies boost
cd "$HOME"
git clone https://github.com/spack/spack
cd spack
git checkout v0.12.1
export SPACK_ROOT=${PWD}
. $SPACK_ROOT/share/spack/setup-env.sh
spack install --jobs 2 boost %gcc
if [ $TRAVIS_OS_NAME = osx ]; then
    . $(brew --prefix modules)/init/bash;
fi
# Run again to be able to load the module
. $SPACK_ROOT/share/spack/setup-env.sh
spack load boost
