#!/bin/sh

for i in */Run.sh ; do
    cd "`dirname $i`"
    echo ========================= $i =========================
    echo "" | ./Run.sh
    echo ""
    cd ".."
done
