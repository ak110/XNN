#!/bin/sh

PATH=../../XNN:$(PATH)

echo ====== 学習 ======
XNN XNN.conf

echo ====== 検証 ======
XNN XNN.conf task=pred

