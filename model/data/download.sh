#!/bin/bash

if [[ $1 != "apple2orange" && $1 != "summer2winter_yosemite" &&  $1 != "horse2zebra" && $1 != "monet2photo" && $1 != "cezanne2photo" && $1 != "ukiyoe2photo" && $1 != "vangogh2photo" && $1 != "maps" && $1 != "facades" && $1 != "iphone2dslr_flower" ]]
then
    echo "Incorrect dataset name, please try again"
    exit 1
fi

url=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$1.zip

wget $url
unzip $1.zip
rm $1.zip
