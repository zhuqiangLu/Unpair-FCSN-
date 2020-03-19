#!bin/sh


RawVideo="./RawVideos"

if [ ! -d $RawVideo ]
then
    mkdir $RawVideo
fi

cd $RawVideo

tvsum="tvsum.tgz"
summe="summe.zip"
youtube="youtube.tar.gz"

datasets=(
    "tvsum.tgz"
    "summe.zip"
)
urls=(
    "http://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz"
    "https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip"
)


len=${#datasets[@]}
for (( i=0; i<$len; i++ ))
    do
        dataset=${datasets[$i]}
        url=${urls[$i]}
        if [ ! -f $dataset ]
        then
            echo "downloading $dataset" 
            curl -o $dataset $url
        fi

done
#pids=( $(pgrep 'curl') )
#ps -ef | grep curl | grep -v grep | awk '{print $2}'
