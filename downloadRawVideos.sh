#!bin/sh

RawVideo="./rawVideo"
if [ ! -d $RawVideo ]
then
    mkdir $RawVideo
fi

cd $RawVideo

tvsum="tvsum.tgz"
summe="summe.zip"
youtube='youtube.tar.gz'
ovp
if [ ! -f $tvsum ]
then    
    echo "downloading ${tvsum}"
    curl -o $tvsum http://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz
fi

if [ ! -f $summe ]
then    
    echo "downloading ${summe}"
    curl -o $summe https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
fi

if [ ! -f $youtube ]
then    
    echo "downloading ${youtube}"
    curl -o $youtube http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz
fi




# echo "downloading SumMe dataset"
# curl -o tvsum50.tgz http://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz
# curl -o summe.zip  https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
