#!bin/sh

tar zxvf ./RawVideos/tvsum.tgz  -C ./RawVideos/ 
unzip ./RawVideos/summe.zip -d ./RawVideos/summe
unzip ./RawVideos/ovp.zip -d ./RawVideos/
unzip ./RawVideos/youtube.zip -d ./RawVideos/


unzip ./RawVideos/ydata-tvsum50-v1_1/'*.zip' -d ./RawVideos/tvsum/
rm -rf ./RawVideos/ydata-tvsum50-v1_1/

rm ./RawVideos/WebscopeReadMe.txt

rm ./RawVideos/tvsum.tgz
rm ./RawVideos/summe.zip
rm ./RawVideos/ovp.zip
rm ./RawVideos/youtube.zip

