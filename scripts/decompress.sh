#!bin/sh

tar zxvf ./RawVideos/tvsum.tgz  -C ./RawVideos/ 
unzip ./RawVideos/summe.zip -d ./RawVideos/summe

unzip ./RawVideos/ydata-tvsum50-v1_1/'*.zip' -d ./RawVideos/tvsum/
rm -rf ./RawVideos/ydata-tvsum50-v1_1/

rm ./RawVideos/WebscopeReadMe.txt
