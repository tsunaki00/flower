docker build -t conda:init .
DIR=`pwd`
docker run -i -t -v ${DIR}/python:/home/python conda:init /bin/bash
