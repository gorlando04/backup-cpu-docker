#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1

    for name in   'beans' '20news_300'
	    do

                for id in 1 2 3 4 5 6 7 8 9 10
                    do
                        echo
                        python3 experiment.py -name $name -N 100 -D 50 -iter $id -kmax 50 
                        sleep 1
                    done
	    done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
