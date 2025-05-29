#!/bin/bash

function clean_job() {
	  echo "Limpando ambiente..."
	    rm -rf "${local_job}"
    }
repeat(){

	sleep 1

    for name in   'gaussian-7-Overlap' 'gaussian-7-noOverlap' 'gaussian-5-noOverlap' 'beta'
	    do

        for N in 25000 
            do
                for id in 1 2 3 4 5 6 7 8 9 10
                    do
                        echo
                        python3 200d_experiment.py -name $name -N $N -D 200 -iter $id -kmax 50 
                        sleep 1
                    done
            done
	    done
	exit
}

    trap clean_job EXIT HUP INT TERM ERR

    set -eE

    umask 077

    repeat

    echo exit
