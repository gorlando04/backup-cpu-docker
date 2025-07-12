#!/bin/bash

function clean_job() {
          echo "Limpando ambiente..."
            rm -rf "${local_job}"
    }
repeat(){

        sleep 1

    for name in   'gaussian-7-Overlap' 
            do

        for N in 5000 10000 15000 20000 30000 40000 
            do
		for dim in 10 20 50 100 200
		do
                	for id in 1 2 3 4 5 6 7 8 9 10
                    	do
                        	echo
                        	python3 2_experiment.py -name $name -N $N -D $dim -iter $id
                        	sleep 1
                    	done
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
