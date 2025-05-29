sudo systemctl daemon-reload
sudo systemctl restart docker

docker run --net=host  --name CPU_HDBSCAN --rm -it -v /home/gmcorlando:/nndescent -w /nndescent --shm-size=1g --ulimit memlock=-1 ubuntu

