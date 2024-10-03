# sudo docker build -f Dockerfile -t pytorch-lifestream-tests .
sudo docker run --name ptls_tests -it -v ${PWD}/ptls:/ptls -v ${PWD}/ptls_tests:/ptls_tests pytorch-lifestream-tests