plda:src/plda.cc
	g++ -O3 -o bin/plda src/plda.cc -Ieigen3/ -L./ -lboost_program_options

test:plda
	./bin/plda data/train.txt data/labels data/test.txt
