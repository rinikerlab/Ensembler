#clean up:
make clean;

#make doku
##configurations
sphinx-apidoc -o _source ../../ensembler

cp ../../examples/Tutorial*ipynb ./Tutorials
cp ../../examples/Example*ipynb ./Examples

python conf.py

##execute making docu
make html
make latex

cp -r _build/hmtl/*  ../
