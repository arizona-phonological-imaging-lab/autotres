
test: clean
	python autotrace.py

clean:
	rm -rf *.hdf5 *.json *.pyc a3/*.pyc *.log __pycache__
