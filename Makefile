all:
	python setup.py build_ext --inplace
	rm -rf build

clean:
	rm -rf nms/*.so
	rm -rf utils/*.so
