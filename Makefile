FILES = SOIKA/*

lint:
	python -m pylint ${FILES}

format:
	python -m black ${FILES}