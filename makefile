TestGomoku:
	@echo "#!/bin/bash" > TestGomoku
	@echo "pypy3 test_gomoku.py \"\$$@\"" >> TestGomoku
	@chmod u+x TestGomoku
	@cat report.txt

clean:
	@rm -f TestGomoku
